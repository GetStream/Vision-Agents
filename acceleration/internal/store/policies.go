package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// PolicyScope is what a policy document belongs to.
type PolicyScope string

const (
	// ScopeOrganization is every app in an organization.
	ScopeOrganization PolicyScope = "organization"
	// ScopeApp is one app.
	ScopeApp PolicyScope = "app"
)

// BudgetInterval is how often a spend cap resets. Every interval starts on a UTC boundary,
// so a new interval is a new window and nothing has to be reset.
type BudgetInterval string

const (
	BudgetHourly  BudgetInterval = "hourly"
	BudgetDaily   BudgetInterval = "daily"
	BudgetWeekly  BudgetInterval = "weekly"
	BudgetMonthly BudgetInterval = "monthly"
)

// Valid reports whether the interval is one a budget can reset on.
func (i BudgetInterval) Valid() bool {
	switch i {
	case BudgetHourly, BudgetDaily, BudgetWeekly, BudgetMonthly:
		return true
	}
	return false
}

// Window returns the interval containing now: where it started and where the next one
// starts. A week starts on Monday.
func (i BudgetInterval) Window(now time.Time) (start, end time.Time) {
	now = now.UTC()
	day := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	switch i {
	case BudgetHourly:
		start = now.Truncate(time.Hour)
		return start, start.Add(time.Hour)
	case BudgetWeekly:
		start = day.AddDate(0, 0, -((int(day.Weekday()) + 6) % 7))
		return start, start.AddDate(0, 0, 7)
	case BudgetMonthly:
		start = time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.UTC)
		return start, start.AddDate(0, 1, 0)
	default:
		return day, day.AddDate(0, 0, 1)
	}
}

// Budget caps what a scope may spend in one interval.
type Budget struct {
	LimitMicros int64          `json:"limit_micros"`
	Interval    BudgetInterval `json:"interval"`
}

// PolicyDocument is what an organization or an app has decided.
//
// Every field is optional, and absent means "no opinion" rather than "off": an app that
// says nothing about training inherits whatever its organization said.
type PolicyDocument struct {
	Budget *Budget `json:"budget,omitempty"`
	// DataPolicy is the floor every routed request is held to, on top of whatever the
	// request itself asked for.
	DataPolicy options.DataPolicy `json:"data_policy,omitzero"`
	// PromptInjection screens what an LLM is asked for injection attempts.
	PromptInjection *bool `json:"prompt_injection,omitempty"`
}

// Policy is one scope's stored document.
type Policy struct {
	bun.BaseModel `bun:"table:policies,alias:pol"`

	Scope     PolicyScope    `bun:"scope,pk"`
	ScopeID   string         `bun:"scope_id,pk"`
	Document  PolicyDocument `bun:"document,type:jsonb,notnull"`
	UpdatedAt time.Time      `bun:"updated_at,notnull"`
}

// AppOrganization records which organization an app was last seen under.
type AppOrganization struct {
	bun.BaseModel `bun:"table:app_organizations,alias:ao"`

	AppID          string    `bun:"app_id,pk"`
	OrganizationID string    `bun:"organization_id,notnull"`
	SeenAt         time.Time `bun:"seen_at,notnull"`
}

// Policy returns a scope's document. A scope nobody has written a policy for gets the
// empty document rather than an error, since that is what almost every scope is.
func (s *Store) Policy(ctx context.Context, scope PolicyScope, id string) (PolicyDocument, error) {
	if id == "" {
		return PolicyDocument{}, errors.New("store: a policy needs a scope id")
	}
	var policy Policy
	err := s.db.NewSelect().Model(&policy).
		Where("scope = ?", scope).
		Where("scope_id = ?", id).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PolicyDocument{}, nil
	}
	if err != nil {
		return PolicyDocument{}, fmt.Errorf("store: policy: %w", err)
	}
	return policy.Document, nil
}

// SavePolicy replaces a scope's document.
func (s *Store) SavePolicy(ctx context.Context, scope PolicyScope, id string, document PolicyDocument) error {
	if id == "" {
		return errors.New("store: a policy needs a scope id")
	}
	policy := &Policy{Scope: scope, ScopeID: id, Document: document, UpdatedAt: time.Now().UTC()}
	_, err := s.db.NewInsert().Model(policy).
		On("CONFLICT (scope, scope_id) DO UPDATE").
		Set("document = EXCLUDED.document").
		Set("updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save policy: %w", err)
	}
	return nil
}

// JoinOrganization records that an app was seen under an organization.
func (s *Store) JoinOrganization(ctx context.Context, appID, organizationID string) error {
	if appID == "" || organizationID == "" {
		return errors.New("store: joining an organization needs an app and an organization")
	}
	membership := &AppOrganization{AppID: appID, OrganizationID: organizationID, SeenAt: time.Now().UTC()}
	_, err := s.db.NewInsert().Model(membership).
		On("CONFLICT (app_id) DO UPDATE").
		Set("organization_id = EXCLUDED.organization_id").
		Set("seen_at = EXCLUDED.seen_at").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: join organization: %w", err)
	}
	return nil
}

// OrganizationOf returns the organization an app was last seen under, or nothing for an app
// that has never named one.
func (s *Store) OrganizationOf(ctx context.Context, appID string) (string, error) {
	var membership AppOrganization
	err := s.db.NewSelect().Model(&membership).Where("app_id = ?", appID).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("store: organization of: %w", err)
	}
	return membership.OrganizationID, nil
}

// SpendSince returns what one customer has spent since a moment, across every modality.
func (s *Store) SpendSince(ctx context.Context, customerID string, since time.Time) (int64, error) {
	var spent int64
	err := s.db.NewSelect().Model((*Request)(nil)).
		ColumnExpr("COALESCE(SUM(cost_micros), 0)").
		Where("customer_id = ?", customerID).
		Where("started_at >= ?", since).
		Scan(ctx, &spent)
	if err != nil {
		return 0, fmt.Errorf("store: spend since: %w", err)
	}
	return spent, nil
}

// OrganizationSpendSince returns what every app seen under an organization has spent since
// a moment.
func (s *Store) OrganizationSpendSince(ctx context.Context, organizationID string, since time.Time) (int64, error) {
	var spent int64
	err := s.db.NewSelect().Model((*Request)(nil)).
		ColumnExpr("COALESCE(SUM(cost_micros), 0)").
		Where("customer_id IN (?)", s.db.NewSelect().Model((*AppOrganization)(nil)).
			Column("app_id").Where("organization_id = ?", organizationID)).
		Where("started_at >= ?", since).
		Scan(ctx, &spent)
	if err != nil {
		return 0, fmt.Errorf("store: organization spend since: %w", err)
	}
	return spent, nil
}
