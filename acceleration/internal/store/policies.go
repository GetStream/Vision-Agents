package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// PolicyScope is what a policy document belongs to.
type PolicyScope string

const (
	// ScopeOrganization is every app in an organization.
	ScopeOrganization PolicyScope = "organization"
	// ScopeApp is one app.
	ScopeApp PolicyScope = "app"
	// ScopeUser is one end user of one app, by UserScopeID. Its document holds a quota only.
	ScopeUser PolicyScope = "user"
)

// UserScopeID is the scope id of one app's end user. A user id is the app's own name for
// somebody, so the same id in two apps is two users.
func UserScopeID(appID, userID string) string { return appID + "/" + userID }

// SplitUserScopeID is the app and the user a UserScopeID names. A user id holds no '/', so
// the last one splits them.
func SplitUserScopeID(id string) (appID, userID string, ok bool) {
	slash := strings.LastIndexByte(id, '/')
	if slash <= 0 || slash == len(id)-1 {
		return "", "", false
	}
	return id[:slash], id[slash+1:], true
}

// PolicyDocument is what an organization, an app or one end user has decided. Spend limits
// are not here: they are rows of their own (Limit), at every level.
//
// Every field is optional, and absent means "no opinion" rather than "off": an app that
// says nothing about training inherits whatever its organization said.
type PolicyDocument struct {
	// Quota is what one end user may spend in a UTC day: on an organization's document the
	// default for every user of its apps, on an app's the default for its users, and on a
	// user's their own.
	Quota *Quota `json:"quota,omitempty"`
	// DataPolicy is the floor every routed request is held to, on top of whatever the
	// request itself asked for.
	DataPolicy options.DataPolicy `json:"data_policy,omitzero"`
	// PromptInjection screens what an LLM is asked for injection attempts.
	PromptInjection *bool `json:"prompt_injection,omitempty"`
	// AllowedModels are the "provider/model" names requests may be routed to. Nil allows
	// every model, and an empty list allows none.
	AllowedModels *[]string `json:"allowed_models,omitempty"`
	// Tags are laid over the request's own on every row of usage.
	Tags map[string]string `json:"tags,omitempty"`
	// RequireOwnStreamApp keeps an app out of the deployment's own Stream app: it is never
	// written there for want of one of its own, and what it wrote there before can only be
	// read. An organization's is set only by the operator.
	RequireOwnStreamApp *bool `json:"require_own_stream_app,omitempty"`
	// BalanceWarningMicros is the balance, in micros, at or below which the organization is
	// warned, over its tier's; on an organization's document only.
	BalanceWarningMicros *int64 `json:"balance_warning_micros,omitempty"`
	// ToolOutputRetention is the longest any of the scope's sessions keeps what its tools
	// were given and gave back. An agent config can shorten it and never lengthen it.
	ToolOutputRetention options.Retention `json:"tool_output_retention,omitempty"`
}

// Quota is what one end user may spend in a UTC day. A nil field is no opinion, so the
// next level's number shows through, and the deployment's below them all; zero is no cap.
type Quota struct {
	MessagesPerUserPerDay *int64 `json:"messages_per_user_per_day,omitempty"`
	TokensPerUserPerDay   *int64 `json:"tokens_per_user_per_day,omitempty"`
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

// OrganizationTier is the tier Stream last sent for an organization, an opaque string, kept for
// work no request carried.
type OrganizationTier struct {
	bun.BaseModel `bun:"table:organization_tiers,alias:ot"`

	OrganizationID string    `bun:"organization_id,pk"`
	Tier           string    `bun:"tier,notnull"`
	SeenAt         time.Time `bun:"seen_at,notnull"`
}

// SetOrganizationTier records the tier an organization was seen with at seenAt. A tier seen
// earlier than the one recorded changes nothing: routers write what they saw in the background,
// and two writes may land in either order. Earlier is by each writing router's clock, so
// routers whose clocks disagree by more than the gap between two tiers may keep the older.
func (s *Store) SetOrganizationTier(ctx context.Context, organizationID, tier string, seenAt time.Time) error {
	if organizationID == "" || tier == "" {
		return errors.New("store: an organization's tier needs an organization and a tier")
	}
	seen := &OrganizationTier{OrganizationID: organizationID, Tier: tier, SeenAt: seenAt.UTC()}
	_, err := s.db.NewInsert().Model(seen).
		On("CONFLICT (organization_id) DO UPDATE").
		Set("tier = EXCLUDED.tier").
		Set("seen_at = EXCLUDED.seen_at").
		Where("ot.seen_at < EXCLUDED.seen_at").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set organization tier: %w", err))
	}
	return nil
}

// TierOf returns the tier an organization was last seen with, or nothing for one never seen
// with any.
func (s *Store) TierOf(ctx context.Context, organizationID string) (string, error) {
	var seen OrganizationTier
	err := s.db.NewSelect().Model(&seen).Where("organization_id = ?", organizationID).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("store: tier of: %w", err))
	}
	return seen.Tier, nil
}

// Policy returns a scope's document. A scope nobody has written a policy for gets the
// empty document rather than an error, since that is what almost every scope is.
func (s *Store) Policy(ctx context.Context, scope PolicyScope, id string) (PolicyDocument, error) {
	if id == "" {
		return PolicyDocument{}, stack.Wrap(errors.New("store: a policy needs a scope id"))
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
		return PolicyDocument{}, stack.Wrap(fmt.Errorf("store: policy: %w", err))
	}
	return policy.Document, nil
}

// SavePolicy replaces a scope's document. An organization's document that leaves
// RequireOwnStreamApp out keeps the one stored.
func (s *Store) SavePolicy(ctx context.Context, scope PolicyScope, id string, document PolicyDocument) error {
	if id == "" {
		return stack.Wrap(errors.New("store: a policy needs a scope id"))
	}
	policy := &Policy{Scope: scope, ScopeID: id, Document: document, UpdatedAt: time.Now().UTC()}
	replace := "document = EXCLUDED.document"
	if scope == ScopeOrganization {
		// The operator sets an organization's require_own_stream_app, and anybody else saves
		// its document without it. The stored value is kept in the same statement that writes
		// the rest, so a save that read the document before the operator wrote cannot undo it.
		replace = "document = jsonb_strip_nulls(jsonb_build_object('require_own_stream_app', " +
			"pol.document->'require_own_stream_app')) || EXCLUDED.document"
	}
	_, err := s.db.NewInsert().Model(policy).
		On("CONFLICT (scope, scope_id) DO UPDATE").
		Set(replace).
		Set("updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save policy: %w", err))
	}
	return nil
}

// DeletePolicy removes a scope's document, and reports whether there was one.
func (s *Store) DeletePolicy(ctx context.Context, scope PolicyScope, id string) (bool, error) {
	if id == "" {
		return false, stack.Wrap(errors.New("store: a policy needs a scope id"))
	}
	result, err := s.db.NewDelete().Model((*Policy)(nil)).
		Where("scope = ?", scope).
		Where("scope_id = ?", id).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: delete policy: %w", err))
	}
	deleted, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: delete policy: %w", err))
	}
	return deleted > 0, nil
}

// UserPolicyLimit is the default and the most users UserPolicies answers in one page.
const UserPolicyLimit = 100

// UserPolicies is a page of an app's users with a policy of their own, by user id in byte
// order, after the user id after when it is not empty. It reads limit+1 rows, so a caller
// can tell whether there is more.
func (s *Store) UserPolicies(ctx context.Context, appID, after string, limit int) ([]Policy, error) {
	if appID == "" {
		return nil, stack.Wrap(errors.New("store: an app's users need an app"))
	}
	if limit <= 0 || limit > UserPolicyLimit {
		limit = UserPolicyLimit
	}
	var policies []Policy
	query := s.db.NewSelect().Model(&policies).
		Where("scope = ?", ScopeUser).
		Where(`(scope_id COLLATE "C") LIKE ? ESCAPE '\'`, likePrefix(UserScopeID(appID, ""))).
		OrderExpr(`scope_id COLLATE "C"`).
		Limit(limit + 1)
	if after != "" {
		query = query.Where(`(scope_id COLLATE "C") > ?`, UserScopeID(appID, after))
	}
	if err := query.Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: user policies: %w", err))
	}
	return policies, nil
}

// HasUserPolicies reports whether any user of an app has a policy of their own.
func (s *Store) HasUserPolicies(ctx context.Context, appID string) (bool, error) {
	exists, err := s.db.NewSelect().Model((*Policy)(nil)).
		Where("scope = ?", ScopeUser).
		Where(`(scope_id COLLATE "C") LIKE ? ESCAPE '\'`, likePrefix(UserScopeID(appID, ""))).
		Exists(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: has user policies: %w", err))
	}
	return exists, nil
}

// likePrefix is a LIKE pattern matching what starts with prefix, its own % and _ taken as
// themselves.
func likePrefix(prefix string) string {
	return strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`).Replace(prefix) + "%"
}

// AppInOrganization reports whether an app is the organization's: registered under it, or
// last seen under it by a request.
func (s *Store) AppInOrganization(ctx context.Context, appID, organizationID string) (bool, error) {
	if appID == "" || organizationID == "" {
		return false, nil
	}
	seen, err := s.OrganizationOf(ctx, appID)
	if err != nil {
		return false, err
	}
	if seen != "" {
		return seen == organizationID, nil
	}
	registered, err := s.db.NewSelect().Model((*App)(nil)).
		Where("id = ?", appID).
		Where("organization_id = ?", organizationID).
		Exists(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: app in organization: %w", err))
	}
	return registered, nil
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
		return "", stack.Wrap(fmt.Errorf("store: organization of: %w", err))
	}
	return membership.OrganizationID, nil
}

// KnownApps is which of ids this deployment knows: an app registered here, or one a request
// named with its organization. An id it has never seen is another deployment's.
func (s *Store) KnownApps(ctx context.Context, ids []string) (map[string]bool, error) {
	known := make(map[string]bool, len(ids))
	if len(ids) == 0 {
		return known, nil
	}
	var registered, seen []string
	if err := s.db.NewSelect().Model((*App)(nil)).Column("id").Where("id IN (?)", bun.In(ids)).
		Scan(ctx, &registered); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: known apps: %w", err))
	}
	if err := s.db.NewSelect().Model((*AppOrganization)(nil)).Column("app_id").Where("app_id IN (?)", bun.In(ids)).
		Scan(ctx, &seen); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: known apps: %w", err))
	}
	for _, id := range append(registered, seen...) {
		known[id] = true
	}
	return known, nil
}

// SpendSince returns what one customer has spent since a moment, across every modality.
func (s *Store) SpendSince(ctx context.Context, customerID string, since time.Time) (int64, error) {
	spent, err := spendOf(ctx, s, []string{customerID}, since)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: spend since: %w", err))
	}
	return spent, nil
}

// spendOf is what some customers have spent since a moment, in every shard.
func spendOf(ctx context.Context, s *Store, customerIDs []string, since time.Time) (int64, error) {
	answers, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (int64, error) {
		var spent int64
		err := db.NewSelect().Model((*Request)(nil)).
			ColumnExpr("COALESCE(SUM(cost_micros), 0)").
			Where("customer_id IN (?)", bun.In(customerIDs)).
			Where("started_at >= ?", since).
			Scan(ctx, &spent)
		return spent, err
	})
	var spent int64
	for _, answer := range answers {
		spent += answer
	}
	return spent, err
}
