package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// LimitLevel is whose ids a limit names: the subject's own, or a level above it, where the row
// is a default for every subject of its kind there.
type LimitLevel string

const (
	LevelOrganization LimitLevel = "organization"
	LevelApp          LimitLevel = "app"
	LevelAPIKey       LimitLevel = "api_key"
	LevelUser         LimitLevel = "user"
)

// limitLevels is each subject's levels, most specific first, which is the order a subject's
// limit is resolved in (20261112120000_limits.sql, limits_level_check).
var limitLevels = map[usage.Subject][]LimitLevel{
	usage.SubjectOrganization: {LevelOrganization},
	usage.SubjectApp:          {LevelApp},
	usage.SubjectAPIKey:       {LevelAPIKey, LevelApp, LevelOrganization},
	usage.SubjectUser:         {LevelUser, LevelApp, LevelOrganization},
	usage.SubjectCategory:     {LevelOrganization},
	usage.SubjectAppCategory:  {LevelApp},
}

// LimitLevels is the levels a subject can be limited at, most specific first. A subject that
// has none is no subject a limit can name.
func LimitLevels(subject usage.Subject) []LimitLevel {
	return limitLevels[subject]
}

// ErrNoLimit is a limit that is not there, or is another organization's.
var ErrNoLimit = errors.New("store: no such limit")

// ErrLimitExists is a limit written for a subject, level and interval that already have one.
type ErrLimitExists struct{ ID int64 }

func (e *ErrLimitExists) Error() string {
	return "store: the subject already has a limit at that level and interval: " + strconv.FormatInt(e.ID, 10)
}

// Limit is one of the customer's spend limits: what Subject may spend in each Interval window,
// at Level, whose ids it names.
type Limit struct {
	bun.BaseModel `bun:"table:limits,alias:lim"`

	ID             int64          `bun:"id,pk,autoincrement"`
	Subject        usage.Subject  `bun:"subject,notnull"`
	Level          LimitLevel     `bun:"level,notnull"`
	OrganizationID string         `bun:"organization_id,notnull"`
	AppID          string         `bun:"app_id,notnull"`
	APIKey         string         `bun:"api_key,notnull"`
	UserID         string         `bun:"user_id,notnull"`
	Category       string         `bun:"category,notnull"`
	Interval       usage.Interval `bun:"interval,notnull"`
	LimitMicros    int64          `bun:"limit_micros,notnull"`
	Note           string         `bun:"note,notnull"`
	UpdatedBy      string         `bun:"updated_by,notnull"`
	CreatedAt      time.Time      `bun:"created_at,nullzero,notnull,default:current_timestamp"`
	UpdatedAt      time.Time      `bun:"updated_at,nullzero,notnull,default:current_timestamp"`
}

// Covers reports whether the row limits the subject: the same subject and organization, and
// every id the row's level names equal to the subject's.
func (l Limit) Covers(ref usage.SubjectRef) bool {
	if l.Subject != ref.Subject || l.OrganizationID != ref.OrganizationID || l.Category != ref.Category {
		return false
	}
	switch l.Level {
	case LevelApp:
		return l.AppID == ref.AppID
	case LevelAPIKey:
		return l.AppID == ref.AppID && l.APIKey == ref.APIKey
	case LevelUser:
		return l.AppID == ref.AppID && l.UserID == ref.UserID
	}
	return true
}

// UsageReset says that from WindowStart's window on, a subject's limits in Interval count its
// spend less BaselineMicros.
type UsageReset struct {
	bun.BaseModel `bun:"table:usage_resets,alias:ur"`

	ID             int64          `bun:"id,pk,autoincrement"`
	OrganizationID string         `bun:"organization_id,notnull"`
	Subject        usage.Subject  `bun:"subject,notnull"`
	AppID          string         `bun:"app_id,notnull"`
	APIKey         string         `bun:"api_key,notnull"`
	UserID         string         `bun:"user_id,notnull"`
	Category       string         `bun:"category,notnull"`
	Interval       usage.Interval `bun:"interval,notnull"`
	WindowStart    time.Time      `bun:"window_start,notnull"`
	BaselineMicros int64          `bun:"baseline_micros,notnull"`
	Reason         string         `bun:"reason,notnull"`
	Actor          string         `bun:"actor,notnull"`
	CreatedAt      time.Time      `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// Ref is the subject the reset is of.
func (r UsageReset) Ref() usage.SubjectRef {
	return usage.SubjectRef{OrganizationID: r.OrganizationID, Subject: r.Subject, AppID: r.AppID, APIKey: r.APIKey, UserID: r.UserID, Category: r.Category}
}

// LimitAudit is who a write of a limit, a reset or a grant is recorded as made by, which the
// store records in the transaction the write commits in.
type LimitAudit struct {
	CustomerID string
	Source     string
	ActorID    string
	ActorName  string
	RequestID  string
}

func (a LimitAudit) entry(resourceID, name, action string, changes []AuditChange) *AuditEntry {
	return &AuditEntry{
		CustomerID: a.CustomerID, ResourceType: AuditSpendLimit, ResourceID: resourceID, ResourceName: name,
		Action: action, Source: a.Source, ActorID: a.ActorID, ActorName: a.ActorName, RequestID: a.RequestID,
		Changes: changes,
	}
}

// limitFields is a limit as its audit rows name it.
func limitFields(l *Limit) map[string]any {
	if l == nil {
		return map[string]any{}
	}
	fields := map[string]any{
		"subject": l.Subject, "level": l.Level, "organization_id": l.OrganizationID, "interval": l.Interval,
		"limit_micros": l.LimitMicros,
	}
	for name, value := range map[string]string{"app_id": l.AppID, "api_key": l.APIKey, "user_id": l.UserID, "category": l.Category, "note": l.Note} {
		if value != "" {
			fields[name] = value
		}
	}
	return fields
}

// fieldChanges is every field whose value differs between before and after, in name order.
func fieldChanges(before, after map[string]any) []AuditChange {
	names := map[string]bool{}
	for name := range before {
		names[name] = true
	}
	for name := range after {
		names[name] = true
	}
	changes := []AuditChange{}
	for _, name := range slices.Sorted(maps.Keys(names)) {
		was, had := before[name]
		now, has := after[name]
		if had == has && fmt.Sprint(was) == fmt.Sprint(now) {
			continue
		}
		changes = append(changes, AuditChange{Field: name, Before: was, After: now})
	}
	return changes
}

// LimitsFor is every limit of the organizations.
func (s *Store) LimitsFor(ctx context.Context, organizationIDs []string) ([]Limit, error) {
	limits := []Limit{}
	if len(organizationIDs) == 0 {
		return limits, nil
	}
	if err := s.db.NewSelect().Model(&limits).Where("organization_id IN (?)", bun.In(organizationIDs)).Order("id").Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: limits for: %w", err))
	}
	return limits, nil
}

// Limit is one limit by id, of the organization, or of any when organizationID is empty.
func (s *Store) Limit(ctx context.Context, organizationID string, id int64) (Limit, error) {
	return limitOf(ctx, s.db, organizationID, id, false)
}

func limitOf(ctx context.Context, db bun.IDB, organizationID string, id int64, lock bool) (Limit, error) {
	var limit Limit
	query := db.NewSelect().Model(&limit).Where("id = ?", id)
	if organizationID != "" {
		query = query.Where("organization_id = ?", organizationID)
	}
	if lock {
		query = query.For("UPDATE")
	}
	err := query.Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Limit{}, ErrNoLimit
	}
	if err != nil {
		return Limit{}, stack.Wrap(fmt.Errorf("store: limit: %w", err))
	}
	return limit, nil
}

// CreateLimit writes a limit and its audit row. A subject that has one at the level and
// interval already is ErrLimitExists, naming it.
func (s *Store) CreateLimit(ctx context.Context, limit *Limit, audit LimitAudit) error {
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.NewInsert().Model(limit).Returning("*").Exec(ctx); err != nil {
			return err
		}
		return recordAuditEntry(ctx, tx, audit.entry(strconv.FormatInt(limit.ID, 10), string(limit.Subject), AuditCreated,
			fieldChanges(nil, limitFields(limit))))
	})
	if isViolation(err, uniqueViolation) {
		var existing Limit
		found := s.db.NewSelect().Model(&existing).
			Where("organization_id = ?", limit.OrganizationID).Where("subject = ?", limit.Subject).
			Where("level = ?", limit.Level).Where("interval = ?", limit.Interval).
			Where("app_id = ?", limit.AppID).Where("api_key = ?", limit.APIKey).
			Where("user_id = ?", limit.UserID).Where("category = ?", limit.Category).Scan(ctx)
		if found != nil {
			return stack.Wrap(fmt.Errorf("store: create limit: %w", found))
		}
		return &ErrLimitExists{ID: existing.ID}
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: create limit: %w", err))
	}
	return nil
}

// LimitChange is what an update of a limit changes: its amount and its note, nil leaving each
// as it is. What a row limits never changes.
type LimitChange struct {
	LimitMicros *int64
	Note        *string
	UpdatedBy   string
}

// UpdateLimit changes a limit of the organization, or of any when organizationID is empty, and
// records what moved. A change that moves nothing writes nothing.
func (s *Store) UpdateLimit(ctx context.Context, organizationID string, id int64, change LimitChange, audit LimitAudit) (Limit, error) {
	var updated Limit
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		before, err := limitOf(ctx, tx, organizationID, id, true)
		if err != nil {
			return err
		}
		updated = before
		if change.LimitMicros != nil {
			updated.LimitMicros = *change.LimitMicros
		}
		if change.Note != nil {
			updated.Note = *change.Note
		}
		changes := fieldChanges(limitFields(&before), limitFields(&updated))
		if len(changes) == 0 {
			return nil
		}
		updated.UpdatedBy = change.UpdatedBy
		if _, err := tx.NewUpdate().Model(&updated).Column("limit_micros", "note", "updated_by").
			Set("updated_at = now()").WherePK().Returning("*").Exec(ctx); err != nil {
			return err
		}
		return recordAuditEntry(ctx, tx, audit.entry(strconv.FormatInt(id, 10), string(updated.Subject), AuditUpdated, changes))
	})
	if errors.Is(err, ErrNoLimit) {
		return Limit{}, ErrNoLimit
	}
	if err != nil {
		return Limit{}, stack.Wrap(fmt.Errorf("store: update limit: %w", err))
	}
	return updated, nil
}

// DeleteLimit removes a limit of the organization, or of any when organizationID is empty, and
// records it, returning what it was.
func (s *Store) DeleteLimit(ctx context.Context, organizationID string, id int64, audit LimitAudit) (Limit, error) {
	var deleted Limit
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		before, err := limitOf(ctx, tx, organizationID, id, true)
		if err != nil {
			return err
		}
		deleted = before
		if _, err := tx.NewDelete().Model(&before).WherePK().Exec(ctx); err != nil {
			return err
		}
		return recordAuditEntry(ctx, tx, audit.entry(strconv.FormatInt(id, 10), string(before.Subject), AuditDeleted,
			fieldChanges(limitFields(&before), nil)))
	})
	if errors.Is(err, ErrNoLimit) {
		return Limit{}, ErrNoLimit
	}
	if err != nil {
		return Limit{}, stack.Wrap(fmt.Errorf("store: delete limit: %w", err))
	}
	return deleted, nil
}

// LimitFilter picks one organization's limits. Each empty field is every row.
type LimitFilter struct {
	OrganizationID string
	Subject        usage.Subject
	Level          LimitLevel
	AppID          string
	APIKey         string
	UserID         string
	Category       string
	Interval       usage.Interval
	// CategoryPrefix keeps only the rows whose category starts with it, and
	// NotCategoryPrefix only the ones whose category does not.
	CategoryPrefix    string
	NotCategoryPrefix string
}

// How many limits are listed at once.
const (
	defaultLimitPage = 50
	maxLimitPage     = 200
)

// LimitPage is the page size ListLimits uses for the size asked for. It returns one row more.
func LimitPage(asked int) int { return clampLimit(asked, defaultLimitPage, maxLimitPage) }

// ListLimits lists an organization's limits by id, after the id given, LimitPage(limit) and one
// more.
func (s *Store) ListLimits(ctx context.Context, filter LimitFilter, after int64, limit int) ([]Limit, error) {
	if filter.OrganizationID == "" {
		return nil, stack.Wrap(errors.New("store: listing limits needs an organization"))
	}
	limits := []Limit{}
	query := s.db.NewSelect().Model(&limits).Where("organization_id = ?", filter.OrganizationID)
	for column, value := range map[string]string{
		"subject": string(filter.Subject), "level": string(filter.Level), "app_id": filter.AppID, "api_key": filter.APIKey,
		"user_id": filter.UserID, "category": filter.Category, "interval": string(filter.Interval),
	} {
		if value != "" {
			query = query.Where("? = ?", bun.Ident(column), value)
		}
	}
	if filter.CategoryPrefix != "" {
		query = query.Where("starts_with(category, ?)", filter.CategoryPrefix)
	}
	if filter.NotCategoryPrefix != "" {
		query = query.Where("NOT starts_with(category, ?)", filter.NotCategoryPrefix)
	}
	if after > 0 {
		query = query.Where("id > ?", after)
	}
	if err := query.Order("id").Limit(LimitPage(limit) + 1).Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list limits: %w", err))
	}
	return limits, nil
}

// Resets is the organizations' resets of windows that started at since or later.
func (s *Store) Resets(ctx context.Context, organizationIDs []string, since time.Time) ([]UsageReset, error) {
	resets := []UsageReset{}
	if len(organizationIDs) == 0 {
		return resets, nil
	}
	if err := s.db.NewSelect().Model(&resets).Where("organization_id IN (?)", bun.In(organizationIDs)).
		Where("window_start >= ?", since.UTC()).Order("id").Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: resets: %w", err))
	}
	return resets, nil
}

// AddResets writes resets, each with its audit row, in one transaction.
func (s *Store) AddResets(ctx context.Context, resets []UsageReset, audit LimitAudit) error {
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		for i := range resets {
			reset := &resets[i]
			reset.WindowStart = reset.WindowStart.UTC()
			if _, err := tx.NewInsert().Model(reset).Returning("*").Exec(ctx); err != nil {
				return err
			}
			after := map[string]any{
				"subject": reset.Subject, "organization_id": reset.OrganizationID, "interval": reset.Interval,
				"window_start": reset.WindowStart.Format(time.RFC3339), "baseline_micros": reset.BaselineMicros, "reason": reset.Reason,
			}
			for name, value := range map[string]string{"app_id": reset.AppID, "api_key": reset.APIKey, "user_id": reset.UserID, "category": reset.Category} {
				if value != "" {
					after[name] = value
				}
			}
			if err := recordAuditEntry(ctx, tx, audit.entry("reset:"+strconv.FormatInt(reset.ID, 10), string(reset.Subject),
				AuditCreated, fieldChanges(nil, after))); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: add resets: %w", err))
	}
	return nil
}

// SubjectSpendSince sums the billable cost of a subject's rows since a moment, across every
// shard, for a deployment with no usage service to count it. A subject of the whole
// organization (the organization, a category) counts every app seen under it, and the app
// whose id the organization's is, for an app that is its own organization.
func (s *Store) SubjectSpendSince(ctx context.Context, ref usage.SubjectRef, since time.Time) (int64, error) {
	apps := []string{ref.AppID}
	if ref.Subject == usage.SubjectOrganization || ref.Subject == usage.SubjectCategory {
		apps = []string{ref.OrganizationID}
		var members []string
		if err := s.db.NewSelect().Model((*AppOrganization)(nil)).Column("app_id").
			Where("organization_id = ?", ref.OrganizationID).Scan(ctx, &members); err != nil {
			return 0, stack.Wrap(fmt.Errorf("store: subject spend since: %w", err))
		}
		apps = append(apps, members...)
	}
	answers, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (int64, error) {
		var spent int64
		query := db.NewSelect().Model((*Request)(nil)).
			ColumnExpr("COALESCE(SUM(cost_micros), 0)").
			Where("customer_id IN (?)", bun.In(apps)).
			Where("started_at >= ?", since).
			Where("billable")
		switch ref.Subject {
		case usage.SubjectAPIKey:
			query = query.Where("api_key = ?", ref.APIKey)
		case usage.SubjectUser:
			query = query.Where("user_id = ?", ref.UserID)
		case usage.SubjectCategory, usage.SubjectAppCategory:
			query = query.Where("usage_categories @> ARRAY[?]::text[]", ref.Category)
		}
		err := query.Scan(ctx, &spent)
		return spent, err
	})
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: subject spend since: %w", err))
	}
	var spent int64
	for _, answer := range answers {
		spent += answer
	}
	return spent, nil
}
