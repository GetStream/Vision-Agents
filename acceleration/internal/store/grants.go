package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// MaxLiveGrants is the most grants one organization has live at once, so admission's search
// for one stays a short scan.
const MaxLiveGrants = 20

// ErrNoGrant is a grant that is not there, or is another organization's.
var ErrNoGrant = errors.New("store: no such grant")

// ErrTooManyGrants is a grant past MaxLiveGrants live in its organization.
var ErrTooManyGrants = errors.New("store: the organization has as many live grants as it may")

// LimitGrant is a one-time amount one subject may spend past its own spent limits, until it is
// used up, expires or is revoked. What it paid for is the billable cost of the request rows that
// name it (GrantUsed), so nothing here changes as it is drawn.
type LimitGrant struct {
	bun.BaseModel `bun:"table:limit_grants,alias:lg"`

	ID             int64         `bun:"id,pk,autoincrement"`
	OrganizationID string        `bun:"organization_id,notnull"`
	Subject        usage.Subject `bun:"subject,notnull"`
	AppID          string        `bun:"app_id,notnull"`
	APIKey         string        `bun:"api_key,notnull"`
	UserID         string        `bun:"user_id,notnull"`
	Category       string        `bun:"category,notnull"`
	AmountMicros   int64         `bun:"amount_micros,notnull"`
	ExpiresAt      *time.Time    `bun:"expires_at"`
	RevokedAt      *time.Time    `bun:"revoked_at"`
	RevokedBy      string        `bun:"revoked_by,notnull"`
	Reason         string        `bun:"reason,notnull"`
	Actor          string        `bun:"actor,notnull"`
	CreatedAt      time.Time     `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// Ref is the subject the grant is for.
func (g LimitGrant) Ref() usage.SubjectRef {
	return usage.SubjectRef{OrganizationID: g.OrganizationID, Subject: g.Subject, AppID: g.AppID, APIKey: g.APIKey, UserID: g.UserID, Category: g.Category}
}

// LiveAt reports whether the grant can be drawn at now: not revoked, and not expired.
func (g LimitGrant) LiveAt(now time.Time) bool {
	return g.RevokedAt == nil && (g.ExpiresAt == nil || now.Before(*g.ExpiresAt))
}

// grantFields is a grant as its audit rows name it.
func grantFields(g *LimitGrant) map[string]any {
	fields := map[string]any{
		"subject": g.Subject, "organization_id": g.OrganizationID, "amount_micros": g.AmountMicros, "reason": g.Reason,
	}
	for name, value := range map[string]string{"app_id": g.AppID, "api_key": g.APIKey, "user_id": g.UserID, "category": g.Category} {
		if value != "" {
			fields[name] = value
		}
	}
	if g.ExpiresAt != nil {
		fields["expires_at"] = g.ExpiresAt.UTC().Format(time.RFC3339)
	}
	if g.RevokedAt != nil {
		fields["revoked_at"] = g.RevokedAt.UTC().Format(time.RFC3339)
		fields["revoked_by"] = g.RevokedBy
	}
	return fields
}

// LiveGrants is the organizations' grants that are neither revoked nor expired at now.
func (s *Store) LiveGrants(ctx context.Context, organizationIDs []string, now time.Time) ([]LimitGrant, error) {
	grants := []LimitGrant{}
	if len(organizationIDs) == 0 {
		return grants, nil
	}
	err := s.db.NewSelect().Model(&grants).
		Where("organization_id IN (?)", bun.In(organizationIDs)).Where("revoked_at IS NULL").
		Where("expires_at IS NULL OR expires_at > ?", now.UTC()).Order("id").Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: live grants: %w", err))
	}
	return grants, nil
}

// AddGrant writes a grant and its audit row, refusing one past MaxLiveGrants live in the
// organization with ErrTooManyGrants. The count and the write hold the organization's lock, so
// two grants made at once cannot both be the twenty-first.
func (s *Store) AddGrant(ctx context.Context, grant *LimitGrant, audit LimitAudit) error {
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended('limit_grants:' || ?, 0))", grant.OrganizationID); err != nil {
			return err
		}
		live, err := tx.NewSelect().Model((*LimitGrant)(nil)).Where("organization_id = ?", grant.OrganizationID).
			Where("revoked_at IS NULL").Where("expires_at IS NULL OR expires_at > now()").Count(ctx)
		if err != nil {
			return err
		}
		if live >= MaxLiveGrants {
			return ErrTooManyGrants
		}
		if _, err := tx.NewInsert().Model(grant).Returning("*").Exec(ctx); err != nil {
			return err
		}
		return recordAuditEntry(ctx, tx, audit.entry("grant:"+strconv.FormatInt(grant.ID, 10), string(grant.Subject), AuditCreated,
			fieldChanges(nil, grantFields(grant))))
	})
	if errors.Is(err, ErrTooManyGrants) {
		return ErrTooManyGrants
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: add grant: %w", err))
	}
	return nil
}

// Grant is one grant by id, of the organization, or of any when organizationID is empty.
func (s *Store) Grant(ctx context.Context, organizationID string, id int64) (LimitGrant, error) {
	return grantOf(ctx, s.db, organizationID, id, false)
}

func grantOf(ctx context.Context, db bun.IDB, organizationID string, id int64, lock bool) (LimitGrant, error) {
	var grant LimitGrant
	query := db.NewSelect().Model(&grant).Where("id = ?", id)
	if organizationID != "" {
		query = query.Where("organization_id = ?", organizationID)
	}
	if lock {
		query = query.For("UPDATE")
	}
	err := query.Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return LimitGrant{}, ErrNoGrant
	}
	if err != nil {
		return LimitGrant{}, stack.Wrap(fmt.Errorf("store: grant: %w", err))
	}
	return grant, nil
}

// RevokeGrant ends a grant of the organization, or of any when organizationID is empty, and
// records it with the reason. Revoking one already revoked changes nothing and records
// nothing. The grant and what it paid for stay, for the history.
func (s *Store) RevokeGrant(ctx context.Context, organizationID string, id int64, by, reason string, audit LimitAudit) (LimitGrant, error) {
	var revoked LimitGrant
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		before, err := grantOf(ctx, tx, organizationID, id, true)
		if err != nil {
			return err
		}
		revoked = before
		if before.RevokedAt != nil {
			return nil
		}
		if _, err := tx.NewUpdate().Model(&revoked).Set("revoked_at = now()").Set("revoked_by = ?", by).
			WherePK().Returning("*").Exec(ctx); err != nil {
			return err
		}
		if audit.CustomerID == "" {
			audit.CustomerID = revoked.OrganizationID
		}
		after := grantFields(&revoked)
		after["revoke_reason"] = reason
		return recordAuditEntry(ctx, tx, audit.entry("grant:"+strconv.FormatInt(id, 10), string(revoked.Subject), AuditUpdated,
			fieldChanges(grantFields(&before), after)))
	})
	if errors.Is(err, ErrNoGrant) {
		return LimitGrant{}, ErrNoGrant
	}
	if err != nil {
		return LimitGrant{}, stack.Wrap(fmt.Errorf("store: revoke grant: %w", err))
	}
	return revoked, nil
}

// GrantFilter picks one organization's grants. Each empty field is every grant.
type GrantFilter struct {
	OrganizationID string
	Subject        usage.Subject
	AppID          string
	APIKey         string
	UserID         string
	Category       string
	// CategoryPrefix and NotCategoryPrefix keep the grants whose category does, or does not,
	// start with them.
	CategoryPrefix    string
	NotCategoryPrefix string
	// Revoked, when set, keeps only the revoked grants or only the others; Expired, likewise,
	// the grants expired at now or the others.
	Revoked *bool
	Expired *bool
	Now     time.Time
}

// ListGrants lists an organization's grants newest first, before the id given when it is not
// 0, LimitPage(limit) and one more. Revoked and expired grants are listed too.
func (s *Store) ListGrants(ctx context.Context, filter GrantFilter, before int64, limit int) ([]LimitGrant, error) {
	if filter.OrganizationID == "" {
		return nil, stack.Wrap(errors.New("store: listing grants needs an organization"))
	}
	grants := []LimitGrant{}
	query := s.db.NewSelect().Model(&grants).Where("organization_id = ?", filter.OrganizationID)
	for column, value := range map[string]string{
		"subject": string(filter.Subject), "app_id": filter.AppID, "api_key": filter.APIKey, "user_id": filter.UserID, "category": filter.Category,
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
	if filter.Revoked != nil {
		if *filter.Revoked {
			query = query.Where("revoked_at IS NOT NULL")
		} else {
			query = query.Where("revoked_at IS NULL")
		}
	}
	if filter.Expired != nil {
		if *filter.Expired {
			query = query.Where("expires_at IS NOT NULL AND expires_at <= ?", filter.Now.UTC())
		} else {
			query = query.Where("expires_at IS NULL OR expires_at > ?", filter.Now.UTC())
		}
	}
	if before > 0 {
		query = query.Where("id < ?", before)
	}
	if err := query.Order("id DESC").Limit(LimitPage(limit) + 1).Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list grants: %w", err))
	}
	return grants, nil
}

// GrantUsed is what each grant has paid for: the billable cost of every request row naming it,
// in every shard. A grant no row names is left out. It reads requests_limit_grants_idx.
func (s *Store) GrantUsed(ctx context.Context, grantIDs []int64) (map[int64]int64, error) {
	used := map[int64]int64{}
	if len(grantIDs) == 0 {
		return used, nil
	}
	type sum struct {
		Grant  int64 `bun:"grant"`
		Micros int64 `bun:"micros"`
	}
	answers, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]sum, error) {
		var sums []sum
		err := db.NewRaw("SELECT g AS grant, COALESCE(SUM(r.cost_micros), 0) AS micros "+
			"FROM requests r, unnest(r.limit_grants) g "+
			"WHERE r.limit_grants && ?::bigint[] AND r.limit_grants <> '{}' AND r.billable AND g = ANY(?::bigint[]) "+
			"GROUP BY g", pgdialect.Array(grantIDs), pgdialect.Array(grantIDs)).Scan(ctx, &sums)
		return sums, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: grant used: %w", err))
	}
	for _, shard := range answers {
		for _, each := range shard {
			used[each.Grant] += each.Micros
		}
	}
	return used, nil
}
