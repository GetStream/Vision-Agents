package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
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
// used up, expires or is revoked. UsedMicros is what it paid for: the billable cost of the
// request rows that name it, added as the writer writes them (AddGrantUses).
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
	UsedMicros     int64         `bun:"used_micros,notnull"`
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

// whereLive keeps the grants that can still be drawn at now: not revoked, not expired, and not
// used up.
func whereLive(query *bun.SelectQuery, now any) *bun.SelectQuery {
	return query.Where("revoked_at IS NULL").Where("expires_at IS NULL OR expires_at > ?", now).
		Where("used_micros < amount_micros")
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

// LiveGrants is the organizations' grants that are neither revoked, expired at now, nor used up.
func (s *Store) LiveGrants(ctx context.Context, organizationIDs []string, now time.Time) ([]LimitGrant, error) {
	grants := []LimitGrant{}
	if len(organizationIDs) == 0 {
		return grants, nil
	}
	query := s.db.NewSelect().Model(&grants).Where("organization_id IN (?)", bun.In(organizationIDs))
	err := whereLive(query, now.UTC()).Order("id").Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: live grants: %w", err))
	}
	return grants, nil
}

// AddGrant writes a grant and its audit row, refusing one past MaxLiveGrants live in the
// organization with ErrTooManyGrants; a used-up grant is not live. The count and the write hold
// the organization's lock, so two grants made at once cannot both be the twenty-first.
func (s *Store) AddGrant(ctx context.Context, grant *LimitGrant, audit LimitAudit) error {
	grant.UsedMicros = 0
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended('limit_grants:' || ?, 0))", grant.OrganizationID); err != nil {
			return err
		}
		query := tx.NewSelect().Model((*LimitGrant)(nil)).Where("organization_id = ?", grant.OrganizationID)
		live, err := whereLive(query, bun.Safe("now()")).Count(ctx)
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
	// the grants expired at now or the others; Used the grants used up or the others.
	Revoked *bool
	Expired *bool
	Used    *bool
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
	if filter.Used != nil {
		if *filter.Used {
			query = query.Where("used_micros >= amount_micros")
		} else {
			query = query.Where("used_micros < amount_micros")
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

// GrantUse is one row counted toward one grant: its billable cost, by its usage id.
type GrantUse struct {
	GrantID int64
	UsageID uuid.UUID
	Micros  int64
}

// GrantUsesOf is what the rows written of requests (written names them by usage id) count
// toward their grants: each billable row with a cost, once per grant it names.
func GrantUsesOf(requests []Request, written []uuid.UUID) []GrantUse {
	wrote := make(map[uuid.UUID]bool, len(written))
	for _, id := range written {
		wrote[id] = true
	}
	var uses []GrantUse
	for _, request := range requests {
		if !wrote[request.UsageID] || !request.Billable || request.CostMicros <= 0 {
			continue
		}
		// Only the first of two rows sent with one usage id was written.
		delete(wrote, request.UsageID)
		for _, grant := range slices.Compact(slices.Sorted(slices.Values(request.LimitGrants))) {
			uses = append(uses, GrantUse{GrantID: grant, UsageID: request.UsageID, Micros: request.CostMicros})
		}
	}
	return uses
}

const (
	// grantUsesChunk is how many uses one statement counts.
	grantUsesChunk = 1_000
	// GrantUsesKept is how long which rows counted toward a grant is kept once it expired or
	// was revoked; after that it counts nothing more, so what it paid for stays as it was.
	GrantUsesKept = 30 * 24 * time.Hour
	// grantUsesPruned is how many of them one statement of PruneGrantUses forgets.
	grantUsesPruned = 5_000
)

// grantEndedBefore is the SQL condition a grant g expired or was revoked before ? seconds ago.
const grantEndedBefore = "least(g.revoked_at, g.expires_at) < now() - make_interval(secs => ?)"

// AddGrantUses counts the rows toward their grants in the shared database: each (grant, usage
// id) once, however often it is sent, added to the grant's used_micros. A grant that is not
// there, or ended more than GrantUsesKept ago, counts nothing. It says how many it had not
// counted before.
func (s *Store) AddGrantUses(ctx context.Context, uses []GrantUse) (int, error) {
	added := 0
	for start := 0; start < len(uses); start += grantUsesChunk {
		chunk := uses[start:min(start+grantUsesChunk, len(uses))]
		err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
			n, err := addGrantUses(ctx, tx, chunk)
			added += n
			return err
		})
		if err != nil {
			return added, stack.Wrap(fmt.Errorf("store: add grant uses: %w", err))
		}
	}
	return added, nil
}

// addGrantUses is AddGrantUses within tx. The grants are locked in id order first, so two
// writers counting rows toward the same grants cannot deadlock.
func addGrantUses(ctx context.Context, tx bun.Tx, uses []GrantUse) (int, error) {
	if len(uses) == 0 {
		return 0, nil
	}
	grants := make([]int64, len(uses))
	usageIDs := make([]string, len(uses))
	micros := make([]int64, len(uses))
	for i, use := range uses {
		grants[i], usageIDs[i], micros[i] = use.GrantID, use.UsageID.String(), use.Micros
	}
	if _, err := tx.ExecContext(ctx, "SELECT id FROM limit_grants WHERE id = ANY(?::bigint[]) ORDER BY id FOR UPDATE",
		pgdialect.Array(slices.Compact(slices.Sorted(slices.Values(grants))))); err != nil {
		return 0, err
	}
	var added int
	err := tx.NewRaw(`
WITH sent AS (
    SELECT * FROM unnest(?::bigint[], ?::uuid[], ?::bigint[]) AS s (grant_id, usage_id, micros)
), counted AS (
    INSERT INTO limit_grant_uses (grant_id, usage_id, micros)
    SELECT s.grant_id, s.usage_id, s.micros FROM sent s JOIN limit_grants g ON g.id = s.grant_id
    WHERE NOT coalesce(`+grantEndedBefore+`, false)
    ON CONFLICT DO NOTHING
    RETURNING grant_id, micros
), raised AS (
    UPDATE limit_grants g SET used_micros = g.used_micros + c.micros
    FROM (SELECT grant_id, sum(micros) AS micros FROM counted GROUP BY grant_id) c
    WHERE g.id = c.grant_id
    RETURNING 1
)
SELECT count(*) FROM counted`, pgdialect.Array(grants), pgdialect.Array(usageIDs), pgdialect.Array(micros), GrantUsesKept.Seconds()).Scan(ctx, &added)
	return added, err
}

// PruneGrantUses forgets which rows counted toward the grants that expired or were revoked
// more than GrantUsesKept ago, grantUsesPruned at a time until none is left; each grant keeps
// what it paid for. It says how many it forgot.
func (s *Store) PruneGrantUses(ctx context.Context) (int64, error) {
	var pruned int64
	for {
		result, err := s.db.ExecContext(ctx, `
DELETE FROM limit_grant_uses WHERE (grant_id, usage_id) IN (
    SELECT u.grant_id, u.usage_id FROM limit_grants g JOIN limit_grant_uses u ON u.grant_id = g.id
    WHERE `+grantEndedBefore+` LIMIT ?)`, GrantUsesKept.Seconds(), grantUsesPruned)
		if err != nil {
			return pruned, stack.Wrap(fmt.Errorf("store: prune grant uses: %w", err))
		}
		n, err := result.RowsAffected()
		pruned += n
		if err != nil || n < grantUsesPruned {
			return pruned, err
		}
	}
}

// GrantUsesCounted is, of the rows named by usage id, how many grants each counts toward.
func (s *Store) GrantUsesCounted(ctx context.Context, usageIDs []uuid.UUID) (map[uuid.UUID]int, error) {
	counted := map[uuid.UUID]int{}
	if len(usageIDs) == 0 {
		return counted, nil
	}
	ids := make([]string, len(usageIDs))
	for i, id := range usageIDs {
		ids[i] = id.String()
	}
	var rows []struct {
		UsageID uuid.UUID `bun:"usage_id"`
		Grants  int       `bun:"grants"`
	}
	err := s.db.NewRaw("SELECT usage_id, count(*) AS grants FROM limit_grant_uses WHERE usage_id = ANY(?::uuid[]) GROUP BY usage_id",
		pgdialect.Array(ids)).Scan(ctx, &rows)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: grant uses counted: %w", err))
	}
	for _, row := range rows {
		counted[row.UsageID] = row.Grants
	}
	return counted, nil
}

const (
	// GrantReconcileOverlap is how far before where it last got the reconcile reads a shard
	// again: a row written at most this long after it started is found. One written later,
	// whose count its writer dropped too, is not.
	GrantReconcileOverlap = time.Hour
	// grantReconcilePage is how many rows one read of a shard takes; where the reconcile got
	// is kept after each, so one cut short goes on from there.
	grantReconcilePage = 2_000
)

// ErrReconcileBusy is ReconcileGrantUses finding another router reconciling.
var ErrReconcileBusy = errors.New("store: another router is reconciling grant uses")

// ReconcileGrantUses counts toward their grants the billable rows naming one, in every shard,
// that are not counted yet: a row whose writer wrote it to its shard and dropped, or stopped
// before, counting it in the shared database. Each shard is read on from where the last
// reconcile of it got (limit_grant_reconciles), less GrantReconcileOverlap, in started_at
// order (requests_limit_grants_started_idx). One router reconciles at a time, under an advisory lock
// in the shared database: another finds ErrReconcileBusy. With one database the writer counts
// every row as it writes it, so there is nothing to do. It says how many it counted.
func (s *Store) ReconcileGrantUses(ctx context.Context) (int, error) {
	if !s.Sharded() {
		return 0, nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: reconcile grant uses: %w", err))
	}
	defer tx.Rollback() //nolint:errcheck // it writes nothing; ending it lets go of the lock
	var locked bool
	if err := tx.QueryRowContext(ctx, "SELECT pg_try_advisory_xact_lock(hashtextextended('limit_grant_reconciles', 0))").Scan(&locked); err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: reconcile grant uses: lock: %w", err))
	}
	if !locked {
		return 0, ErrReconcileBusy
	}
	now := time.Now().UTC()
	shards := s.shards()
	counted := make([]int, len(shards))
	failed := make([]error, len(shards))
	var wg sync.WaitGroup
	for i, shard := range shards {
		wg.Go(func() { counted[i], failed[i] = s.reconcileShard(ctx, shard, s.shardNames[i], now) })
	}
	wg.Wait()
	total := 0
	for _, n := range counted {
		total += n
	}
	if err := errors.Join(failed...); err != nil {
		return total, stack.Wrap(fmt.Errorf("store: reconcile grant uses: %w", err))
	}
	return total, nil
}

// reconcileGrantsQuery is one page of a shard's rows naming a grant, on from (started_at, id),
// read by requests_limit_grants_started_idx.
const reconcileGrantsQuery = "SELECT id, started_at, usage_id, cost_micros, limit_grants FROM requests " +
	"WHERE limit_grants <> '{}' AND (started_at, id) > (?, ?) AND billable AND cost_micros > 0 AND usage_id IS NOT NULL " +
	"ORDER BY started_at, id LIMIT ?"

// reconcileShard is ReconcileGrantUses for one shard, read up to now.
func (s *Store) reconcileShard(ctx context.Context, shard bun.IDB, name string, now time.Time) (int, error) {
	var mark time.Time
	err := s.db.NewRaw("SELECT started_at FROM limit_grant_reconciles WHERE shard = ?", name).Scan(ctx, &mark)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return 0, fmt.Errorf("shard %s: %w", name, err)
	}
	var after time.Time
	var afterID int64
	if !mark.IsZero() {
		after = mark.Add(-GrantReconcileOverlap)
	}
	counted := 0
	for {
		var rows []struct {
			ID          int64     `bun:"id"`
			StartedAt   time.Time `bun:"started_at"`
			UsageID     uuid.UUID `bun:"usage_id"`
			CostMicros  int64     `bun:"cost_micros"`
			LimitGrants []int64   `bun:"limit_grants,array"`
		}
		err := shard.NewRaw(reconcileGrantsQuery, after, afterID, grantReconcilePage).Scan(ctx, &rows)
		if err != nil {
			return counted, fmt.Errorf("shard %s: %w", name, err)
		}
		var uses []GrantUse
		for _, row := range rows {
			for _, grant := range slices.Compact(slices.Sorted(slices.Values(row.LimitGrants))) {
				uses = append(uses, GrantUse{GrantID: grant, UsageID: row.UsageID, Micros: row.CostMicros})
			}
		}
		n, err := s.AddGrantUses(ctx, uses)
		counted += n
		if err != nil {
			return counted, fmt.Errorf("shard %s: %w", name, err)
		}
		reached := now
		if len(rows) == grantReconcilePage {
			last := rows[len(rows)-1]
			after, afterID = last.StartedAt, last.ID
			if last.StartedAt.Before(now) {
				reached = last.StartedAt
			}
		}
		if _, err := s.db.NewRaw("INSERT INTO limit_grant_reconciles (shard, started_at) VALUES (?, ?) "+
			"ON CONFLICT (shard) DO UPDATE SET started_at = GREATEST(limit_grant_reconciles.started_at, EXCLUDED.started_at), updated_at = now()",
			name, reached).Exec(ctx); err != nil {
			return counted, fmt.Errorf("shard %s: keep where it got: %w", name, err)
		}
		if len(rows) < grantReconcilePage {
			return counted, nil
		}
	}
}
