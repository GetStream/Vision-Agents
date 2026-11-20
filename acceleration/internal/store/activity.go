package store

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/schema"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ActivityDelta is what one app did in one area of the API, from one client, on one UTC day,
// since the last time it was written: customer_activity's row, as counts to add to it.
type ActivityDelta struct {
	OrganizationID string
	AppID          string
	Area           string
	// Client is one of the audit sources a client may name itself as (AuditSource).
	Client string
	// Day is the UTC day the requests arrived on; only its date is read.
	Day      time.Time
	Requests int64
	Refused  int64
	// LastRefusalCode is the code of the latest refusal counted, and LastRefusalAt when it
	// was; both are empty when none was.
	LastRefusalCode string
	LastRefusalAt   time.Time
	FirstAt         time.Time
	LastAt          time.Time
}

// activityKey is what one customer_activity row is known by.
type activityKey struct {
	appID, area, client, day string
}

// activityBatch is how many rows one statement writes. Each takes 12 parameters, and a
// statement may carry at most 65535 (PostgreSQL's Bind message counts them in an Int16:
// https://www.postgresql.org/docs/current/protocol-message-formats.html), so a thousand
// leave room to spare.
const activityBatch = 1000

// RecordActivity adds deltas to the rows they name, making the ones that are not there yet.
// A delta naming no organization, such as a hook's or a forward's, which name only the app, is
// the organization app_organizations holds for its app, when it holds one. Counts are added,
// the first and last times widen, and a refusal later than the one the row holds replaces it.
// Deltas naming the same row are summed first: one statement cannot update a row twice.
func (s *Store) RecordActivity(ctx context.Context, deltas []ActivityDelta) error {
	folded := make(map[activityKey]*ActivityDelta, len(deltas))
	order := make([]activityKey, 0, len(deltas))
	for _, delta := range deltas {
		key := activityKey{delta.AppID, delta.Area, delta.Client, delta.Day.UTC().Format(time.DateOnly)}
		held, found := folded[key]
		if !found {
			copied := delta
			folded[key] = &copied
			order = append(order, key)
			continue
		}
		held.Add(delta)
	}
	// Every replica writes the rows it shares with another in the same order, so two writes
	// at once wait for each other rather than deadlock.
	slices.SortFunc(order, func(a, b activityKey) int {
		return cmp.Or(cmp.Compare(a.appID, b.appID), cmp.Compare(a.area, b.area),
			cmp.Compare(a.client, b.client), cmp.Compare(a.day, b.day))
	})

	// One transaction, so a failed write wrote nothing and the caller can keep the deltas
	// for its next try without counting any of them twice.
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		for start := 0; start < len(order); start += activityBatch {
			if err := writeActivity(ctx, tx, folded, order[start:min(start+activityBatch, len(order))]); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("store: record activity: %w", err)
	}
	return nil
}

// activityUpsert is one customer_activity row as writeActivity upserts it.
type activityUpsert struct {
	bun.BaseModel `bun:"table:customer_activity,alias:act"`

	OrganizationID  activityOrganization `bun:"organization_id,notnull"`
	AppID           string               `bun:"app_id,pk"`
	Area            string               `bun:"area,pk"`
	Client          string               `bun:"client,pk"`
	Day             string               `bun:"day,pk"`
	Requests        int64                `bun:"requests,notnull"`
	Refused         int64                `bun:"refused,notnull"`
	LastRefusalCode string               `bun:"last_refusal_code,nullzero"`
	LastRefusalAt   time.Time            `bun:"last_refusal_at,nullzero"`
	FirstAt         time.Time            `bun:"first_at,notnull"`
	LastAt          time.Time            `bun:"last_at,notnull"`
}

// activityOrganization is the organization a row is written with: the one its delta names, or,
// when it names none, the one app_organizations holds for its app, or none.
type activityOrganization struct {
	named, appID string
}

// AppendQuery writes o as the SQL expression that picks it, so bun puts it in the row's VALUES.
func (o activityOrganization) AppendQuery(gen schema.QueryGen, b []byte) ([]byte, error) {
	return gen.AppendQuery(b, "COALESCE(NULLIF(?, ''), (SELECT organization_id FROM app_organizations WHERE app_id = ?), '')",
		o.named, o.appID), nil
}

// writeActivity upserts the rows of keys in one statement.
func writeActivity(ctx context.Context, tx bun.Tx, folded map[activityKey]*ActivityDelta, keys []activityKey) error {
	rows := make([]activityUpsert, 0, len(keys))
	for _, key := range keys {
		delta := folded[key]
		rows = append(rows, activityUpsert{
			OrganizationID: activityOrganization{delta.OrganizationID, key.appID},
			AppID:          key.appID, Area: key.area, Client: key.client, Day: key.day,
			Requests: delta.Requests, Refused: delta.Refused, LastRefusalCode: delta.LastRefusalCode,
			LastRefusalAt: delta.LastRefusalAt, FirstAt: delta.FirstAt, LastAt: delta.LastAt,
		})
	}
	_, err := tx.NewInsert().Model(&rows).
		On("CONFLICT (app_id, area, client, day) DO UPDATE").
		Set("organization_id = EXCLUDED.organization_id").
		Set("requests = act.requests + EXCLUDED.requests").
		Set("refused = act.refused + EXCLUDED.refused").
		Set(`last_refusal_code = CASE
        WHEN act.last_refusal_at IS NULL OR EXCLUDED.last_refusal_at >= act.last_refusal_at
        THEN EXCLUDED.last_refusal_code
        ELSE act.last_refusal_code END`).
		Set("last_refusal_at = GREATEST(act.last_refusal_at, EXCLUDED.last_refusal_at)").
		Set("first_at = LEAST(act.first_at, EXCLUDED.first_at)").
		Set("last_at = GREATEST(act.last_at, EXCLUDED.last_at)").
		Exec(ctx)
	return err
}

// Add sums other, a delta for the same row, into d. The organization is the latest one's.
func (d *ActivityDelta) Add(other ActivityDelta) {
	d.Requests += other.Requests
	d.Refused += other.Refused
	if !other.LastRefusalAt.Before(d.LastRefusalAt) {
		d.LastRefusalCode, d.LastRefusalAt = other.LastRefusalCode, other.LastRefusalAt
	}
	if other.FirstAt.Before(d.FirstAt) {
		d.FirstAt = other.FirstAt
	}
	if other.LastAt.After(d.LastAt) {
		d.LastAt = other.LastAt
	}
	d.OrganizationID = other.OrganizationID
}

// ActivityArea is what some apps did in one area of the API from one client, summed over the
// days read.
type ActivityArea struct {
	Area     string    `bun:"area"`
	Client   string    `bun:"client"`
	FirstAt  time.Time `bun:"first_at"`
	LastAt   time.Time `bun:"last_at"`
	Requests int64     `bun:"requests"`
	Refused  int64     `bun:"refused"`
	// LastRefusalCode is the code of the latest refusal, empty when there was none.
	LastRefusalCode string `bun:"last_refusal_code"`
}

// ActivityAreas sums the activity of apps, at least one, from the UTC day of since on, by area
// and client. The record keeps days, so the day since falls on is read whole.
func (s *Store) ActivityAreas(ctx context.Context, apps []string, since time.Time) ([]ActivityArea, error) {
	var areas []ActivityArea
	err := s.db.NewSelect().
		TableExpr("customer_activity").
		ColumnExpr("area, client, MIN(first_at) AS first_at, MAX(last_at) AS last_at").
		ColumnExpr("SUM(requests) AS requests, SUM(refused) AS refused").
		ColumnExpr("COALESCE((array_agg(last_refusal_code ORDER BY last_refusal_at DESC)"+
			" FILTER (WHERE last_refusal_at IS NOT NULL))[1], '') AS last_refusal_code").
		Where("app_id IN (?)", bun.In(apps)).
		Where("day >= ?::date", since.UTC().Format(time.DateOnly)).
		Group("area", "client").
		Order("area", "client").
		Scan(ctx, &areas)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: activity areas: %w", err))
	}
	return areas, nil
}

// ActivityDays is how many UTC days from the day of since on apps had a request answered, not
// refused, in any of areas.
func (s *Store) ActivityDays(ctx context.Context, apps []string, since time.Time, areas []string) (int, error) {
	var days int
	err := s.db.NewSelect().
		TableExpr("customer_activity").
		ColumnExpr("COUNT(DISTINCT day)").
		Where("app_id IN (?)", bun.In(apps)).
		Where("day >= ?::date", since.UTC().Format(time.DateOnly)).
		Where("area IN (?)", bun.In(areas)).
		Where("requests > refused").
		Scan(ctx, &days)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: activity days: %w", err))
	}
	return days, nil
}

// MeterUse is what some apps' work of one modality came to. Requests leaves out the sliced
// time of runtime and sandboxes, which DurationMs measures instead.
type MeterUse struct {
	Modality     string `bun:"modality"`
	Requests     int64  `bun:"requests"`
	InputTokens  int64  `bun:"input_tokens"`
	OutputTokens int64  `bun:"output_tokens"`
	AudioMs      int64  `bun:"audio_ms"`
	Characters   int64  `bun:"characters"`
	DurationMs   int64  `bun:"duration_ms"`
	CostMicros   int64  `bun:"cost_micros"`
}

// MeterShards names the shards MeterUses reads, this one's first: those in Options.Shards, or
// this shard alone, named by a digest of its DSN, when that is unset.
func (s *Store) MeterShards() []string { return slices.Clone(s.shardNames) }

// MeterUses sums the requests apps, at least one, made from since on by modality, over every
// shard, in the order of their modality.
func (s *Store) MeterUses(ctx context.Context, apps []string, since time.Time) ([]MeterUse, error) {
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]MeterUse, error) {
		var used []MeterUse
		err := db.NewSelect().
			TableExpr("requests").
			ColumnExpr("modality").
			ColumnExpr(requestCount+" AS requests").
			ColumnExpr("COALESCE(SUM(input_tokens), 0) AS input_tokens").
			ColumnExpr("COALESCE(SUM(output_tokens), 0) AS output_tokens").
			ColumnExpr("COALESCE(SUM(audio_ms), 0) AS audio_ms").
			ColumnExpr("COALESCE(SUM(characters), 0) AS characters").
			ColumnExpr("COALESCE(SUM(duration_ms), 0)::bigint AS duration_ms").
			ColumnExpr("COALESCE(SUM(cost_micros), 0) AS cost_micros").
			Where("customer_id IN (?)", bun.In(apps)).
			Where("started_at >= ?", since).
			Group("modality").
			Scan(ctx, &used)
		return used, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: meter uses: %w", err))
	}
	// Sorted first, so the rows of one modality from every shard are next to each other.
	var all []MeterUse
	for _, list := range lists {
		all = append(all, list...)
	}
	slices.SortStableFunc(all, func(a, b MeterUse) int { return cmp.Compare(a.Modality, b.Modality) })
	var uses []MeterUse
	for _, used := range all {
		if last := len(uses) - 1; last >= 0 && uses[last].Modality == used.Modality {
			held := &uses[last]
			held.Requests += used.Requests
			held.InputTokens += used.InputTokens
			held.OutputTokens += used.OutputTokens
			held.AudioMs += used.AudioMs
			held.Characters += used.Characters
			held.DurationMs += used.DurationMs
			held.CostMicros += used.CostMicros
			continue
		}
		uses = append(uses, used)
	}
	return uses, nil
}
