package store

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// DaySpend is what an organization's billable requests cost in one UTC day.
type DaySpend struct {
	OrganizationID string    `bun:"organization_id"`
	Day            time.Time `bun:"day"`
	CostMicros     int64     `bun:"cost_micros"`
}

// DailySpend is every organization's billable spend in each UTC day its requests started in,
// from from to before to, summed over every shard and ordered by organization and day. A day
// without spend has no row. A request that names no organization yet is left out: the usage
// sweeper writes its organization within a sweep of it.
func (s *Store) DailySpend(ctx context.Context, from, to time.Time) ([]DaySpend, error) {
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]DaySpend, error) {
		var found []DaySpend
		err := db.NewSelect().Model((*Request)(nil)).
			ColumnExpr("organization_id").
			ColumnExpr("(started_at AT TIME ZONE 'UTC')::date AS day").
			ColumnExpr("SUM(cost_micros) AS cost_micros").
			Where("started_at >= ?", from).
			Where("started_at < ?", to).
			Where("billable").
			Where("organization_id <> ''").
			GroupExpr("1, 2").
			Scan(ctx, &found)
		return found, err
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	type key struct {
		organization string
		day          time.Time
	}
	sums := map[key]int64{}
	for _, list := range lists {
		for _, spend := range list {
			sums[key{spend.OrganizationID, spend.Day.UTC()}] += spend.CostMicros
		}
	}
	var all []DaySpend
	for k, cost := range sums {
		all = append(all, DaySpend{OrganizationID: k.organization, Day: k.day, CostMicros: cost})
	}
	slices.SortFunc(all, func(a, b DaySpend) int {
		return cmp.Or(cmp.Compare(a.OrganizationID, b.OrganizationID), a.Day.Compare(b.Day))
	})
	return all, nil
}

// CostAnomaly is a UTC day on which an organization's spend jumped past its own baseline: what
// it had spent that day when the check saw it, and its median day before. One is written per
// organization and day.
type CostAnomaly struct {
	bun.BaseModel `bun:"table:cost_anomaly_notices,alias:can"`

	OrganizationID string     `bun:"organization_id,pk"`
	Day            time.Time  `bun:"day,pk,type:date"`
	SpentMicros    int64      `bun:"spent_micros,notnull"`
	BaselineMicros int64      `bun:"baseline_micros,notnull"`
	CreatedAt      time.Time  `bun:"created_at,nullzero,notnull,default:current_timestamp"`
	ClaimedAt      time.Time  `bun:"claimed_at,nullzero,notnull,default:current_timestamp"`
	PostedAt       *time.Time `bun:"posted_at"`
}

// NoteCostAnomaly writes the anomaly unless one of its organization and day already is.
// written says this call wrote it, and so holds the claim to post it.
func (s *Store) NoteCostAnomaly(ctx context.Context, anomaly CostAnomaly) (written bool, err error) {
	if anomaly.OrganizationID == "" {
		return false, stack.Wrap(errors.New("store: a cost anomaly needs an organization"))
	}
	anomaly.CreatedAt, anomaly.ClaimedAt, anomaly.PostedAt = time.Time{}, time.Time{}, nil
	result, err := s.db.NewInsert().Model(&anomaly).On("CONFLICT DO NOTHING").Exec(ctx)
	if err != nil {
		return false, stack.Wrap(err)
	}
	affected, _ := result.RowsAffected()
	return affected > 0, nil
}

// ClaimUnpostedCostAnomalies claims, and returns, up to limit anomalies never posted whose last
// claim is older than stale, so one whose post failed, or whose router stopped before posting
// it, is posted by the next check.
func (s *Store) ClaimUnpostedCostAnomalies(ctx context.Context, stale time.Duration, limit int) ([]CostAnomaly, error) {
	var anomalies []CostAnomaly
	due := s.db.NewSelect().Model((*CostAnomaly)(nil)).Column("organization_id", "day").
		Where("posted_at IS NULL").
		Where("claimed_at < now() - ? * interval '1 millisecond'", stale.Milliseconds()).
		OrderExpr("claimed_at").Limit(limit).For("UPDATE SKIP LOCKED")
	err := s.db.NewUpdate().Model((*CostAnomaly)(nil)).
		Set("claimed_at = now()").
		Where("(can.organization_id, can.day) IN (?)", due).
		Returning("*").
		Scan(ctx, &anomalies)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return anomalies, nil
}

// MarkCostAnomalyPosted records that the organization's anomaly of the day was posted.
func (s *Store) MarkCostAnomalyPosted(ctx context.Context, organization string, day time.Time) error {
	_, err := s.db.NewUpdate().Model((*CostAnomaly)(nil)).
		Set("posted_at = now()").
		Where("organization_id = ?", organization).
		Where("day = ?::date", day.UTC().Format(time.DateOnly)).
		Exec(ctx)
	return stack.Wrap(err)
}
