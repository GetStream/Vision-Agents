package store

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/uptrace/bun"
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

// activityBatch is how many rows one statement writes. Each takes 11 parameters, and a
// statement may carry at most 65535 (PostgreSQL's Bind message counts them in an Int16:
// https://www.postgresql.org/docs/current/protocol-message-formats.html), so a thousand
// leave room to spare.
const activityBatch = 1000

// RecordActivity adds deltas to the rows they name, making the ones that are not there yet.
// Counts are added, the first and last times widen, and a refusal later than the one the
// row holds replaces it. Deltas naming the same row are summed first: one statement cannot
// update a row twice.
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

// writeActivity upserts the rows of keys in one statement.
func writeActivity(ctx context.Context, tx bun.Tx, folded map[activityKey]*ActivityDelta, keys []activityKey) error {
	rows := make([]string, 0, len(keys))
	args := make([]any, 0, len(keys)*11)
	for _, key := range keys {
		delta := folded[key]
		var refusedAt *time.Time
		if !delta.LastRefusalAt.IsZero() {
			refusedAt = &delta.LastRefusalAt
		}
		rows = append(rows, "(?, ?, ?, ?, ?::date, ?, ?, NULLIF(?, ''), ?::timestamptz, ?, ?)")
		args = append(args, delta.OrganizationID, key.appID, key.area, key.client, key.day,
			delta.Requests, delta.Refused, delta.LastRefusalCode, refusedAt, delta.FirstAt, delta.LastAt)
	}
	query := `
INSERT INTO customer_activity (organization_id, app_id, area, client, day, requests, refused,
    last_refusal_code, last_refusal_at, first_at, last_at)
VALUES ` + strings.Join(rows, ", ") + `
ON CONFLICT (app_id, area, client, day) DO UPDATE SET
    organization_id = EXCLUDED.organization_id,
    requests = customer_activity.requests + EXCLUDED.requests,
    refused = customer_activity.refused + EXCLUDED.refused,
    last_refusal_code = CASE
        WHEN customer_activity.last_refusal_at IS NULL
            OR EXCLUDED.last_refusal_at >= customer_activity.last_refusal_at
        THEN EXCLUDED.last_refusal_code
        ELSE customer_activity.last_refusal_code END,
    last_refusal_at = GREATEST(customer_activity.last_refusal_at, EXCLUDED.last_refusal_at),
    first_at = LEAST(customer_activity.first_at, EXCLUDED.first_at),
    last_at = GREATEST(customer_activity.last_at, EXCLUDED.last_at)`
	_, err := tx.ExecContext(ctx, query, args...)
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
