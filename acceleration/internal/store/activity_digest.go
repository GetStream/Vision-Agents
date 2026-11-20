package store

import (
	"context"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// OrganizationDayArea is what one organization's apps did in one area of the API from one
// client on one UTC day.
type OrganizationDayArea struct {
	OrganizationID string `bun:"organization_id"`
	// FirstDay says the organization has no activity on any earlier day.
	FirstDay bool `bun:"first_day"`
	ActivityArea
}

// DayActivity is what every organization's apps did on day, a UTC day, by organization, area
// and client in that order. Rows that name no organization are left out: they are an app's
// whose organization app_organizations did not hold when the row was written.
func (s *Store) DayActivity(ctx context.Context, day time.Time) ([]OrganizationDayArea, error) {
	date := day.UTC().Format(time.DateOnly)
	var areas []OrganizationDayArea
	err := s.db.NewSelect().
		TableExpr("customer_activity AS ca").
		ColumnExpr("ca.organization_id, ca.area, ca.client, MIN(ca.first_at) AS first_at, MAX(ca.last_at) AS last_at").
		ColumnExpr("SUM(ca.requests) AS requests, SUM(ca.refused) AS refused").
		ColumnExpr("COALESCE((array_agg(ca.last_refusal_code ORDER BY ca.last_refusal_at DESC)"+
			" FILTER (WHERE ca.last_refusal_at IS NOT NULL))[1], '') AS last_refusal_code").
		ColumnExpr("NOT EXISTS (SELECT 1 FROM customer_activity AS earlier"+
			" WHERE earlier.organization_id = ca.organization_id AND earlier.day < ?::date) AS first_day", date).
		Where("ca.day = ?::date", date).
		Where("ca.organization_id <> ''").
		Group("ca.organization_id", "ca.area", "ca.client").
		Order("ca.organization_id", "ca.area", "ca.client").
		Scan(ctx, &areas)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: day activity: %w", err))
	}
	return areas, nil
}

// FirstAnswer is an organization whose apps had a request of an area answered for the first
// time on a day.
type FirstAnswer struct {
	OrganizationID string `bun:"organization_id"`
	// Days is how many days before that day the organization's apps made their first request,
	// of any area, refused ones included.
	Days int `bun:"days"`
}

// FirstAnswers are the organizations whose apps had a request of area answered, not refused,
// on day, a UTC day, and on no earlier day, by organization. Rows that name no organization
// are left out, as in DayActivity.
func (s *Store) FirstAnswers(ctx context.Context, area string, day time.Time) ([]FirstAnswer, error) {
	date := day.UTC().Format(time.DateOnly)
	var answers []FirstAnswer
	err := s.db.NewSelect().
		TableExpr("customer_activity AS ca").
		ColumnExpr("ca.organization_id").
		ColumnExpr("?::date - (SELECT MIN(any_area.day) FROM customer_activity AS any_area"+
			" WHERE any_area.organization_id = ca.organization_id) AS days", date).
		Where("ca.day = ?::date", date).
		Where("ca.area = ?", area).
		Where("ca.requests > ca.refused").
		Where("ca.organization_id <> ''").
		Where("NOT EXISTS (SELECT 1 FROM customer_activity AS earlier"+
			" WHERE earlier.organization_id = ca.organization_id AND earlier.area = ca.area"+
			" AND earlier.day < ca.day AND earlier.requests > earlier.refused)").
		Group("ca.organization_id").
		Order("ca.organization_id").
		Scan(ctx, &answers)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: first answers: %w", err))
	}
	return answers, nil
}

// ActiveOrganizations is how many organizations' apps made a request on day, a UTC day,
// refused ones included.
func (s *Store) ActiveOrganizations(ctx context.Context, day time.Time) (int, error) {
	var active int
	err := s.db.NewSelect().
		TableExpr("customer_activity").
		ColumnExpr("COUNT(DISTINCT organization_id)").
		Where("day = ?::date", day.UTC().Format(time.DateOnly)).
		Where("organization_id <> ''").
		Scan(ctx, &active)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: active organizations: %w", err))
	}
	return active, nil
}

// ClaimActivityDigest claims the activity digest of day, a UTC day, unless it was posted or
// another claim of it is younger than stale, so one whose post failed, or whose router stopped
// before posting it, is posted by a later check. claimed says this call holds the claim, and
// so posts the digest.
func (s *Store) ClaimActivityDigest(ctx context.Context, day time.Time, stale time.Duration) (claimed bool, err error) {
	result, err := s.db.ExecContext(ctx, `
INSERT INTO activity_digests (day) VALUES (?::date)
ON CONFLICT (day) DO UPDATE SET claimed_at = now()
WHERE activity_digests.posted_at IS NULL
    AND activity_digests.claimed_at < now() - ? * interval '1 millisecond'`,
		day.UTC().Format(time.DateOnly), stale.Milliseconds())
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: claim activity digest: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: claim activity digest: %w", err))
	}
	return affected > 0, nil
}

// activityKeepDays is how many UTC days before today's customer_activity and activity_digests
// keep: 13 months and a margin, so staff can set a month beside the same month a year before.
// Kept no longer than that need, as GDPR's storage limitation asks (Art. 5(1)(e),
// https://eur-lex.europa.eu/eli/reg/2016/679/oj).
const activityKeepDays = 400

// PruneActivity deletes the activity rows and activity digests of the UTC days more than
// activityKeepDays before now's, and returns how many rows it deleted.
func (s *Store) PruneActivity(ctx context.Context, now time.Time) (int, error) {
	before := now.UTC().AddDate(0, 0, -activityKeepDays).Format(time.DateOnly)
	var pruned int64
	for _, table := range []string{"customer_activity", "activity_digests"} {
		result, err := s.db.NewDelete().TableExpr(table).Where("day < ?::date", before).Exec(ctx)
		if err != nil {
			return 0, stack.Wrap(fmt.Errorf("store: prune %s: %w", table, err))
		}
		deleted, _ := result.RowsAffected()
		pruned += deleted
	}
	return int(pruned), nil
}

// MarkActivityDigestPosted records that the activity digest of day, a UTC day, was posted.
func (s *Store) MarkActivityDigestPosted(ctx context.Context, day time.Time) error {
	_, err := s.db.ExecContext(ctx, "UPDATE activity_digests SET posted_at = now() WHERE day = ?::date",
		day.UTC().Format(time.DateOnly))
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: mark activity digest posted: %w", err))
	}
	return nil
}
