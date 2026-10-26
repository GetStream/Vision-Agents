package ledger

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// counted is how many rows of one organization a statement matched.
type counted struct {
	OrganizationID string `bun:"organization_id"`
	Count          int64  `bun:"count"`
}

// ForgetUser replaces an app's end user id with forget.Replacement in every event and in the
// user's own windows, in one transaction, and audits it once per organization, under the
// replacement alone. A window the replacement already has, from a pass before, takes the
// forgotten one's spend. With forget.DryRun it counts the rows and changes nothing.
func (l *Ledger) ForgetUser(ctx context.Context, forget usage.ForgetUser, actor Actor) (usage.ForgottenUser, error) {
	var events, windows []counted
	err := l.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if forget.DryRun {
			if err := tx.NewRaw(`SELECT organization_id, count(*) FROM usage_events WHERE app_id = ? AND user_id = ?
				GROUP BY organization_id`, forget.AppID, forget.UserID).Scan(ctx, &events); err != nil {
				return err
			}
			return tx.NewRaw(`SELECT organization_id, count(*) FROM usage_windows WHERE subject = ? AND app_id = ? AND user_id = ?
				GROUP BY organization_id`, usage.SubjectUser, forget.AppID, forget.UserID).Scan(ctx, &windows)
		}
		if err := tx.NewRaw(`WITH moved AS (
				UPDATE usage_events SET user_id = ? WHERE app_id = ? AND user_id = ? RETURNING organization_id)
			SELECT organization_id, count(*) FROM moved GROUP BY organization_id`,
			forget.Replacement, forget.AppID, forget.UserID).Scan(ctx, &events); err != nil {
			return err
		}
		if err := tx.NewRaw(`WITH moved AS (
				DELETE FROM usage_windows WHERE subject = ? AND app_id = ? AND user_id = ? RETURNING *
			), merged AS (
				INSERT INTO usage_windows AS uw (organization_id, subject, app_id, api_key, user_id, category, interval,
					window_start, window_end, spent_micros, events, updated_at)
				SELECT organization_id, subject, app_id, api_key, ?, category, interval, window_start, window_end,
					spent_micros, events, ?
				FROM moved
				ON CONFLICT (organization_id, subject, app_id, api_key, user_id, category, interval, window_start) DO UPDATE SET
					spent_micros = uw.spent_micros + EXCLUDED.spent_micros, events = uw.events + EXCLUDED.events,
					updated_at = EXCLUDED.updated_at
			)
			SELECT organization_id, count(*) FROM moved GROUP BY organization_id`,
			usage.SubjectUser, forget.AppID, forget.UserID, forget.Replacement, l.now().UTC()).Scan(ctx, &windows); err != nil {
			return err
		}
		return auditForget(ctx, tx, actor, forget, events, windows)
	})
	if err != nil {
		return usage.ForgottenUser{}, fmt.Errorf("ledger: forget user: %w", err)
	}
	var forgotten usage.ForgottenUser
	organizations := map[string]bool{}
	for _, row := range events {
		forgotten.Events += row.Count
		organizations[row.OrganizationID] = true
	}
	for _, row := range windows {
		forgotten.Windows += row.Count
		organizations[row.OrganizationID] = true
	}
	forgotten.Organizations = append([]string{}, slices.Sorted(maps.Keys(organizations))...)
	return forgotten, nil
}

// auditForget records a forgetting once for each organization it rewrote rows of, by the
// replacement and the counts: the forgotten id is what the audit must not keep.
func auditForget(ctx context.Context, tx bun.Tx, actor Actor, forget usage.ForgetUser, events, windows []counted) error {
	type rewritten struct {
		AppID       string `json:"app_id"`
		Replacement string `json:"replacement"`
		Events      int64  `json:"events"`
		Windows     int64  `json:"windows"`
	}
	byOrganization := map[string]*rewritten{}
	of := func(organization string) *rewritten {
		if byOrganization[organization] == nil {
			byOrganization[organization] = &rewritten{AppID: forget.AppID, Replacement: forget.Replacement}
		}
		return byOrganization[organization]
	}
	for _, row := range events {
		of(row.OrganizationID).Events += row.Count
	}
	for _, row := range windows {
		of(row.OrganizationID).Windows += row.Count
	}
	for _, organization := range slices.Sorted(maps.Keys(byOrganization)) {
		if err := audit(ctx, tx, actor, organization, ActionForgetUser, "", nil, byOrganization[organization]); err != nil {
			return err
		}
	}
	return nil
}
