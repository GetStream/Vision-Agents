package store

import (
	"context"
	"database/sql"
	"fmt"
	"maps"
	"slices"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// UserSettings are the shared database's rows that name one end user of an app: their own
// policy and their user-level limits, which forgetting them deletes, and their resets, grants
// and the campaigns they scheduled, whose id it replaces.
type UserSettings struct {
	Policies  int64 `bun:"policies"`
	Limits    int64 `bun:"limits"`
	Resets    int64 `bun:"resets"`
	Grants    int64 `bun:"grants"`
	Campaigns int64 `bun:"campaigns"`
}

// Changes are the counts as an audit entry records them.
func (u UserSettings) Changes() []AuditChange {
	return []AuditChange{
		{Field: "policies_deleted", After: u.Policies}, {Field: "limits_deleted", After: u.Limits},
		{Field: "resets", After: u.Resets}, {Field: "grants", After: u.Grants}, {Field: "campaigns", After: u.Campaigns},
	}
}

// UserRequests is how many requests on every shard name an app's end user.
func (s *Store) UserRequests(ctx context.Context, appID, userID string) (int64, error) {
	counts, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (int, error) {
		return db.NewSelect().Model((*Request)(nil)).Where("customer_id = ?", appID).Where("user_id = ?", userID).Count(ctx)
	})
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: user requests: %w", err))
	}
	var total int64
	for _, count := range counts {
		total += int64(count)
	}
	return total, nil
}

// RenameUserRequests replaces an app's end user id with replacement on every shard's requests,
// and answers how many rows it rewrote.
func (s *Store) RenameUserRequests(ctx context.Context, appID, userID, replacement string) (int64, error) {
	renamed, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*Request)(nil)).Set("user_id = ?", replacement).
			Where("customer_id = ?", appID).Where("user_id = ?", userID).Exec(ctx)
	})
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: rename user requests: %w", err))
	}
	return renamed, nil
}

// UserOrganizations are the organizations an app's end user may have counters and holds under:
// those of their requests on every shard, the app's own, and those of their limits, resets and
// grants.
func (s *Store) UserOrganizations(ctx context.Context, appID, userID string) ([]string, error) {
	shards, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]string, error) {
		var found []string
		err := db.NewRaw(`SELECT DISTINCT organization_id FROM requests WHERE customer_id = ? AND user_id = ?`,
			appID, userID).Scan(ctx, &found)
		return found, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: user organizations: %w", err))
	}
	var shared []string
	if err := s.db.NewRaw(`SELECT organization_id FROM app_organizations WHERE app_id = ?
		UNION SELECT organization_id FROM apps WHERE id = ?
		UNION SELECT organization_id FROM limits WHERE level = ? AND app_id = ? AND user_id = ?
		UNION SELECT organization_id FROM usage_resets WHERE app_id = ? AND user_id = ?
		UNION SELECT organization_id FROM limit_grants WHERE app_id = ? AND user_id = ?`,
		appID, appID, LevelUser, appID, userID, appID, userID, appID, userID).Scan(ctx, &shared); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: user organizations: %w", err))
	}
	organizations := map[string]bool{}
	for _, found := range append(shards, shared) {
		for _, organization := range found {
			if organization != "" {
				organizations[organization] = true
			}
		}
	}
	return slices.Sorted(maps.Keys(organizations)), nil
}

// CountUserSettings counts the shared database's rows that name an app's end user.
func (s *Store) CountUserSettings(ctx context.Context, appID, userID string) (UserSettings, error) {
	var counts UserSettings
	err := s.db.NewRaw(`SELECT
		(SELECT count(*) FROM policies WHERE scope = ? AND scope_id = ?) AS policies,
		(SELECT count(*) FROM limits WHERE level = ? AND app_id = ? AND user_id = ?) AS limits,
		(SELECT count(*) FROM usage_resets WHERE app_id = ? AND user_id = ?) AS resets,
		(SELECT count(*) FROM limit_grants WHERE app_id = ? AND user_id = ?) AS grants,
		(SELECT count(*) FROM campaigns WHERE customer_id = ? AND user_id = ?) AS campaigns`,
		ScopeUser, UserScopeID(appID, userID), LevelUser, appID, userID, appID, userID, appID, userID, appID, userID).
		Scan(ctx, &counts)
	if err != nil {
		return UserSettings{}, stack.Wrap(fmt.Errorf("store: count user settings: %w", err))
	}
	return counts, nil
}

// ForgetUserSettings deletes an app's end user's own policy and user-level limits, so they fall
// back to the app's, replaces their id with replacement on their resets, grants and campaigns,
// and records entry, with these counts added to its changes, in the same transaction.
func (s *Store) ForgetUserSettings(ctx context.Context, appID, userID, replacement string, entry *AuditEntry) (UserSettings, error) {
	var counts UserSettings
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		for _, change := range []struct {
			count *int64
			query string
			args  []any
		}{
			{&counts.Policies, "DELETE FROM policies WHERE scope = ? AND scope_id = ?", []any{ScopeUser, UserScopeID(appID, userID)}},
			{&counts.Limits, "DELETE FROM limits WHERE level = ? AND app_id = ? AND user_id = ?", []any{LevelUser, appID, userID}},
			{&counts.Resets, "UPDATE usage_resets SET user_id = ? WHERE app_id = ? AND user_id = ?", []any{replacement, appID, userID}},
			{&counts.Grants, "UPDATE limit_grants SET user_id = ? WHERE app_id = ? AND user_id = ?", []any{replacement, appID, userID}},
			{&counts.Campaigns, "UPDATE campaigns SET user_id = ? WHERE customer_id = ? AND user_id = ?", []any{replacement, appID, userID}},
		} {
			result, err := tx.ExecContext(ctx, change.query, change.args...)
			if err != nil {
				return err
			}
			if *change.count, err = result.RowsAffected(); err != nil {
				return err
			}
		}
		entry.Changes = append(entry.Changes, counts.Changes()...)
		return recordAuditEntry(ctx, tx, entry)
	})
	if err != nil {
		return UserSettings{}, stack.Wrap(fmt.Errorf("store: forget user settings: %w", err))
	}
	return counts, nil
}
