package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	defaultQuotaChangeLimit = 25
	maxQuotaChangeLimit     = 200
)

// QuotaChangePosition is where a page of quota changes ended: the cursor behind it.
type QuotaChangePosition struct {
	ChangedAt time.Time `json:"c"`
	ID        string    `json:"id"`
}

// QuotaChangeLimit is the page size a quota change list uses for the limit asked for.
func QuotaChangeLimit(asked int) int {
	return clampLimit(asked, defaultQuotaChangeLimit, maxQuotaChangeLimit)
}

// ChangeAppQuota replaces the daily limit in an app's policy and appends the change to
// the record of them, who made it and what it was before, in one transaction: a change is
// either stored and recorded, or neither. The rest of the app's policy is left as it is.
func (s *Store) ChangeAppQuota(ctx context.Context, customerID, changedBy, reason string, quota *Quota) (AppQuotaChange, error) {
	if customerID == "" || changedBy == "" {
		return AppQuotaChange{}, stack.Wrap(errors.New("store: a quota change needs an app and who made it"))
	}
	change := AppQuotaChange{ID: newID(), CustomerID: customerID, ChangedBy: changedBy, Reason: reason}
	if quota != nil {
		change.NewMessagesPerUserPerDay = quota.MessagesPerUserPerDay
		change.NewTokensPerUserPerDay = quota.TokensPerUserPerDay
	}
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		// The row is locked, so two changes of one app are recorded one after the other and
		// the second one's previous numbers are the first one's new ones.
		var policy Policy
		err := tx.NewSelect().Model(&policy).
			Where("scope = ?", ScopeApp).
			Where("scope_id = ?", customerID).
			For("UPDATE").
			Scan(ctx)
		if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return err
		}
		// Taken once the lock is held, so the order of the times is the order of the changes.
		change.ChangedAt = time.Now().UTC().Truncate(time.Microsecond)
		if previous := policy.Document.Quota; previous != nil {
			change.PreviousMessagesPerUserPerDay = previous.MessagesPerUserPerDay
			change.PreviousTokensPerUserPerDay = previous.TokensPerUserPerDay
		}
		policy.Scope, policy.ScopeID, policy.UpdatedAt = ScopeApp, customerID, change.ChangedAt
		policy.Document.Quota = quota
		if _, err := tx.NewInsert().Model(&policy).
			On("CONFLICT (scope, scope_id) DO UPDATE").
			Set("document = EXCLUDED.document").
			Set("updated_at = EXCLUDED.updated_at").
			Exec(ctx); err != nil {
			return err
		}
		_, err = tx.NewInsert().Model(&change).Exec(ctx)
		return err
	})
	if err != nil {
		return AppQuotaChange{}, stack.Wrap(fmt.Errorf("store: change an app's quota: %w", err))
	}
	return change, nil
}

// AppQuotaChanges returns the changes made to an app's daily limit, newest first, one more
// than the page the limit asks for so a caller can tell it is not the last.
func (s *Store) AppQuotaChanges(ctx context.Context, customerID string, limit int, after *QuotaChangePosition) ([]AppQuotaChange, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: a customer id is required"))
	}
	changes := []AppQuotaChange{}
	query := s.db.NewSelect().Model(&changes).Where("customer_id = ?", customerID)
	if after != nil {
		query = query.Where("(changed_at, id) < (?, ?)", after.ChangedAt, after.ID)
	}
	err := query.
		Order("changed_at DESC", "id DESC").
		Limit(QuotaChangeLimit(limit) + 1).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list an app's quota changes: %w", err))
	}
	return changes, nil
}
