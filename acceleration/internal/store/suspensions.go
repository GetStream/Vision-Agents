package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNotSuspended is an app whose telephony is not suspended.
var ErrNotSuspended = errors.New("store: this app's telephony is not suspended")

// TelephonySuspension returns why an app's telephony is suspended, or ErrNotSuspended.
func (s *Store) TelephonySuspension(ctx context.Context, customerID string) (TelephonySuspension, error) {
	var suspension TelephonySuspension
	err := s.db.NewSelect().Model(&suspension).Where("customer_id = ?", customerID).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return TelephonySuspension{}, stack.Wrap(ErrNotSuspended)
	}
	if err != nil {
		return TelephonySuspension{}, stack.Wrap(fmt.Errorf("store: telephony suspension: %w", err))
	}
	return suspension, nil
}

// SuspendTelephony suspends an app's telephony, replacing the reason of a suspension already
// in place.
func (s *Store) SuspendTelephony(ctx context.Context, suspension *TelephonySuspension) error {
	if suspension.CustomerID == "" || suspension.Reason == "" || suspension.SuspendedBy == "" {
		return stack.Wrap(errors.New("store: a suspension needs a customer, a reason and who made it"))
	}
	suspension.SuspendedAt = time.Now().UTC().Truncate(time.Microsecond)
	_, err := s.db.NewInsert().Model(suspension).
		On("CONFLICT (customer_id) DO UPDATE").
		Set("reason = EXCLUDED.reason").
		Set("suspended_by = EXCLUDED.suspended_by").
		Set("suspended_at = EXCLUDED.suspended_at").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: suspend telephony: %w", err))
	}
	return nil
}

// LiftTelephonySuspension lets an app use the phone again, or returns ErrNotSuspended.
func (s *Store) LiftTelephonySuspension(ctx context.Context, customerID string) error {
	result, err := s.db.NewDelete().Model((*TelephonySuspension)(nil)).
		Where("customer_id = ?", customerID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: lift telephony suspension: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: lift telephony suspension: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(ErrNotSuspended)
	}
	return nil
}
