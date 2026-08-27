package store

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ParkBridge saves what a vendor should be told when the person it is calling answers.
func (s *Store) ParkBridge(ctx context.Context, bridge *CallBridge) error {
	if bridge.Token == "" {
		return errors.New("store: a bridge needs a token")
	}
	if bridge.CustomerID == "" || bridge.Vendor == "" {
		return errors.New("store: a bridge needs a customer and a vendor")
	}
	if bridge.TrunkURI == "" {
		return errors.New("store: a bridge needs somewhere to bridge to")
	}
	if bridge.ExpiresAt.IsZero() {
		return errors.New("store: a bridge needs to expire")
	}
	if bridge.CreatedAt.IsZero() {
		bridge.CreatedAt = time.Now().UTC()
	}

	if _, err := s.db.NewInsert().Model(bridge).Exec(ctx); err != nil {
		return fmt.Errorf("store: park bridge: %w", err)
	}
	return nil
}

// ClaimBridge returns a parked bridge and removes it, so a token works exactly once.
//
// The read and the delete are one statement rather than two, because two vendors retrying
// the same fetch at once would otherwise both be told to bridge, and the second transfer
// would arrive at a trunk the first had already taken.
func (s *Store) ClaimBridge(ctx context.Context, token string) (CallBridge, error) {
	if token == "" {
		return CallBridge{}, errors.New("store: a token is required")
	}

	var claimed []CallBridge
	err := s.db.NewDelete().Model((*CallBridge)(nil)).
		Where("token = ?", token).
		Where("expires_at > ?", time.Now().UTC()).
		Returning("*").
		Scan(ctx, &claimed)
	if err != nil {
		return CallBridge{}, fmt.Errorf("store: claim bridge: %w", err)
	}
	if len(claimed) == 0 {
		// One message for a token that never existed, one that has been used and one that
		// has expired: telling them apart would say whether a guessed token was ever real.
		return CallBridge{}, errors.New("store: that bridge is not waiting to be claimed")
	}
	return claimed[0], nil
}

// SweepBridges removes bridges nobody claimed, which is what a call that rang out leaves
// behind.
func (s *Store) SweepBridges(ctx context.Context) (int64, error) {
	result, err := s.db.NewDelete().Model((*CallBridge)(nil)).
		Where("expires_at <= ?", time.Now().UTC()).
		Exec(ctx)
	if err != nil {
		return 0, fmt.Errorf("store: sweep bridges: %w", err)
	}
	swept, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("store: sweep bridges: %w", err)
	}
	return swept, nil
}
