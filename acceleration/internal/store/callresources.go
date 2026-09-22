package store

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// RecordCallResource saves the trunk and route one call leg was given, so the call's end
// can find and delete them.
func (s *Store) RecordCallResource(ctx context.Context, resource *CallResource) error {
	if resource.TrunkID == "" {
		return errors.New("store: a call resource needs a trunk")
	}
	if resource.RouteID == "" {
		return errors.New("store: a call resource needs a route")
	}
	if resource.CallID == "" {
		return errors.New("store: a call resource needs a call")
	}
	if resource.CreatedAt.IsZero() {
		resource.CreatedAt = time.Now().UTC()
	}

	if _, err := s.db.NewInsert().Model(resource).Exec(ctx); err != nil {
		return fmt.Errorf("store: record call resource: %w", err)
	}
	return nil
}

// ReleaseCallResources returns every trunk and route recorded for a call and removes them,
// so a caller can delete them at Stream and a retried delivery of the same event finds
// nothing left to release.
func (s *Store) ReleaseCallResources(ctx context.Context, callType, callID string) ([]CallResource, error) {
	var released []CallResource
	err := s.db.NewDelete().Model((*CallResource)(nil)).
		Where("call_type = ?", callType).
		Where("call_id = ?", callID).
		Returning("*").
		Scan(ctx, &released)
	if err != nil {
		return nil, fmt.Errorf("store: release call resources: %w", err)
	}
	return released, nil
}
