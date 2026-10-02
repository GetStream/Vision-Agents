package store

import (
	"context"
	"database/sql"
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

// ReleaseCallResourcesInApp deletes the trunks and routes one call's legs were given in
// one Stream app, and returns them. A call's id is only unique within an app, so an event
// about it releases only what that app holds; unpinned also takes the rows written before
// apps had identities, which were all made in the deployment's own app.
func (s *Store) ReleaseCallResourcesInApp(ctx context.Context, app int64, unpinned bool, callType, callID string) ([]CallResource, error) {
	var released []CallResource
	query := s.db.NewDelete().Model((*CallResource)(nil)).
		Where("call_type = ?", callType).
		Where("call_id = ?", callID)
	switch {
	case unpinned && app != 0:
		query = query.Where("(stream_app_pk = ? OR stream_app_pk IS NULL)", app)
	case unpinned:
		query = query.Where("stream_app_pk IS NULL")
	default:
		query = query.Where("stream_app_pk = ?", app)
	}
	if err := query.Returning("*").Scan(ctx, &released); err != nil {
		return nil, fmt.Errorf("store: release call resources: %w", err)
	}
	return released, nil
}

// CallPin is the Stream app a customer's call has its lines in: the app the trunk carrying
// it was made in, whether an outbound leg's or an attached number's. A session joining the
// call has to be in that app. Found is false for a call with no lines of its own.
func (s *Store) CallPin(ctx context.Context, customerID, callType, callID string) (int64, bool, error) {
	var pin sql.NullInt64
	err := s.db.NewSelect().Model((*CallResource)(nil)).Column("stream_app_pk").
		Where("customer_id = ?", customerID).
		Where("call_type = ?", callType).
		Where("call_id = ?", callID).
		OrderExpr("created_at DESC").
		Limit(1).
		Scan(ctx, &pin)
	if errors.Is(err, sql.ErrNoRows) {
		err = s.db.NewSelect().Model((*PhoneNumber)(nil)).Column("stream_app_pk").
			Where("customer_id = ?", customerID).
			Where("stream_call_type = ?", callType).
			Where("stream_call_id = ?", callID).
			Where("released_at IS NULL").
			Limit(1).
			Scan(ctx, &pin)
	}
	if errors.Is(err, sql.ErrNoRows) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, fmt.Errorf("store: call pin: %w", err)
	}
	return pin.Int64, true, nil
}
