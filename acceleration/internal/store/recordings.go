package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// CreateRecording accepts a job and fills in its id and timestamps. It is written before
// anything is sent to a provider, so a caller handed an id can always ask about it.
func (s *Store) CreateRecording(ctx context.Context, recording *Recording) error {
	if recording.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if recording.Modality == "" {
		return errors.New("store: a recording names the modality it is for")
	}

	recording.ID = newID()
	now := time.Now().UTC()
	recording.CreatedAt = now
	recording.UpdatedAt = now
	recording.Status = RecordingQueued
	if recording.Tags == nil {
		recording.Tags = map[string]string{}
	}

	if _, err := s.db.NewInsert().Model(recording).Exec(ctx); err != nil {
		return fmt.Errorf("store: create recording: %w", err)
	}
	return nil
}

// StartRecording marks a job as at the provider and records which one took it.
func (s *Store) StartRecording(ctx context.Context, id, provider, model string) error {
	_, err := s.db.NewUpdate().Model((*Recording)(nil)).
		Set("status = ?", RecordingRunning).
		Set("provider = ?", provider).
		Set("model = ?", model).
		Set("updated_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: start recording: %w", err)
	}
	return nil
}

// FinishRecording writes the result a job produced, or the reason it produced none.
func (s *Store) FinishRecording(ctx context.Context, id, provider, model string, result json.RawMessage, failure error) error {
	now := time.Now().UTC()
	status := RecordingCompleted
	reason := ""
	if failure != nil {
		status = RecordingFailed
		reason = failure.Error()
	}

	_, err := s.db.NewUpdate().Model((*Recording)(nil)).
		Set("status = ?", status).
		Set("provider = ?", provider).
		Set("model = ?", model).
		Set("result = ?", result).
		Set("error = ?", reason).
		Set("updated_at = ?", now).
		Set("completed_at = ?", now).
		Where("id = ?", id).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish recording: %w", err)
	}
	return nil
}

// Recording returns one job a customer holds.
func (s *Store) Recording(ctx context.Context, customerID, id string) (Recording, error) {
	if customerID == "" || id == "" {
		return Recording{}, errors.New("store: a customer and a recording id are required")
	}

	var recording Recording
	err := s.db.NewSelect().Model(&recording).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Recording{}, fmt.Errorf("store: there is no recording %s", id)
	}
	if err != nil {
		return Recording{}, fmt.Errorf("store: recording: %w", err)
	}
	return recording, nil
}
