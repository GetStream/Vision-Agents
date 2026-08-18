package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// defaultCallLimit caps a call list that did not ask for one.
const defaultCallLimit = 50

// maxCallLimit caps one a caller asked too much of.
const maxCallLimit = 500

// StartCall records that a conversation began.
//
// It is written when the agent joins rather than when the call ends, so a call that is
// still running is findable and one the process died during is not simply lost. Starting
// the same session twice updates the row instead of failing: a retried write must not turn
// into a second call.
func (s *Store) StartCall(ctx context.Context, call *Call) error {
	if call.ID == "" {
		return errors.New("store: a call id is required")
	}
	if call.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if call.StartedAt.IsZero() {
		call.StartedAt = time.Now().UTC()
	}
	if call.Direction == "" {
		call.Direction = Inbound
	}
	if call.Tags == nil {
		call.Tags = map[string]string{}
	}

	_, err := s.db.NewInsert().Model(call).
		On("CONFLICT (id) DO UPDATE").
		Set("ended_at = NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: start call: %w", err)
	}
	return nil
}

// FinishCall records that a conversation ended. A call that already ended keeps the first
// time: the agent leaves once, and a second close is the same leaving reported again.
func (s *Store) FinishCall(ctx context.Context, id string, at time.Time) error {
	if id == "" {
		return errors.New("store: a call id is required")
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}

	_, err := s.db.NewUpdate().Model((*Call)(nil)).
		Set("ended_at = ?", at).
		Where("id = ?", id).
		Where("ended_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish call: %w", err)
	}
	return nil
}

// ReviewCall writes what a model made of the call once it was over.
func (s *Store) ReviewCall(ctx context.Context, customerID, id, summary string, score *int, notes string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a call id are required")
	}

	result, err := s.db.NewUpdate().Model((*Call)(nil)).
		Set("summary = ?", summary).
		Set("review_score = ?", score).
		Set("review_notes = ?", notes).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: review call: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: review call: %w", err)
	}
	if affected == 0 {
		return unknownCall(id)
	}
	return nil
}

// Call returns one call a customer ran.
func (s *Store) Call(ctx context.Context, customerID, id string) (Call, error) {
	if customerID == "" || id == "" {
		return Call{}, errors.New("store: a customer and a call id are required")
	}

	var call Call
	err := s.db.NewSelect().Model(&call).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Call{}, unknownCall(id)
	}
	if err != nil {
		return Call{}, fmt.Errorf("store: call: %w", err)
	}
	return call, nil
}

// CustomerCalls returns a customer's calls, newest first.
func (s *Store) CustomerCalls(ctx context.Context, customerID string, filter CallFilter) ([]Call, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	limit := filter.Limit
	if limit <= 0 {
		limit = defaultCallLimit
	}
	if limit > maxCallLimit {
		limit = maxCallLimit
	}

	query := s.db.NewSelect().Model((*Call)(nil)).
		Where("customer_id = ?", customerID).
		Order("started_at DESC").
		Limit(limit)
	if filter.AgentID != "" {
		query = query.Where("agent_id = ?", filter.AgentID)
	}
	if filter.CampaignID != "" {
		query = query.Where("campaign_id = ?", filter.CampaignID)
	}
	if filter.Running {
		query = query.Where("ended_at IS NULL")
	}
	if !filter.From.IsZero() {
		query = query.Where("started_at >= ?", filter.From)
	}
	if !filter.To.IsZero() {
		query = query.Where("started_at < ?", filter.To)
	}

	var calls []Call
	if err := query.Scan(ctx, &calls); err != nil {
		return nil, fmt.Errorf("store: customer calls: %w", err)
	}
	return calls, nil
}

// CallTurns returns the timings of one call's exchanges, oldest first. They are keyed by
// agent rather than by call, so a call is found by the agent that ran it within the window
// it was on for.
func (s *Store) CallTurns(ctx context.Context, customerID, agentID string, from time.Time, to *time.Time) ([]Turn, error) {
	if customerID == "" || agentID == "" {
		return nil, errors.New("store: a customer and an agent id are required")
	}

	query := s.db.NewSelect().Model((*Turn)(nil)).
		Where("customer_id = ?", customerID).
		Where("agent_id = ?", agentID).
		Where("started_at >= ?", from).
		Order("started_at ASC")
	if to != nil {
		query = query.Where("started_at <= ?", *to)
	}

	var turns []Turn
	if err := query.Scan(ctx, &turns); err != nil {
		return nil, fmt.Errorf("store: call turns: %w", err)
	}
	return turns, nil
}

func unknownCall(id string) error {
	return fmt.Errorf("store: there is no call %s", id)
}
