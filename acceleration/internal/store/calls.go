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

// How much of a call's reasoning is handed back at once. A busy call makes a few
// judgements a second, so an hour of one is thousands of rows and a page of them is what
// anybody actually reads.
const (
	defaultCallEventLimit = 1000
	maxCallEventLimit     = 10000
)

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

// UsedModel is a provider/model that successfully served a call.
type UsedModel struct {
	Modality string `bun:"modality"`
	Provider string `bun:"provider"`
	Model    string `bun:"model"`
}

// CallUsedModels returns the distinct provider/models that did work on a call, most
// recently used first. They are keyed by agent and window the way turns are, because a
// request row carries the Stream call rather than the session that recorded it.
func (s *Store) CallUsedModels(ctx context.Context, customerID, agentID string, from time.Time, to *time.Time) ([]UsedModel, error) {
	if customerID == "" || agentID == "" {
		return nil, errors.New("store: a customer and an agent id are required")
	}

	query := s.db.NewSelect().
		TableExpr("requests").
		ColumnExpr("modality, provider, model").
		Where("customer_id = ?", customerID).
		Where("agent_id = ?", agentID).
		Where("success").
		Where("started_at >= ?", from).
		Group("modality", "provider", "model").
		OrderExpr("MAX(started_at) DESC")
	if to != nil {
		query = query.Where("started_at <= ?", *to)
	}

	var used []UsedModel
	if err := query.Scan(ctx, &used); err != nil {
		return nil, fmt.Errorf("store: call used models: %w", err)
	}
	return used, nil
}

// RecordCallEvents writes a batch of judgements. They are written together because they
// arrive together: a call makes several decisions a second, and one round trip each would
// have the writer permanently behind.
func (s *Store) RecordCallEvents(ctx context.Context, events []CallEvent) error {
	if len(events) == 0 {
		return nil
	}
	for i := range events {
		if events[i].CustomerID == "" || events[i].CallID == "" {
			return errors.New("store: a customer and a call id are required")
		}
		if events[i].At.IsZero() {
			events[i].At = time.Now().UTC()
		}
	}

	if _, err := s.db.NewInsert().Model(&events).Exec(ctx); err != nil {
		return fmt.Errorf("store: record call events: %w", err)
	}
	return nil
}

// CallEvents returns the judgements made on one call, oldest first, which read in order
// are how the call was handled and why. They are keyed by the call the agent joined rather
// than by the row recording it, the way turns are, so a call is found by the window it was
// on for.
func (s *Store) CallEvents(ctx context.Context, customerID, callID string, from time.Time, to *time.Time, limit int) ([]CallEvent, error) {
	if customerID == "" || callID == "" {
		return nil, errors.New("store: a customer and a call id are required")
	}
	if limit <= 0 {
		limit = defaultCallEventLimit
	}
	if limit > maxCallEventLimit {
		limit = maxCallEventLimit
	}

	query := s.db.NewSelect().Model((*CallEvent)(nil)).
		Where("customer_id = ?", customerID).
		Where("call_id = ?", callID).
		Where("at >= ?", from).
		Order("at ASC", "id ASC").
		Limit(limit)
	if to != nil {
		query = query.Where("at <= ?", *to)
	}

	var events []CallEvent
	if err := query.Scan(ctx, &events); err != nil {
		return nil, fmt.Errorf("store: call events: %w", err)
	}
	return events, nil
}

func unknownCall(id string) error {
	return fmt.Errorf("store: there is no call %s", id)
}
