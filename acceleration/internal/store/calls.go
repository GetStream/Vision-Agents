package store

import (
	"cmp"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

	_, err := s.shard.NewInsert().Model(call).
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

	_, err := s.shard.NewUpdate().Model((*Call)(nil)).
		Set("ended_at = ?", at).
		Where("id = ?", id).
		Where("ended_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish call: %w", err)
	}
	return nil
}

// ChangeCallModels rewrites the targets and voice a running call is on, after it was moved
// onto other models mid-call.
func (s *Store) ChangeCallModels(ctx context.Context, call *Call) error {
	if call.ID == "" {
		return errors.New("store: a call id is required")
	}

	// Set rather than Column: the fields are nullzero, and a target a call no longer runs
	// is the empty string these columns hold, never NULL.
	_, err := s.shard.NewUpdate().Model((*Call)(nil)).
		Set("stt = ?", call.STT).
		Set("tts = ?", call.TTS).
		Set("llm = ?", call.LLM).
		Set("subagent = ?", call.Subagent).
		Set("sts = ?", call.STS).
		Set("voice = ?", call.Voice).
		Where("id = ?", call.ID).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: change call models: %w", err)
	}
	return nil
}

// ReviewCall writes what a model made of the call once it was over.
func (s *Store) ReviewCall(ctx context.Context, customerID, id, summary string, score *int, notes string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a call id are required")
	}

	affected, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*Call)(nil)).
			Set("summary = ?", summary).
			Set("review_score = ?", score).
			Set("review_notes = ?", notes).
			Where("id = ?", id).
			Where("customer_id = ?", customerID).
			Exec(ctx)
	})
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
		return Call{}, stack.Wrap(errors.New("store: a customer and a call id are required"))
	}

	call, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (Call, error) {
		var call Call
		err := db.NewSelect().Model(&call).
			Where("id = ?", id).
			Where("customer_id = ?", customerID).
			Limit(1).
			Scan(ctx)
		return call, err
	})
	if errors.Is(err, sql.ErrNoRows) {
		return Call{}, unknownCall(id)
	}
	if err != nil {
		return Call{}, stack.Wrap(fmt.Errorf("store: call: %w", err))
	}
	return call, nil
}

// CustomerCalls returns a customer's calls, newest first.
func (s *Store) CustomerCalls(ctx context.Context, customerID string, filter CallFilter) ([]Call, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	limit := filter.Limit
	if limit <= 0 {
		limit = defaultCallLimit
	}
	if limit > maxCallLimit {
		limit = maxCallLimit
	}

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]Call, error) {
		query := db.NewSelect().Model((*Call)(nil)).
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
		if filter.MaxReviewScore > 0 {
			query = query.Where("review_score <= ?", filter.MaxReviewScore)
		}
		var calls []Call
		err := query.Scan(ctx, &calls)
		return calls, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer calls: %w", err))
	}
	return merged(lists, func(a, b Call) int { return b.StartedAt.Compare(a.StartedAt) }, limit), nil
}

// CallModelCalls returns every model response in one call, including flow and delegated
// requests, in the order they were sent. TurnID relates each request to its exchange.
func (s *Store) CallModelCalls(ctx context.Context, customerID, agentID, callID string, from time.Time, to *time.Time) ([]Request, error) {
	if customerID == "" || agentID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and an agent id are required"))
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]Request, error) {
		query := db.NewSelect().Model((*Request)(nil)).
			Where("customer_id = ?", customerID).
			Where("agent_id = ?", agentID).
			Where("modality = ?", "llm").
			Where("started_at >= ?", from).
			Order("started_at ASC", "id ASC")
		if callID != "" {
			query = query.Where("call_id = ?", callID)
		}
		if to != nil {
			query = query.Where("started_at <= ?", *to)
		}
		var requests []Request
		err := query.Scan(ctx, &requests)
		return requests, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: call model calls: %w", err))
	}
	return merged(lists, func(a, b Request) int {
		return cmp.Or(a.StartedAt.Compare(b.StartedAt), cmp.Compare(a.ID, b.ID))
	}, 0), nil
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
		return nil, stack.Wrap(errors.New("store: a customer and an agent id are required"))
	}

	type usedAt struct {
		UsedModel
		LastUsed time.Time `bun:"last_used"`
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]usedAt, error) {
		query := db.NewSelect().
			TableExpr("requests").
			ColumnExpr("modality, provider, model, MAX(started_at) AS last_used").
			Where("customer_id = ?", customerID).
			Where("agent_id = ?", agentID).
			Where("success").
			Where("started_at >= ?", from).
			Group("modality", "provider", "model").
			OrderExpr("MAX(started_at) DESC")
		if to != nil {
			query = query.Where("started_at <= ?", *to)
		}
		var used []usedAt
		err := query.Scan(ctx, &used)
		return used, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: call used models: %w", err))
	}

	all := distinct(merged(lists, func(a, b usedAt) int { return b.LastUsed.Compare(a.LastUsed) }, 0),
		func(u usedAt) string { return u.Modality + "\x00" + u.Provider + "\x00" + u.Model })
	var used []UsedModel
	for _, model := range all {
		used = append(used, model.UsedModel)
	}
	return used, nil
}

// CallUsage is what a call spent, summed over every request it made.
type CallUsage struct {
	InputTokens       int64 `bun:"input_tokens"`
	CachedInputTokens int64 `bun:"cached_input_tokens"`
	OutputTokens      int64 `bun:"output_tokens"`
	CostMicros        int64 `bun:"cost_micros"`
	Requests          int64 `bun:"requests"`
}

// RuntimeModality is the modality of the rows that bill the time a session was open, a slice
// at a time. They cost what the session's time cost, and are not requests.
const RuntimeModality = "runtime"

// requestCount counts a query's rows that are requests, leaving runtime out.
const requestCount = "COUNT(*) FILTER (WHERE modality <> '" + RuntimeModality + "')"

// CallUsage totals the work one call paid for. Like the models it used, it is keyed by
// agent and window rather than by call, because a request row carries the Stream call
// rather than the session that recorded it. Runtime is the exception: it is the session's
// (sessionID), so another session of the same agent at the same time is not counted.
//
// Failed requests are counted: a model that read the prompt and then fell over is still
// billed for having read it. The call's runtime is in its cost and not among its requests.
func (s *Store) CallUsage(ctx context.Context, customerID, agentID, sessionID string, from time.Time, to *time.Time) (CallUsage, error) {
	if customerID == "" || agentID == "" {
		return CallUsage{}, stack.Wrap(errors.New("store: a customer and an agent id are required"))
	}

	answers, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (CallUsage, error) {
		query := db.NewSelect().
			TableExpr("requests").
			ColumnExpr("COALESCE(SUM(input_tokens), 0) AS input_tokens").
			ColumnExpr("COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens").
			ColumnExpr("COALESCE(SUM(output_tokens), 0) AS output_tokens").
			ColumnExpr("COALESCE(SUM(cost_micros), 0) AS cost_micros").
			ColumnExpr(requestCount+" AS requests").
			Where("customer_id = ?", customerID).
			Where("agent_id = ?", agentID).
			Where("(modality <> ? OR session_id = ?)", RuntimeModality, sessionID).
			Where("started_at >= ?", from)
		if to != nil {
			query = query.Where("started_at <= ?", *to)
		}
		var spent CallUsage
		err := query.Scan(ctx, &spent)
		return spent, err
	})
	if err != nil {
		return CallUsage{}, stack.Wrap(fmt.Errorf("store: call usage: %w", err))
	}
	var spent CallUsage
	for _, answer := range answers {
		spent.InputTokens += answer.InputTokens
		spent.CachedInputTokens += answer.CachedInputTokens
		spent.OutputTokens += answer.OutputTokens
		spent.CostMicros += answer.CostMicros
		spent.Requests += answer.Requests
	}
	return spent, nil
}

// ModelTokens is what one model read and wrote over a call.
type ModelTokens struct {
	Modality          string     `bun:"modality"`
	Provider          string     `bun:"provider"`
	Model             string     `bun:"model"`
	InputTokens       int64      `bun:"input_tokens"`
	CachedInputTokens int64      `bun:"cached_input_tokens"`
	OutputTokens      int64      `bun:"output_tokens"`
	CostMicros        int64      `bun:"cost_micros"`
	OutputCostMicros  int64      `bun:"output_cost_micros"`
	Requests          int64      `bun:"requests"`
	InputParts        InputParts `bun:"embed:input_"`
}

// CallTokens is what each model a call used read, wrote and cost, the busiest first. Models
// that bill by audio or characters rather than tokens read zero tokens. Keyed by agent and
// window like CallUsage, its runtime the session's, and read while the call is still going
// as well as after: to is nil for one that has not ended. Runtime is no model's requests.
func (s *Store) CallTokens(ctx context.Context, customerID, agentID, sessionID string, from time.Time, to *time.Time) ([]ModelTokens, error) {
	if customerID == "" || agentID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and an agent id are required"))
	}

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]ModelTokens, error) {
		query := db.NewSelect().
			TableExpr("requests").
			ColumnExpr("modality, provider, model").
			ColumnExpr("COALESCE(SUM(input_tokens), 0) AS input_tokens").
			ColumnExpr("COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens").
			ColumnExpr("COALESCE(SUM(output_tokens), 0) AS output_tokens").
			ColumnExpr("COALESCE(SUM(cost_micros), 0) AS cost_micros").
			ColumnExpr("COALESCE(SUM(output_cost_micros), 0) AS output_cost_micros").
			ColumnExpr(requestCount+" AS requests").
			ColumnExpr("COALESCE(SUM(input_instruction_tokens), 0) AS input_instruction_tokens").
			ColumnExpr("COALESCE(SUM(input_message_tokens), 0) AS input_message_tokens").
			ColumnExpr("COALESCE(SUM(input_tool_definition_tokens), 0) AS input_tool_definition_tokens").
			ColumnExpr("COALESCE(SUM(input_tool_use_tokens), 0) AS input_tool_use_tokens").
			ColumnExpr("COALESCE(SUM(input_image_tokens), 0) AS input_image_tokens").
			ColumnExpr("COALESCE(SUM(input_video_tokens), 0) AS input_video_tokens").
			Where("customer_id = ?", customerID).
			Where("agent_id = ?", agentID).
			Where("(modality <> ? OR session_id = ?)", RuntimeModality, sessionID).
			Where("started_at >= ?", from).
			Group("modality", "provider", "model").
			OrderExpr("SUM(input_tokens + output_tokens) DESC, provider, model")
		if to != nil {
			query = query.Where("started_at <= ?", *to)
		}
		var used []ModelTokens
		err := query.Scan(ctx, &used)
		return used, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: call tokens: %w", err))
	}
	if len(lists) == 1 {
		return lists[0], nil
	}

	var order []string
	sums := map[string]*ModelTokens{}
	for _, list := range lists {
		for _, m := range list {
			key := m.Modality + "\x00" + m.Provider + "\x00" + m.Model
			sum := sums[key]
			if sum == nil {
				sum = &ModelTokens{Modality: m.Modality, Provider: m.Provider, Model: m.Model}
				sums[key] = sum
				order = append(order, key)
			}
			sum.InputTokens += m.InputTokens
			sum.CachedInputTokens += m.CachedInputTokens
			sum.OutputTokens += m.OutputTokens
			sum.CostMicros += m.CostMicros
			sum.OutputCostMicros += m.OutputCostMicros
			sum.Requests += m.Requests
			sum.InputParts.InstructionTokens += m.InputParts.InstructionTokens
			sum.InputParts.MessageTokens += m.InputParts.MessageTokens
			sum.InputParts.ToolDefinitionTokens += m.InputParts.ToolDefinitionTokens
			sum.InputParts.ToolUseTokens += m.InputParts.ToolUseTokens
			sum.InputParts.ImageTokens += m.InputParts.ImageTokens
			sum.InputParts.VideoTokens += m.InputParts.VideoTokens
		}
	}
	var used []ModelTokens
	for _, key := range order {
		used = append(used, *sums[key])
	}
	slices.SortStableFunc(used, func(a, b ModelTokens) int {
		return cmp.Or(cmp.Compare(b.InputTokens+b.OutputTokens, a.InputTokens+a.OutputTokens),
			strings.Compare(a.Provider, b.Provider), strings.Compare(a.Model, b.Model))
	})
	return used, nil
}

func unknownCall(id string) error {
	return stack.Wrap(fmt.Errorf("store: there is no call %s", id))
}
