package live

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// RecentFor is how long a call's turns and decisions are kept. They explain a call somebody
// is looking at now; what lasts is in agent_logs and the turn stats.
const RecentFor = time.Hour

// maxCallEvents is the most decisions one call keeps, the latest ones, and
// defaultCallEvents how many are read back when the caller names no limit.
const (
	maxCallEvents     = 10_000
	defaultCallEvents = 1000
)

// Turn is one exchange in a conversation, measured the way the caller experienced it.
// The legs are pointers because a pipeline need not have all of them, and an interrupted
// turn stops partway through.
type Turn struct {
	CustomerID string `json:"customer_id"`
	AgentID    string `json:"agent_id"`
	CallID     string `json:"call_id,omitempty"`
	// TurnID is the agent's own identifier for the exchange, unique per agent.
	TurnID             string            `json:"turn_id"`
	Tags               map[string]string `json:"tags,omitempty"`
	StartedAt          time.Time         `json:"started_at"`
	CadenceMs          *float64          `json:"cadence_ms,omitempty"`
	DecisionMs         *float64          `json:"decision_ms,omitempty"`
	ModelToFirstTextMs *float64          `json:"model_to_first_text_ms,omitempty"`
	TextToTTSMs        *float64          `json:"text_to_tts_ms,omitempty"`
	TTSToAudioMs       *float64          `json:"tts_to_audio_ms,omitempty"`
	// ReplyHoldMs is the first-audio silence hold, included in TTSToAudioMs and RoundtripMs.
	ReplyHoldMs *float64 `json:"reply_hold_ms,omitempty"`
	// STTLatencyMs is the provider's decode time for the transcript that settled the turn.
	STTLatencyMs *float64 `json:"stt_latency_ms,omitempty"`
	// LLMTTFTMs is the wait between asking the model and its first token.
	LLMTTFTMs *float64 `json:"llm_ttft_ms,omitempty"`
	// TTSTTFBMs is the wait between sending the first sentence and the first audio.
	TTSTTFBMs *float64 `json:"tts_ttfb_ms,omitempty"`
	// RoundtripMs is the whole delay: settled transcript to first audio published.
	RoundtripMs *float64 `json:"roundtrip_ms,omitempty"`
	// SpeechEndToAudioMs is voice in to voice out.
	SpeechEndToAudioMs *float64 `json:"speech_end_to_audio_ms,omitempty"`
	// FirstFrameQueuedMs is settled transcript to the edge queueing the first frame of the reply,
	// FirstAudibleFrameMs to the track taking the first frame that was not silence, and
	// SpeechEndToAudibleMs is SpeechEndToAudioMs measured to that moment. RoundtripMs stops at
	// publishing returning, which for a long first chunk is later than both.
	FirstFrameQueuedMs   *float64 `json:"first_frame_queued_ms,omitempty"`
	FirstAudibleFrameMs  *float64 `json:"first_audible_frame_ms,omitempty"`
	SpeechEndToAudibleMs *float64 `json:"speech_end_to_audible_ms,omitempty"`
	// AudioOutMs is how much speech the agent published for this turn.
	AudioOutMs *float64 `json:"audio_out_ms,omitempty"`
	// AudioDroppedMs is speech that was synthesised for this turn but never published.
	// Set on a turn that was not interrupted, it means the agent cut itself off.
	AudioDroppedMs *float64 `json:"audio_dropped_ms,omitempty"`
	Interrupted    bool     `json:"interrupted,omitempty"`
}

// CallEvent is one judgement the conversation made about how to handle a call: why the
// agent waited rather than answering, why it read something as not meant for it, why it
// stopped mid-sentence.
type CallEvent struct {
	CustomerID string `json:"customer_id"`
	// CallID is the session the judgement was made in.
	CallID  string    `json:"call_id"`
	AgentID string    `json:"agent_id"`
	At      time.Time `json:"at"`
	// Kind is what was decided.
	Kind string `json:"kind"`
	// Reason is why, in words.
	Reason string `json:"reason"`
	// TurnID is the exchange it was about, so a judgement lines up with the timings of
	// the turn it produced.
	TurnID      string `json:"turn_id,omitempty"`
	Participant string `json:"participant,omitempty"`
	// Said is what was heard, or what the agent decided to say.
	Said string `json:"said,omitempty"`
	// LatencyMs is what the flow controller took to rule, where anything was asked.
	LatencyMs *float64 `json:"latency_ms,omitempty"`
}

// RecordTurn keeps one turn for RecentFor, in one round trip. Turns are kept by agent and
// ordered by when they started, so a call is read back by the agent and the window it ran
// in.
func (c *Client) RecordTurn(ctx context.Context, turn Turn) error {
	if turn.CustomerID == "" || turn.AgentID == "" || turn.TurnID == "" {
		return errors.New("live: a customer, an agent and a turn id are required")
	}
	member, err := json.Marshal(turn)
	if err != nil {
		return fmt.Errorf("live: encode turn: %w", err)
	}

	key := turnsKey(turn.CustomerID, turn.AgentID)
	forgotten := time.Now().Add(-RecentFor).UnixMilli()
	for _, response := range c.redis.DoMulti(ctx,
		c.redis.B().Zadd().Key(key).ScoreMember().ScoreMember(float64(turn.StartedAt.UnixMilli()), string(member)).Build(),
		c.redis.B().Zremrangebyscore().Key(key).Min("-inf").Max("("+strconv.FormatInt(forgotten, 10)).Build(),
		c.redis.B().Pexpire().Key(key).Milliseconds(RecentFor.Milliseconds()).Build(),
	) {
		if err := response.Error(); err != nil {
			return fmt.Errorf("live: record turn: %w", err)
		}
	}
	return nil
}

// CallTurns returns the turns one agent finished between from and to, oldest first. A nil
// to reads up to now. Turns older than RecentFor are gone.
func (c *Client) CallTurns(ctx context.Context, customerID, agentID string, from time.Time, to *time.Time) ([]Turn, error) {
	if customerID == "" || agentID == "" {
		return nil, stack.Wrap(errors.New("live: a customer and an agent id are required"))
	}
	until := "+inf"
	if to != nil {
		until = strconv.FormatInt(to.UnixMilli(), 10)
	}
	members, err := c.redis.Do(ctx, c.redis.B().Zrange().Key(turnsKey(customerID, agentID)).
		Min(strconv.FormatInt(from.UnixMilli(), 10)).Max(until).Byscore().Build()).AsStrSlice()
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("live: call turns: %w", err))
	}

	turns := make([]Turn, 0, len(members))
	for _, member := range members {
		var turn Turn
		if err := json.Unmarshal([]byte(member), &turn); err != nil {
			return nil, stack.Wrap(fmt.Errorf("live: decode turn: %w", err))
		}
		turns = append(turns, turn)
	}
	return turns, nil
}

// RecordCallEvents keeps a batch of judgements for RecentFor after the last one, in one
// round trip. A call keeps its latest maxCallEvents.
func (c *Client) RecordCallEvents(ctx context.Context, events []CallEvent) error {
	if len(events) == 0 {
		return nil
	}
	byCall := map[string][]string{}
	var order []string
	for _, event := range events {
		if event.CustomerID == "" || event.CallID == "" {
			return errors.New("live: a customer and a call id are required")
		}
		if event.At.IsZero() {
			event.At = time.Now().UTC()
		}
		encoded, err := json.Marshal(event)
		if err != nil {
			return fmt.Errorf("live: encode call event: %w", err)
		}
		key := callEventsKey(event.CustomerID, event.CallID)
		if _, seen := byCall[key]; !seen {
			order = append(order, key)
		}
		byCall[key] = append(byCall[key], string(encoded))
	}

	commands := make(rueidis.Commands, 0, 3*len(order))
	for _, key := range order {
		commands = append(commands,
			c.redis.B().Rpush().Key(key).Element(byCall[key]...).Build(),
			c.redis.B().Ltrim().Key(key).Start(-maxCallEvents).Stop(-1).Build(),
			c.redis.B().Pexpire().Key(key).Milliseconds(RecentFor.Milliseconds()).Build(),
		)
	}
	for _, response := range c.redis.DoMulti(ctx, commands...) {
		if err := response.Error(); err != nil {
			return fmt.Errorf("live: record call events: %w", err)
		}
	}
	return nil
}

// CallEvents returns up to limit of the judgements made on one call between from and to,
// oldest first. A nil to reads up to now. Judgements older than RecentFor are gone.
func (c *Client) CallEvents(ctx context.Context, customerID, callID string, from time.Time, to *time.Time, limit int) ([]CallEvent, error) {
	if customerID == "" || callID == "" {
		return nil, stack.Wrap(errors.New("live: a customer and a call id are required"))
	}
	members, err := c.redis.Do(ctx, c.redis.B().Lrange().Key(callEventsKey(customerID, callID)).
		Start(0).Stop(-1).Build()).AsStrSlice()
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("live: call events: %w", err))
	}

	if limit <= 0 {
		limit = defaultCallEvents
	}
	if forgotten := time.Now().Add(-RecentFor); from.Before(forgotten) {
		from = forgotten
	}
	events := make([]CallEvent, 0, len(members))
	for _, member := range members {
		var event CallEvent
		if err := json.Unmarshal([]byte(member), &event); err != nil {
			return nil, stack.Wrap(fmt.Errorf("live: decode call event: %w", err))
		}
		if event.At.Before(from) || (to != nil && event.At.After(*to)) {
			continue
		}
		events = append(events, event)
	}
	slices.SortStableFunc(events, func(a, b CallEvent) int { return a.At.Compare(b.At) })
	return events[:min(len(events), limit)], nil
}

func turnsKey(customerID, agentID string) string {
	return "turns:" + customerID + ":" + agentID
}

func callEventsKey(customerID, callID string) string {
	return "call_events:" + customerID + ":" + callID
}
