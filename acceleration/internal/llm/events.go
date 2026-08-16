package llm

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
)

// Event is emitted on the channel returned by LLM.Events.
type Event interface {
	isLLMEvent()
}

// CompletionStarted means the provider accepted a request and is working on it.
type CompletionStarted struct {
	CompletionID string
	Provider     string
	Model        string
	At           time.Time
}

func (CompletionStarted) isLLMEvent() {}

// TextDelta is a piece of the answer. Index counts deltas within one completion, so a
// consumer can tell order from arrival order.
//
// A delta does not say whether it is the last one, for the same reason a TTS audio chunk
// does not: waiting to find out would cost the latency the design is for.
// CompletionComplete is what ends a completion.
type TextDelta struct {
	CompletionID string
	Index        int
	Text         string
}

func (TextDelta) isLLMEvent() {}

// ReasoningDelta is a piece of the model's thinking, which reasoning models stream before
// the answer itself. It is kept separate from TextDelta because it must never be spoken or
// shown as the reply, but it is billed as output all the same.
type ReasoningDelta struct {
	CompletionID string
	Index        int
	Text         string
}

func (ReasoningDelta) isLLMEvent() {}

// CompletionComplete settles one completion. It carries everything a stat row needs, since
// this is the natural unit of billable work.
type CompletionComplete struct {
	CompletionID string
	Provider     string
	Model        string
	// Text is the whole answer, reasoning excluded, so a caller that ignored the deltas
	// still has the reply.
	Text string
	// InputTokens is the whole prompt the model read, cached part included.
	InputTokens int64
	// CachedInputTokens is the part of the prompt the provider served from its cache.
	CachedInputTokens int64
	// OutputTokens is everything generated, reasoning included.
	OutputTokens int64
	// ReasoningTokens is the part of the output spent thinking. It is a subset of
	// OutputTokens, reported because it explains a slow turn.
	ReasoningTokens int64
	// TimeToFirstTokenMs is how long the caller waited for anything at all, which is the
	// number that decides whether a conversation feels alive.
	TimeToFirstTokenMs float64
	// CompletionTimeMs is the whole completion, request to last delta.
	CompletionTimeMs float64
	// FinishReason is why the model stopped, e.g. "stop" or "length".
	FinishReason string
	// Interrupted is true when barge-in cut the completion short.
	Interrupted bool
}

func (CompletionComplete) isLLMEvent() {}

// Connected means the provider is ready to take requests.
type Connected struct {
	Provider string
	Model    string
	At       time.Time
}

func (Connected) isLLMEvent() {}

// Disconnected means the provider is no longer usable. Clean is false for failures.
type Disconnected struct {
	Provider string
	Model    string
	Reason   string
	Clean    bool
	At       time.Time
}

func (Disconnected) isLLMEvent() {}

// Error reports a provider failure. Fatal means the session cannot continue.
type Error struct {
	Provider     string
	Model        string
	CompletionID string
	Err          error
	Context      string
	Fatal        bool
}

func (e Error) Error() string { return e.Err.Error() }

func (e Error) Unwrap() error { return e.Err }

func (Error) isLLMEvent() {}

// Emitter fans provider events out to a single consumer channel. Providers hold one
// rather than managing the channel and its close semantics themselves.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
