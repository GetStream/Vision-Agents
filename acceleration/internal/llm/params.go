package llm

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// ResponseParams is one response to generate.
//
// The whole conversation is passed every time rather than held by the provider, because
// routing may send consecutive turns to different providers and a conversation that lives
// in the caller survives a failover. PreviousResponseID is an optimisation on top of that
// rather than a replacement for it.
type ResponseParams struct {
	// ID correlates every event belonging to one response. Providers generate one when
	// it is empty. It is not sent upstream.
	ID string
	// Purpose names the model's role in a call: reply, flow or subagent.
	Purpose string
	// TurnID is the caller exchange behind this request, including delegated work.
	TurnID string
	// OnTiming reports this request when it settles, including failed attempts.
	OnTiming func(CallTiming)
	// Attributes are stored with the request's usage row. They are asked for when the row is
	// written, so one settled after the request was made, such as whether a speculative reply
	// was kept, is on it. Nil adds none.
	Attributes func() map[string]string
	// Instructions is the system prompt. It is separate from Input so a caller cannot
	// forget it on a retry, and providers prepend it themselves.
	Instructions string
	// Input is the conversation so far, oldest first, ending with what to respond to.
	Input []Message
	// Tools the model may call. An empty list sends none, which is what a request that
	// only wants prose wants: a model offered a tool will eventually reach for it.
	Tools []Tool
	// ToolChoice is how the model picks, one of auto, none or required. Empty leaves the
	// provider's own default in place.
	ToolChoice string
	// MaxOutputTokens caps the response, reasoning included. Zero leaves the provider's
	// own default in place.
	MaxOutputTokens int
	// Temperature controls randomness. Nil leaves the provider's own default in place,
	// which is not the same as zero.
	Temperature *float64
	// Reasoning is how hard the model should think. It means nothing to a model that
	// does not reason, and Create rejects an effort the model does not accept.
	Reasoning ReasoningParams
	// Text shapes the answer itself: whether it is prose or one JSON object, and how
	// much of it there should be.
	Text TextParams
	// Store asks the provider to keep the response so a later one can continue from it.
	// It means nothing to a provider whose Capabilities do not report Store.
	Store bool
	// PreviousResponseID continues from a stored response, so only what has been added
	// since needs sending. Empty replays the whole conversation.
	PreviousResponseID string
	// Conversation is a provider-held conversation this response belongs to. It cannot
	// be combined with PreviousResponseID.
	Conversation string
	// PromptCacheKey buckets requests that share a prefix, which is how the agent's
	// instructions are cached once and reused across every turn of every call it takes.
	PromptCacheKey string
	// PromptCacheOptions is how long that cache lives and whether the provider may pick
	// its own breakpoints.
	PromptCacheOptions PromptCacheOptions
	// Metadata is carried by the provider and returned with the response. It is for the
	// caller's own bookkeeping and never reaches the model.
	Metadata map[string]string
}

// CallTiming is the time spent in one provider attempt.
type CallTiming struct {
	OperationID string
	Purpose     string
	TurnID      string
	Provider    string
	Model       string
	TTFTMs      float64
	DurationMs  float64
	// Success is whether the request ran to its end as asked. ErrorCode says why it did not:
	// "cancelled" when its caller closed it first, which is no failure of the provider's and
	// is still billed, otherwise the provider's or the request's own failure.
	Success   bool
	ErrorCode string
	// InputTokens, CachedInputTokens and OutputTokens are what the request read and wrote,
	// and CostMicros what it was priced at, in millionths of a dollar. Zero for a request
	// that failed before the provider answered.
	InputTokens       int64
	CachedInputTokens int64
	OutputTokens      int64
	CostMicros        int64
}

// Overwrite returns these params with everything the caller asked to change written over
// them.
//
// Only the fields a session may overwrite are read: instructions, tools and the cache key
// belong to the agent that built the turn, and a caller able to rewrite those could make a
// session pretend to be a different agent. What is left is how hard to think, how long to
// answer, how random to be and how much detail to give -- the knobs it is safe to hand to
// whoever opened the conversation.
//
// A field the caller said nothing about is left alone rather than zeroed, which is why the
// numbers travel as pointers: a temperature of zero is a real request, and a missing one is
// not the same thing.
func (p ResponseParams) Overwrite(over options.LLM) ResponseParams {
	if over.ReasoningEffort != "" {
		p.Reasoning.Effort = over.ReasoningEffort
	}
	if over.Temperature != nil {
		p.Temperature = over.Temperature
	}
	if over.MaxOutputTokens != nil {
		p.MaxOutputTokens = *over.MaxOutputTokens
	}
	if over.Verbosity != "" {
		p.Text.Verbosity = over.Verbosity
	}
	return p
}

// ReasoningParams is how much thinking to do before answering.
//
// Effort is deliberately a string rather than an enum: the levels differ per model and do
// not map cleanly onto each other, so what is valid is declared by the model's
// Capabilities rather than fixed here.
type ReasoningParams struct {
	Effort string
}

// TextFormat is the shape of the answer.
type TextFormat string

const (
	// FormatText is prose, which is what a provider does when asked for nothing.
	FormatText TextFormat = "text"
	// FormatJSONObject asks for one JSON object instead of prose. A caller that parses
	// the answer needs it: a model handed a conversation and asked about it will
	// otherwise sometimes carry the conversation on instead.
	FormatJSONObject TextFormat = "json_object"
	// FormatJSONSchema asks for one JSON object that matches TextParams.Schema. A model
	// whose Capabilities do not report JSONSchema is asked for a JSON object instead, so
	// the caller still checks what comes back.
	FormatJSONSchema TextFormat = "json_schema"
)

// JSON reports whether the answer is asked for as JSON rather than prose.
func (f TextFormat) JSON() bool { return f == FormatJSONObject || f == FormatJSONSchema }

// JSONSchema is what an answer asked for in FormatJSONSchema must match.
type JSONSchema struct {
	// Name labels the schema for the provider: letters, digits, underscores and dashes,
	// at most 64 of them.
	Name string
	// Schema is the JSON Schema itself. It is sent strict, so every property has to be
	// required and additionalProperties false.
	Schema map[string]any
}

// TextParams configures the answer's shape and length.
type TextParams struct {
	Format TextFormat
	// Schema is what an answer in FormatJSONSchema must match.
	Schema *JSONSchema
	// Verbosity is the default level of detail, one of low, medium or high. Empty leaves
	// the provider's own default in place. It is dropped for models that do not take it.
	Verbosity string
}

// CacheMode is whether the provider may choose its own cache breakpoints.
type CacheMode string

const (
	// CacheImplicit lets the provider pick a breakpoint, which is what a conversation
	// wants: the prefix grows a turn at a time and each turn caches the one before it.
	CacheImplicit CacheMode = "implicit"
	// CacheExplicit uses only the breakpoints the request names. The instructions are
	// where one goes, since they are the part every turn of every call shares.
	CacheExplicit CacheMode = "explicit"
)

// PromptCacheOptions is the request's cache policy.
type PromptCacheOptions struct {
	Mode CacheMode
	// TTL is how long a cached prefix stays warm. Zero leaves the provider's own default
	// in place, and is dropped for providers that do not take one.
	TTL time.Duration
}
