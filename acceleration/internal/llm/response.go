package llm

import "github.com/GetStream/Vision-Agents/acceleration/internal/pricing"

// ResponseStatus is how a response ended.
type ResponseStatus string

const (
	// StatusCompleted means the model stopped because it was finished.
	StatusCompleted ResponseStatus = "completed"
	// StatusIncomplete means it stopped for a reason of the caller's making, such as the
	// token cap. IncompleteReason says which.
	StatusIncomplete ResponseStatus = "incomplete"
	// StatusFailed means the provider failed part-way through.
	StatusFailed ResponseStatus = "failed"
	// StatusCancelled means the stream was closed before the model finished, which on the
	// live path is barge-in.
	StatusCancelled ResponseStatus = "cancelled"
)

const (
	// ReasonMaxOutputTokens means the token cap ended the response.
	ReasonMaxOutputTokens = "max_output_tokens"
	// ReasonContentFilter means the provider stopped itself.
	ReasonContentFilter = "content_filter"
)

// Response settles one response. It carries everything a stat row needs, since this is the
// natural unit of billable work.
type Response struct {
	// ID correlates the events belonging to this response. It is the caller's, so a turn
	// can be recognised by the id it was asked under.
	ID string
	// ProviderResponseID is what the provider stored the response as, and is what
	// PreviousResponseID continues from. It is empty for a provider that stores nothing.
	ProviderResponseID string
	Provider           string
	Model              string
	// OutputText is the whole answer, reasoning excluded, so a caller that ignored the
	// deltas still has the reply.
	OutputText string
	// ReasoningText is the whole of the model's thinking.
	ReasoningText string
	// ToolCalls are the assembled calls the model asked for, in the order it asked. A
	// response may carry both these and text: a model told to keep the caller company
	// while it acts will say something and call a tool in the same breath.
	ToolCalls []ToolCall
	Usage     Usage
	// UsageReported is true when the provider said what the response used, zero included.
	// A response cut short often carries none, and neither does an adapter that forgot to
	// ask: its cost is then estimated rather than taken as nothing.
	UsageReported bool
	// TimedOut is a response that ran past a deadline, its caller's or the provider's own
	// limit, before the model finished. It failed for whoever was waiting, so its status is
	// StatusFailed, but the provider did nothing wrong and billed what it had read and written.
	TimedOut bool
	// ServiceTier is the tier the provider says it served the response at, such as
	// OpenAI's default, flex or fast, which it bills at different rates. Empty for a
	// provider that does not say.
	ServiceTier string
	Status      ResponseStatus
	// IncompleteReason says why a StatusIncomplete response stopped early.
	IncompleteReason string
	// TimeToFirstTokenMs is how long the caller waited for anything at all, which is the
	// number that decides whether a conversation feels alive.
	TimeToFirstTokenMs float64
	// DurationMs is the whole response, request to last delta.
	DurationMs float64
}

// Usage is what the provider said the response consumed.
type Usage struct {
	// InputTokens is the whole prompt the model read, cached and written parts included.
	InputTokens         int64
	InputTokensDetails  InputTokensDetails
	OutputTokens        int64
	OutputTokensDetails OutputTokensDetails
	// TotalTokens is the provider's own total where it sends one, else input and output
	// added up. A total the two do not add up to is a part of the bill the adapter missed.
	TotalTokens int64
	// VendorCostMicros is what the provider said the response cost it, in millionths of a
	// dollar, for a provider that says. It never prices the response: the price list does,
	// and this is what it is checked against.
	VendorCostMicros int64
}

// Used is what the response is priced by. Cached and cache-written tokens are part of the
// prompt and are billed once, each at its own rate.
func (r Response) Used() pricing.Quantities {
	cached := r.Usage.InputTokensDetails.CachedTokens
	written := r.Usage.InputTokensDetails.CacheWriteTokens
	return pricing.Quantities{
		pricing.InputTokens:           max(r.Usage.InputTokens-cached-written, 0),
		pricing.CachedInputTokens:     cached,
		pricing.CacheWriteInputTokens: written,
		pricing.OutputTokens:          r.Usage.OutputTokens,
	}
}

// VendorCostMicros is what the provider said the response cost it.
func (r Response) VendorCostMicros() int64 { return r.Usage.VendorCostMicros }

// InputTokensDetails breaks the prompt down by how it was served.
type InputTokensDetails struct {
	// CachedTokens is the part of the prompt the provider served from its cache, billed
	// at a discount.
	CachedTokens int64
	// CacheWriteTokens is the part written into the cache for a later turn to read.
	// Recent models bill it above the uncached rate, so a cache that is never read from
	// costs more than no cache at all.
	CacheWriteTokens int64
}

// OutputTokensDetails breaks the generated tokens down.
type OutputTokensDetails struct {
	// ReasoningTokens is the part of the output spent thinking. It is a subset of
	// OutputTokens, reported because it explains a slow turn.
	ReasoningTokens int64
}
