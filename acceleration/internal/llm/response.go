package llm

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
	// ToolCalls are the assembled calls the model asked for, in the order it asked. A
	// response may carry both these and text: a model told to keep the caller company
	// while it acts will say something and call a tool in the same breath.
	ToolCalls []ToolCall
	Usage     Usage
	Status    ResponseStatus
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
	TotalTokens         int64
}

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
