package llm

import (
	"fmt"
	"slices"
	"strings"
	"time"
)

// Capabilities is what one model accepts.
//
// Reasoning levels are the reason this exists. They differ per model and do not map onto
// each other -- one model's minimal is another's none, and only the newest take max -- so
// rather than invent a common scale that would be wrong everywhere, each model declares
// the words it answers to and a request naming anything else is refused.
type Capabilities struct {
	// ReasoningEfforts are the efforts this model accepts. Empty means it does not
	// reason, and a request asking it to is refused.
	ReasoningEfforts []string
	// DefaultEffort is what the provider uses when a request names none.
	DefaultEffort string
	// StreamsReasoning reports whether the model emits ReasoningTextDelta events before
	// its answer. A caller on the live path uses it to decide whether to wait for the
	// first OutputTextDelta or to show thinking in the meantime.
	StreamsReasoning bool
	// Verbosities are the text verbosity levels this model accepts. Empty means it takes
	// none and the field is dropped.
	Verbosities []string
	// Store reports whether the provider keeps a response for a later one to continue
	// from, which is what makes PreviousResponseID mean anything.
	Store bool
	// Conversations reports whether the provider holds conversations of its own.
	Conversations bool
	// PromptCacheKey reports whether the provider buckets its cache by a key the caller
	// chooses.
	PromptCacheKey bool
	// CacheTTLs are the cache lifetimes the provider accepts. Empty means it caches on
	// its own terms or not at all, and a requested TTL is dropped.
	CacheTTLs []time.Duration
}

// Validate reports whether a request asks for something this model does not do.
//
// It is checked before the request goes out rather than after it comes back, because a
// provider's own answer to an unknown reasoning effort is a 400 halfway through a phone
// call.
func (c Capabilities) Validate(params ResponseParams) error {
	if effort := params.Reasoning.Effort; effort != "" {
		if len(c.ReasoningEfforts) == 0 {
			return fmt.Errorf("llm: this model does not reason, so it takes no reasoning effort, and %q was asked for", effort)
		}
		if !slices.Contains(c.ReasoningEfforts, effort) {
			return fmt.Errorf("llm: reasoning effort %q is not one of %s",
				effort, strings.Join(c.ReasoningEfforts, ", "))
		}
	}
	if verbosity := params.Text.Verbosity; verbosity != "" && len(c.Verbosities) > 0 {
		if !slices.Contains(c.Verbosities, verbosity) {
			return fmt.Errorf("llm: text verbosity %q is not one of %s",
				verbosity, strings.Join(c.Verbosities, ", "))
		}
	}
	if params.Conversation != "" && params.PreviousResponseID != "" {
		return fmt.Errorf("llm: a response continues from a conversation or from a previous response, not both")
	}
	return nil
}

// Effort is the reasoning effort to send, which is the request's when it named one and the
// model's own default otherwise. It is empty for a model that does not reason.
func (c Capabilities) Effort(params ResponseParams) string {
	if len(c.ReasoningEfforts) == 0 {
		return ""
	}
	if params.Reasoning.Effort != "" {
		return params.Reasoning.Effort
	}
	return c.DefaultEffort
}
