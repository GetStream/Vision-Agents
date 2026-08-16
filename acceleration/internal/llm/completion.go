package llm

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// Completion tracks one completion in flight.
//
// Every provider needs the same bookkeeping to report a completion honestly: when the
// request went out, when the first token came back, what the answer was and how many
// tokens it cost. Deltas arrive on the provider's read goroutine while the caller may
// interrupt from its own, so this is safe for both.
type Completion struct {
	// ID correlates the events belonging to this completion.
	ID string

	mu           sync.Mutex
	startedAt    time.Time
	firstTokenAt time.Time
	text         strings.Builder
	deltas       int
	usage        tokens
	finishReason string
}

// tokens is what the provider said the completion consumed.
type tokens struct {
	input     int64
	cached    int64
	output    int64
	reasoning int64
}

// NewCompletion starts tracking a completion, generating an ID when the caller has none.
func NewCompletion(id string) *Completion {
	if id == "" {
		id = fmt.Sprintf("c-%d", time.Now().UnixNano())
	}
	return &Completion{ID: id, startedAt: time.Now()}
}

// Delta records a piece of the answer and returns the event to emit for it.
func (c *Completion) Delta(text string) TextDelta {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.markFirstToken()
	c.text.WriteString(text)
	index := c.deltas
	c.deltas++

	return TextDelta{CompletionID: c.ID, Index: index, Text: text}
}

// Reasoning records a piece of the model's thinking and returns the event to emit for it.
// Thinking counts towards time to first token, since it is the provider working, but it is
// not part of the answer.
func (c *Completion) Reasoning(text string) ReasoningDelta {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.markFirstToken()
	index := c.deltas
	c.deltas++

	return ReasoningDelta{CompletionID: c.ID, Index: index, Text: text}
}

// Usage records what the provider reported. Providers that repeat a cumulative usage frame
// on every chunk can call it as often as they like; the last one wins.
func (c *Completion) Usage(input, cached, output, reasoning int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.usage = tokens{input: input, cached: cached, output: output, reasoning: reasoning}
}

// Finish records why the model stopped.
func (c *Completion) Finish(reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.finishReason = reason
}

// Text is the answer so far, reasoning excluded.
func (c *Completion) Text() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.text.String()
}

// Complete returns the event that settles the completion. A completion that produced
// nothing reports a zero time to first token rather than the whole completion time.
func (c *Completion) Complete(provider, model string, interrupted bool) CompletionComplete {
	c.mu.Lock()
	defer c.mu.Unlock()

	var timeToFirstToken float64
	if !c.firstTokenAt.IsZero() {
		timeToFirstToken = float64(c.firstTokenAt.Sub(c.startedAt).Microseconds()) / 1000
	}

	return CompletionComplete{
		CompletionID:       c.ID,
		Provider:           provider,
		Model:              model,
		Text:               c.text.String(),
		InputTokens:        c.usage.input,
		CachedInputTokens:  c.usage.cached,
		OutputTokens:       c.usage.output,
		ReasoningTokens:    c.usage.reasoning,
		TimeToFirstTokenMs: timeToFirstToken,
		CompletionTimeMs:   float64(time.Since(c.startedAt).Microseconds()) / 1000,
		FinishReason:       c.finishReason,
		Interrupted:        interrupted,
	}
}

// markFirstToken stamps the first sign of life from the provider. Callers hold the lock.
func (c *Completion) markFirstToken() {
	if c.firstTokenAt.IsZero() {
		c.firstTokenAt = time.Now()
	}
}
