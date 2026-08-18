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
	// tools holds the calls being assembled, keyed by the index the provider streams
	// them under, and order remembers which index arrived first so the calls settle in
	// the order the model asked for them.
	tools map[int64]*partialCall
	order []int64
}

// partialCall is one tool call being assembled from the fragments a provider streams.
type partialCall struct {
	id        string
	name      string
	arguments strings.Builder
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
	return &Completion{ID: id, startedAt: time.Now(), tools: map[int64]*partialCall{}}
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

// ToolCall records a fragment of a call the model is asking for and returns the event to
// emit for it.
//
// A call arrives in pieces under one index: the first carries the id and the name, and the
// arguments follow as JSON text spread over the fragments after it. Anything empty leaves
// what was already recorded alone, so a fragment carrying only arguments does not erase the
// name.
func (c *Completion) ToolCall(index int64, id, name, arguments string) ToolCallDelta {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.markFirstToken()

	call, known := c.tools[index]
	if !known {
		call = &partialCall{}
		c.tools[index] = call
		c.order = append(c.order, index)
	}
	if id != "" {
		call.id = id
	}
	if name != "" {
		call.name = name
	}
	call.arguments.WriteString(arguments)

	return ToolCallDelta{
		CompletionID: c.ID,
		Index:        index,
		ToolCallID:   id,
		Name:         name,
		Arguments:    arguments,
	}
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
		ToolCalls:          c.calls(),
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

// calls assembles the tool calls in the order the model asked for them. Callers hold the
// lock.
//
// A provider that names its calls but does not identify them gets an id made here, because
// the result sent back has to say which call it answers and the model gave nothing else to
// say it with.
func (c *Completion) calls() []ToolCall {
	if len(c.order) == 0 {
		return nil
	}

	assembled := make([]ToolCall, 0, len(c.order))
	for _, index := range c.order {
		call := c.tools[index]
		id := call.id
		if id == "" {
			id = fmt.Sprintf("%s-tool-%d", c.ID, index)
		}
		assembled = append(assembled, ToolCall{
			ID:        id,
			Name:      call.name,
			Arguments: call.arguments.String(),
		})
	}
	return assembled
}

// markFirstToken stamps the first sign of life from the provider. Callers hold the lock.
func (c *Completion) markFirstToken() {
	if c.firstTokenAt.IsZero() {
		c.firstTokenAt = time.Now()
	}
}
