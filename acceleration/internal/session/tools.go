package session

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// defaultToolTimeout bounds how long a conversation waits on a tool that is being run
// somewhere else.
//
// It is generous because the wait is not silent: the voice model asked for the tool in the
// middle of a reply it is still speaking, and the result only has to arrive before the next
// turn. It is bounded at all because a caller that disconnected mid-call would otherwise
// leave the model holding a call it will never get an answer to.
const defaultToolTimeout = 30 * time.Second

// bridge runs the tools this process does not own by asking whoever does.
//
// It is an agent.ToolRunner whose implementation is a round trip: the request goes out as a
// session event, the answer comes back through Resolve, and the two are matched on the id
// the model gave the call. Everything a caller could get wrong here ends as words for the
// model rather than an error nobody hears: a tool nobody answered is a tool that did not
// work, and the agent apologises for it the same way it would for a failed transfer.
type bridge struct {
	timeout time.Duration
	// ask publishes the request and reports whether anyone was there to receive it.
	ask func(ToolCall) error

	mu sync.Mutex
	// pending is one channel per call in flight, keyed by the id the model gave it.
	pending map[string]chan toolResult
	closed  bool
}

// toolResult is what the caller said happened.
type toolResult struct {
	output  string
	failure string
}

func newBridge(timeout time.Duration, ask func(ToolCall) error) *bridge {
	if timeout <= 0 {
		timeout = defaultToolTimeout
	}
	return &bridge{
		timeout: timeout,
		ask:     ask,
		pending: map[string]chan toolResult{},
	}
}

// Run carries one tool call out to the caller and waits for the answer.
func (b *bridge) Run(ctx context.Context, call llm.ToolCall) (string, error) {
	answer := make(chan toolResult, 1)

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return "", errors.New("session: the call has ended")
	}
	if _, duplicate := b.pending[call.ID]; duplicate {
		b.mu.Unlock()
		return "", fmt.Errorf("session: %s was already asked for", call.ID)
	}
	b.pending[call.ID] = answer
	b.mu.Unlock()

	defer func() {
		b.mu.Lock()
		delete(b.pending, call.ID)
		b.mu.Unlock()
	}()

	if err := b.ask(ToolCall{ID: call.ID, Name: call.Name, Arguments: call.Arguments}); err != nil {
		return "", err
	}

	deadline, cancel := context.WithTimeout(ctx, b.timeout)
	defer cancel()

	select {
	case result := <-answer:
		if result.failure != "" {
			return "", errors.New(result.failure)
		}
		return result.output, nil
	case <-deadline.Done():
		return "", fmt.Errorf("session: %s did not answer within %s", call.Name, b.timeout)
	}
}

// Resolve hands an answer back to the call waiting for it, reporting whether one was.
//
// An answer for a call nobody is waiting on is dropped rather than an error, because the
// commonest reason for one is a caller answering a tool that has already timed out.
func (b *bridge) Resolve(id, output, failure string) bool {
	b.mu.Lock()
	answer, waiting := b.pending[id]
	b.mu.Unlock()
	if !waiting {
		return false
	}

	select {
	case answer <- toolResult{output: output, failure: failure}:
		return true
	default:
		return false
	}
}

// Close fails everything still in flight, so a call that ends does not leave the model
// waiting out the timeout on work nobody is going to do.
func (b *bridge) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.closed = true
	for _, answer := range b.pending {
		select {
		case answer <- toolResult{failure: "the call ended before it finished"}:
		default:
		}
	}
}
