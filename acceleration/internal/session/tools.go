package session

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// interruptible is next with the caller's and the workers' tools that asked to be cancelled
// by an interruption marked so. Nil when none asked, so a session that did not is left
// with next as it was.
func interruptible(tools []harness.Tool, next func(string) agent.ToolPolicy) func(string) agent.ToolPolicy {
	cancels := map[string]bool{}
	for _, tool := range tools {
		if tool.Interruptible {
			cancels[tool.Name] = true
		}
	}
	if len(cancels) == 0 {
		return nil
	}
	return func(name string) agent.ToolPolicy {
		var policy agent.ToolPolicy
		if next != nil {
			policy = next(name)
		}
		if cancels[name] {
			policy.Cancels, policy.Waits = true, false
		}
		return policy
	}
}

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
	// timeouts is the tools that name their own, in place of timeout.
	timeouts map[string]time.Duration
	// ask publishes the request and reports whether anyone was there to receive it.
	ask func(ToolCall) error

	mu sync.Mutex
	// pending is one channel per call in flight, keyed by the id the model gave it.
	pending map[string]pendingTool
	closed  bool
}

type pendingTool struct {
	turnID   string
	call     llm.ToolCall
	user     *ToolUser
	approved bool
	ctx      context.Context
	answer   chan toolResult
	resolved bool
}

// toolResult is what the caller said happened.
type toolResult struct {
	parts   []llm.ContentPart
	failure string
}

func newBridge(timeout time.Duration, tools []harness.Tool, ask func(ToolCall) error) *bridge {
	if timeout <= 0 {
		timeout = defaultToolTimeout
	}
	timeouts := map[string]time.Duration{}
	for _, tool := range tools {
		if tool.Timeout > 0 {
			timeouts[tool.Name] = tool.Timeout
		}
	}
	return &bridge{
		timeout:  timeout,
		timeouts: timeouts,
		ask:      ask,
		pending:  map[string]pendingTool{},
	}
}

// Run carries one tool call out to the caller and waits for the answer.
func (b *bridge) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	timeout := b.timeout
	if own, named := b.timeouts[call.Name]; named {
		timeout = own
	}
	deadline, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	answer := make(chan toolResult, 1)

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, errors.New("session: the call has ended")
	}
	if _, duplicate := b.pending[call.ID]; duplicate {
		b.mu.Unlock()
		return nil, fmt.Errorf("session: %s was already asked for", call.ID)
	}
	var user *ToolUser
	if requester, ok := agent.RequesterFrom(ctx); ok {
		user = &ToolUser{ID: requester.UserID, Name: requester.Name}
	}
	approved := agent.Approved(ctx)
	b.pending[call.ID] = pendingTool{turnID: call.TurnID, call: call, user: user, approved: approved, ctx: deadline, answer: answer}
	b.mu.Unlock()

	defer func() {
		b.mu.Lock()
		delete(b.pending, call.ID)
		b.mu.Unlock()
	}()

	if err := b.ask(ToolCall{ID: call.ID, TurnID: call.TurnID, Name: call.Name, Arguments: call.Arguments, User: user, Approved: approved}); err != nil {
		return nil, err
	}

	select {
	case result := <-answer:
		if result.failure != "" {
			return nil, errors.New(result.failure)
		}
		return result.parts, nil
	case <-deadline.Done():
		_ = b.ask(ToolCall{ID: call.ID, TurnID: call.TurnID, Name: call.Name, Cancel: true})
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("session: %s stopped: %w", call.Name, err)
		}
		return nil, fmt.Errorf("session: %s did not answer within %s: %w", call.Name, timeout, deadline.Err())
	}
}

// Pending returns only requests still waiting for a result at this instant.
// Reconnecting tool hosts must fence repeated execution with durable receipts.
func (b *bridge) Pending() []ToolCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	var calls []ToolCall
	for _, pending := range b.pending {
		if pending.resolved || pending.ctx.Err() != nil {
			continue
		}
		call := pending.call
		calls = append(calls, ToolCall{ID: call.ID, TurnID: call.TurnID, Name: call.Name, Arguments: call.Arguments, User: pending.user, Approved: pending.approved})
	}
	return calls
}

// nameOf is the tool the call id asked for, when it is still waiting.
func (b *bridge) nameOf(id string) (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	pending, waiting := b.pending[id]
	return pending.call.Name, waiting
}

// Resolve hands an answer back to the call waiting for it, reporting whether one was.
//
// An answer for a call nobody is waiting on is dropped rather than an error, because the
// commonest reason for one is a caller answering a tool that has already timed out.
func (b *bridge) Resolve(id, turnID string, parts []llm.ContentPart, failure string) bool {
	b.mu.Lock()
	pending, waiting := b.pending[id]
	if !waiting || pending.resolved || pending.ctx.Err() != nil || turnID != "" && pending.turnID != turnID {
		b.mu.Unlock()
		return false
	}
	pending.resolved = true
	b.pending[id] = pending
	b.mu.Unlock()

	select {
	case pending.answer <- toolResult{parts: parts, failure: failure}:
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
	for _, pending := range b.pending {
		select {
		case pending.answer <- toolResult{failure: "the call ended before it finished"}:
		default:
		}
	}
}
