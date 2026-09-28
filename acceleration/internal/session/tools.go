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

// defaultApprovalTimeout bounds how long a conversation waits on a person instead.
//
// It is a different number because it is a different wait. Thirty seconds is generous for
// a machine and no time at all for somebody who has been handed a decision: they have to
// notice the card, read what it says and mean it. A caller given the machine's deadline to
// approve a refund was told the refund had failed while they were still looking at it, and
// their answer, when it came, went nowhere.
//
// Bounded all the same. Somebody who puts their phone down has still left a turn open, and
// the model is owed an answer either way.
const defaultApprovalTimeout = 5 * time.Minute

// bridge runs the tools this process does not own by asking whoever does.
//
// It is an agent.ToolRunner whose implementation is a round trip: the request goes out as a
// session event, the answer comes back through Resolve, and the two are matched on the id
// the model gave the call. Everything a caller could get wrong here ends as words for the
// model rather than an error nobody hears: a tool nobody answered is a tool that did not
// work, and the agent apologises for it the same way it would for a failed transfer.
type bridge struct {
	timeout  time.Duration
	approval time.Duration
	// ask publishes the request and reports whether anyone was there to receive it.
	ask func(ToolCall) error
	// expired says a call nobody answered is over, so whoever is holding a question open
	// on somebody's screen can take it down.
	expired func(string)

	mu sync.Mutex
	// pending is one call in flight, keyed by the id the model gave it.
	pending map[string]*asked
	closed  bool
}

// asked is one call the caller has been given and has yet to answer.
type asked struct {
	answer chan toolResult
	// waiting is closed once a person has been asked, which is what moves the call onto
	// the approval deadline. It is closed at most once, since the deadline is extended
	// once and not restarted by every notice about the same call.
	waiting chan struct{}
	once    sync.Once
}

// toolResult is what the caller said happened.
type toolResult struct {
	output  string
	failure string
}

func newBridge(
	timeout, approval time.Duration,
	ask func(ToolCall) error,
	expired func(string),
) *bridge {
	if timeout <= 0 {
		timeout = defaultToolTimeout
	}
	if approval <= 0 {
		approval = defaultApprovalTimeout
	}
	return &bridge{
		timeout:  timeout,
		approval: approval,
		ask:      ask,
		expired:  expired,
		pending:  map[string]*asked{},
	}
}

// Run carries one tool call out to the caller and waits for the answer.
//
// The wait is the machine's until the caller says a person has been asked, at which point
// it becomes theirs: see Waiting.
func (b *bridge) Run(ctx context.Context, call llm.ToolCall) (string, error) {
	held := &asked{answer: make(chan toolResult, 1), waiting: make(chan struct{})}

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return "", errors.New("session: the call has ended")
	}
	if _, duplicate := b.pending[call.ID]; duplicate {
		b.mu.Unlock()
		return "", fmt.Errorf("session: %s was already asked for", call.ID)
	}
	b.pending[call.ID] = held
	b.mu.Unlock()

	defer func() {
		b.mu.Lock()
		delete(b.pending, call.ID)
		b.mu.Unlock()
	}()

	if err := b.ask(ToolCall{ID: call.ID, Name: call.Name, Arguments: call.Arguments}); err != nil {
		return "", err
	}

	timeout := b.timeout
	waiting := held.waiting
	person := false
	for {
		deadline, cancel := context.WithTimeout(ctx, timeout)

		select {
		case result := <-held.answer:
			cancel()
			if result.failure != "" {
				return "", errors.New(result.failure)
			}
			return result.output, nil

		case <-waiting:
			cancel()
			// A person has been asked, so the deadline is theirs from here. Nil, because
			// a second notice about the same call must not start the wait over.
			timeout, waiting, person = b.approval, nil, true

		case <-deadline.Done():
			cancel()
			if person {
				// The question is still on somebody's screen, so it is taken down: the
				// model has been told the answer and a card that is answered by tapping
				// it would be answering nobody.
				if b.expired != nil {
					b.expired(call.ID)
				}
				// Words rather than an error. Nothing failed and nothing was done, and a
				// model told a tool broke apologises for a fault where it should be
				// saying the caller never said yes.
				return "Nobody approved this, so nothing was done. Say so, and ask what " +
					"they would like to do instead.", nil
			}
			return "", fmt.Errorf("session: %s did not answer within %s", call.Name, timeout)
		}
	}
}

// Waiting says a person has been asked to allow this call, so the wait is theirs.
//
// It reports whether anything was waiting, the way Resolve does: a notice about a call
// that has already been answered, or that gave up before the caller got to it, is dropped
// rather than an error.
func (b *bridge) Waiting(id string) bool {
	b.mu.Lock()
	held, waiting := b.pending[id]
	b.mu.Unlock()
	if !waiting {
		return false
	}

	held.once.Do(func() { close(held.waiting) })
	return true
}

// Resolve hands an answer back to the call waiting for it, reporting whether one was.
//
// An answer for a call nobody is waiting on is dropped rather than an error, because the
// commonest reason for one is a caller answering a tool that has already timed out.
func (b *bridge) Resolve(id, output, failure string) bool {
	b.mu.Lock()
	held, waiting := b.pending[id]
	b.mu.Unlock()
	if !waiting {
		return false
	}

	select {
	case held.answer <- toolResult{output: output, failure: failure}:
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
	for _, held := range b.pending {
		select {
		case held.answer <- toolResult{failure: "the call ended before it finished"}:
		default:
		}
	}
}
