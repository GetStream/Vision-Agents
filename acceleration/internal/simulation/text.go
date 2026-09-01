package simulation

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// replyPoll is how often a turn in progress is looked at. It is not a timeout: the agent
// reports what it is doing, and this only decides how promptly that is noticed.
const replyPoll = 25 * time.Millisecond

// settleGap is how long the agent has to stay quiet before what it said is taken as all of
// it.
//
// Asking whether it is busy is not quite enough on its own. One thing said to the agent can
// earn several replies -- the turn that called a tool, the turn that read what it returned,
// the turn a subagent's finding was worth -- and in the moment between one of those ending
// and the next beginning the agent is genuinely doing nothing. Nobody is waiting on a
// simulation, so it waits out the handover rather than talking into it.
const settleGap = 600 * time.Millisecond

// written is a conversation held in writing.
type written struct {
	created *session.Session
	closer  func()
	// within is how long one turn waits for the agent before giving up on it.
	within time.Duration
	// opening is the greeting, kept because the turn it was said on is cleared before the
	// caller's first one and it belongs in the conversation either way.
	opening string

	mu sync.Mutex
	// answering accumulates the current turn's replies. One thing said to the agent can
	// earn several: the turn that called a tool, the turn that read what it returned, and
	// whatever a subagent's finding was worth. They are one answer to the caller.
	answering []string
	// lastAt is when the agent last did anything, which is what the quiet is measured from.
	lastAt time.Time
	// failure is the last thing that went wrong, kept so a turn that never arrives can say
	// why rather than only that it did not.
	failure error
	// gone is the session having ended under the conversation.
	gone bool
}

// speak opens a conversation in writing against the agent under test.
func (r *Runner) speak(ctx context.Context, spec session.Spec) (transport, error) {
	// The greeting is said below rather than by the manager. Create says it before the
	// session is reachable, and a watcher attaches afterwards, so a greeting left here
	// would be spoken to nobody and missing from the transcript.
	greeting := spec.Greeting
	spec.Greeting = ""

	created, err := r.sessions.Create(ctx, spec)
	if err != nil {
		return nil, fmt.Errorf("simulation: open the conversation: %w", err)
	}

	events, detach := created.Watch()
	held := &written{created: created, closer: detach, within: replyWithin}

	// The events are drained continuously rather than only while a turn is being waited
	// for: the fan-out drops rather than blocks, so a channel nobody is reading loses the
	// answer instead of delaying it.
	go held.collect(events)

	if greeting != "" {
		if err := created.Say(ctx, greeting); err != nil {
			held.Close()
			return nil, fmt.Errorf("simulation: greet: %w", err)
		}
		held.opening = greeting
		// The greeting arrives on the same channel a reply does, so it is waited out here
		// rather than left to turn up in the middle of the answer to the first question.
		held.settled(ctx)
		held.begin()
	}
	return held, nil
}

func (w *written) Session() *session.Session { return w.created }

func (w *written) Opening() string { return w.opening }

// Say hands the agent one line and waits for the whole of what it says back.
func (w *written) Say(ctx context.Context, text string) (store.SimulationLine, error) {
	w.begin()

	if err := w.created.Respond(ctx, text); err != nil {
		return store.SimulationLine{}, err
	}

	started := time.Now()
	ticker := time.NewTicker(replyPoll)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return store.SimulationLine{}, ctx.Err()
		case <-ticker.C:
			answer, quiet, failure, gone := w.state()
			switch {
			case gone:
				return store.SimulationLine{}, errors.New("simulation: the conversation ended before the agent answered")
			// The agent having said something is not enough: it may be about to say the
			// rest. Waiting until it has nothing left to do and has been quiet about it is
			// what keeps the caller from talking over the second half of an answer.
			case answer != "" && quiet >= settleGap && !w.created.Busy():
				return store.SimulationLine{Text: answer}, nil
			case time.Since(started) >= w.within:
				if failure != nil {
					return store.SimulationLine{}, fmt.Errorf("simulation: the agent did not answer: %w", failure)
				}
				return store.SimulationLine{}, errors.New("simulation: the agent did not answer")
			}
		}
	}
}

func (w *written) Close() error {
	w.closer()
	return nil
}

// collect reads what the agent reported until the conversation ends.
//
// Only Responded is kept. The deltas are the same words arriving in pieces, and the whole
// reply is on the event that closes the turn, so there is no streaming to race with.
func (w *written) collect(events <-chan session.Event) {
	for event := range events {
		switch typed := event.(type) {
		case agent.Responded:
			w.mu.Lock()
			w.lastAt = time.Now()
			// A turn that only called a tool says nothing, and there is another Responded
			// coming with what the tool's answer came to.
			if said := strings.TrimSpace(typed.Text); said != "" {
				w.answering = append(w.answering, said)
			}
			w.mu.Unlock()
		case agent.Error:
			w.mu.Lock()
			w.lastAt = time.Now()
			w.failure = typed.Err
			w.mu.Unlock()
		default:
			w.mu.Lock()
			w.lastAt = time.Now()
			w.mu.Unlock()
		}
	}

	w.mu.Lock()
	w.gone = true
	w.mu.Unlock()
}

// begin forgets the last turn, so what is waited for is this answer rather than the one
// before it.
func (w *written) begin() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.answering = nil
	w.failure = nil
	w.lastAt = time.Now()
}

func (w *written) state() (string, time.Duration, error, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return strings.Join(w.answering, " "), time.Since(w.lastAt), w.failure, w.gone
}

// settled waits for the agent to stop doing whatever it is doing, which is how the greeting
// is kept out of the answer to the first thing the caller says.
func (w *written) settled(ctx context.Context) {
	ticker := time.NewTicker(replyPoll)
	defer ticker.Stop()

	deadline := time.Now().Add(w.within)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if _, quiet, _, _ := w.state(); quiet >= settleGap && !w.created.Busy() {
				return
			}
		}
	}
}
