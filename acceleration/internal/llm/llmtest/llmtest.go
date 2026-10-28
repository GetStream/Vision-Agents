// Package llmtest stands in for a language model provider in tests.
//
// A Script is a response the test drives itself: it answers when the test says so and not
// before, which is what lets a reply be caught mid-sentence the way barge-in finds one. For
// a response that has already happened, llm.Replay is enough and this is not needed.
package llmtest

import (
	"context"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Script is a response a test writes as it goes.
type Script struct {
	stream *llm.Stream

	mu sync.Mutex
	// pending is what the next Advance hands to the writer.
	pending []func(w *llm.ResponseWriter)
	// ready wakes the goroutine draining the stream when there is something to hand it.
	ready chan struct{}
	// ended is closed once the response has been settled or abandoned.
	ended     chan struct{}
	failure   error
	abandoned bool
	closeOnce sync.Once
	// breaking is the context FailOnCancel watches, and broke is closed once the failure is
	// written or the response ended without one.
	breaking context.Context
	broke    chan struct{}
	// closing runs when the code under test closes the stream (OnClose).
	closing func()
}

// New returns a script and the stream it is read from.
func New(options llm.StreamOptions) *Script {
	script := &Script{
		ready: make(chan struct{}, 1),
		ended: make(chan struct{}),
	}
	script.stream = llm.NewStream(options, script)
	return script
}

// Stream is what the code under test drains.
func (s *Script) Stream() *llm.Stream { return s.stream }

// OutputText streams a piece of the answer.
func (s *Script) OutputText(text string) {
	s.write(func(w *llm.ResponseWriter) { w.OutputText(text) })
}

// ReasoningText streams a piece of the model's thinking.
func (s *Script) ReasoningText(text string) {
	s.write(func(w *llm.ResponseWriter) { w.ReasoningText(text) })
}

// ToolCalls asks for tools, whole, in the order given.
func (s *Script) ToolCalls(calls ...llm.ToolCall) {
	s.write(func(w *llm.ResponseWriter) {
		for index, call := range calls {
			w.FunctionCall(int64(index), call.ID, call.Name, call.Arguments, call.Signature)
		}
	})
}

// Usage reports what the response consumed.
func (s *Script) Usage(usage llm.Usage) {
	s.write(func(w *llm.ResponseWriter) { w.Usage(usage) })
}

// Fail reports a provider failure and ends the response.
func (s *Script) Fail(err error, context string) {
	s.write(func(w *llm.ResponseWriter) { w.Fail(err, context, false) })
	s.mu.Lock()
	s.failure = err
	s.mu.Unlock()
	s.Done()
}

// Cancel ends the response as abandoned before it finished, the way a provider ends a stream
// whose context was cancelled: with no failure (openai.go, anthropic/messages.go,
// openaicompat.go all report context.Canceled as ResponseWriter.Cancelled).
func (s *Script) Cancel() {
	s.write(func(w *llm.ResponseWriter) { w.Cancelled() })
	s.Done()
}

// FailOnCancel has the response fail with err once ctx is cancelled: a provider whose
// connection broke as the caller cut the turn off reports a real error, not a cancellation. A
// Close after the cancel waits for that failure, so the failure is always written.
func (s *Script) FailOnCancel(ctx context.Context, err error) {
	broke := make(chan struct{})
	s.mu.Lock()
	s.breaking, s.broke = ctx, broke
	s.mu.Unlock()
	go func() {
		defer close(broke)
		select {
		case <-ctx.Done():
			// A response that settled before the cancel came is over, and nothing broke it.
			select {
			case <-s.ended:
			default:
				s.Fail(err, "read")
			}
		case <-s.ended:
		}
	}()
}

// OnClose runs f when the code under test closes the stream, before the response ends: a
// provider whose connection takes a while to shut down, while the work the close is part of
// waits for it.
func (s *Script) OnClose(f func()) {
	s.mu.Lock()
	s.closing = f
	s.mu.Unlock()
}

// Done settles the response, which is what unblocks whoever is draining it.
func (s *Script) Done() {
	s.closeOnce.Do(func() { close(s.ended) })
	s.wake()
}

// Advance hands over whatever the test has written, waiting when it has written nothing
// yet. This is the whole point of a script: a response that has not been settled stays in
// flight, exactly as a real one does.
func (s *Script) Advance(w *llm.ResponseWriter) bool {
	for {
		s.mu.Lock()
		if len(s.pending) > 0 {
			next := s.pending[0]
			s.pending = s.pending[1:]
			s.mu.Unlock()
			next(w)
			return true
		}
		s.mu.Unlock()

		select {
		case <-s.ready:
		case <-s.ended:
			s.mu.Lock()
			waiting := len(s.pending)
			s.mu.Unlock()
			if waiting == 0 {
				return false
			}
		}
	}
}

// Err is the failure the script reported, if any.
func (s *Script) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return stack.Wrap(s.failure)
}

// Close abandons the response, which is what the code under test calls to barge in.
func (s *Script) Close() error {
	s.mu.Lock()
	breaking, broke, closing := s.breaking, s.broke, s.closing
	s.mu.Unlock()
	if breaking != nil && breaking.Err() != nil {
		<-broke
	}
	if closing != nil {
		closing()
	}

	s.mu.Lock()
	s.abandoned = true
	s.mu.Unlock()

	s.Done()
	return nil
}

// Abandoned reports whether the code under test closed the stream, which is how a test
// tells barge-in from a response that was left to finish.
func (s *Script) Abandoned() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.abandoned
}

func (s *Script) write(step func(w *llm.ResponseWriter)) {
	s.mu.Lock()
	s.pending = append(s.pending, step)
	s.mu.Unlock()
	s.wake()
}

func (s *Script) wake() {
	select {
	case s.ready <- struct{}{}:
	default:
	}
}
