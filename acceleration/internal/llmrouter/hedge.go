package llmrouter

import (
	"context"
	"slices"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// replyPurpose is what the agent calls the request a caller is waiting to hear the answer to.
// The preview of a reply is the same request, so it is called the same.
const replyPurpose = "reply"

// hedges says whether a request is asked a second time when it is late. Only a reply is, since
// it is the one somebody is waiting on, and a response that continues from one the provider
// holds cannot be asked of another provider, which is also why it is never failed over.
func (s *Session) hedges(params llm.ResponseParams) bool {
	return s.hedge != nil && s.hedgeAfter > 0 && params.Purpose == replyPurpose &&
		params.PreviousResponseID == "" && params.Conversation == ""
}

// hedged asks for a response and, if nothing has been said once hedgeAfter has passed, asks
// for the same response on another candidate as well. Whichever says something first is
// returned, and the other is cancelled the moment it loses.
//
// A request that fails, or ends without saying anything, leaves the race and the other goes on
// alone, so a hedge that errors costs the original nothing. When both are out, the original's
// outcome stands: what a request that was never hedged would have had.
func (s *Session) hedged(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	original := race(ctx, func(ctx context.Context) (*llm.Stream, func(), error) {
		stream, err := s.create(ctx, params)
		return stream, nil, err
	})
	late := time.NewTimer(s.hedgeAfter)
	defer late.Stop()

	var hedge *attempt
	tick, originalDone, hedgeDone := late.C, original.done, (<-chan struct{})(nil)
	for {
		select {
		case <-ctx.Done():
			original.abandon()
			hedge.abandon()
			return nil, stack.Wrap(ctx.Err())
		case <-tick:
			tick = nil
			hedge = race(ctx, func(ctx context.Context) (*llm.Stream, func(), error) {
				return s.askAnother(ctx, params)
			})
			hedgeDone = hedge.done
		case <-originalDone:
			originalDone = nil
			// Either it won, or it is out and there is no hedge still to wait for.
			if original.content || hedgeDone == nil {
				hedge.abandon()
				return original.outcome()
			}
		case <-hedgeDone:
			hedgeDone = nil
			if hedge.content {
				original.abandon()
				return hedge.outcome()
			}
			// It is out, so it is read out to its end while the original goes on alone.
			hedge.abandon()
			if originalDone == nil {
				return original.outcome()
			}
		}
	}
}

// askAnother opens the response on another candidate of the same target, which is a child of
// this session for as long as its stream runs.
func (s *Session) askAnother(ctx context.Context, params llm.ResponseParams) (*llm.Stream, func(), error) {
	child, err := s.hedge(ctx)
	if err != nil {
		return nil, nil, err
	}
	stream, err := child.create(ctx, params)
	if err != nil {
		s.releaseChild(child)
		return nil, nil, err
	}
	return stream, func() { s.releaseChild(child) }, nil
}

// hedgeCandidates are where a late request may be asked again, best first: never the
// candidate that was already asked, and none the router has found unavailable. The second
// request should share as little with the first as it can, so a different model comes first,
// then a different provider, and a different provider of the same model last.
func hedgeCandidates(candidates []routing.Candidate, asked routing.ProviderConfig) []routing.Candidate {
	others := slices.DeleteFunc(slices.Clone(candidates), func(candidate routing.Candidate) bool {
		return candidate.Config.Name() == asked.Name() || !candidate.Health.Available
	})
	apart := func(candidate routing.Candidate) (distance int) {
		if candidate.Config.Model != asked.Model {
			distance += 2
		}
		if candidate.Config.Provider != asked.Provider {
			distance++
		}
		return distance
	}
	slices.SortStableFunc(others, func(a, b routing.Candidate) int { return apart(b) - apart(a) })
	return others
}

// attempt is one request in a race. It runs on a goroutine of its own, because asking for a
// response can take as long as the answer does.
type attempt struct {
	cancel context.CancelFunc
	// done is closed once the request has failed, or its response has said something or
	// ended without.
	done chan struct{}

	// mu guards what abandon reads while the attempt is still running.
	mu     sync.Mutex
	stream *llm.Stream
	lost   bool
	// ended is set when the response ended without saying anything. It ended on its own, so
	// it is read out to the end rather than closed, and records as what it was.
	ended bool

	// These are written before done is closed, and read after.
	err     error
	content bool
	release func()
}

// opener asks for a response. It returns what is to be let go of once the stream has settled.
type opener func(context.Context) (stream *llm.Stream, release func(), err error)

// race starts an attempt.
func race(ctx context.Context, open opener) *attempt {
	ctx, cancel := context.WithCancel(ctx)
	a := &attempt{cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(a.done)
		stream, release, err := open(ctx)
		a.release, a.err = release, err
		if err != nil {
			return
		}
		a.mu.Lock()
		a.stream = stream
		lost := a.lost
		a.mu.Unlock()
		if lost {
			stream.Close()
		}
		content := stream.Await()
		a.mu.Lock()
		a.content, a.ended = content, !content
		a.mu.Unlock()
	}()
	return a
}

// outcome is what Create returns for this attempt: its stream, which lets go of the attempt
// once it has settled, or why there is none.
func (a *attempt) outcome() (*llm.Stream, error) {
	if a.err != nil {
		a.finish()
		return nil, a.err
	}
	return a.stream.Observe(func(event llm.Event) {
		if _, done := event.(llm.ResponseCompleted); done {
			a.finish()
		}
	}), nil
}

// abandon gives up on the attempt wherever it is. A request still being made is cancelled, and
// a stream that is open is closed. It is then read out to its end, which is what records it:
// a response cut short is still billed for what it had generated. It is safe to call on nil
// and more than once.
func (a *attempt) abandon() {
	if a == nil {
		return
	}
	a.mu.Lock()
	if a.lost {
		a.mu.Unlock()
		return
	}
	a.lost = true
	stream, ended := a.stream, a.ended
	a.mu.Unlock()

	a.cancel()
	if stream != nil && !ended {
		stream.Close()
	}
	go func() {
		<-a.done
		if a.stream != nil {
			for a.stream.Next() {
			}
		}
		a.finish()
	}()
}

// finish lets go of what the attempt held.
func (a *attempt) finish() {
	a.cancel()
	if a.release != nil {
		a.release()
	}
}
