package llmrouter

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// Session is a live model attached to one customer. It forwards the provider's events
// untouched and records a stat row per completed completion on the way past.
type Session struct {
	provider llm.LLM
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder

	events chan llm.Event

	mu sync.Mutex
	// inFlight tracks completions that have not been settled into a stat row yet.
	inFlight map[string]*turn

	closeOnce sync.Once
}

// turn is what the session remembers about one completion until it settles.
type turn struct {
	// startedAt is when the provider accepted it, so the stat row is stamped with when
	// the customer asked rather than when the answer finished.
	startedAt time.Time
	// errorCode is set when the provider reported a failure for this completion. It turns
	// the completion into a failed row rather than adding a second one.
	errorCode string
}

func newSession(
	provider llm.LLM,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	session := &Session{
		provider: provider,
		config:   config,
		owner:    owner,
		recorder: recorder,
		events:   make(chan llm.Event, 64),
		inFlight: map[string]*turn{},
	}
	go session.forward()
	return session
}

// Respond asks the selected provider for a completion.
func (s *Session) Respond(request llm.Request) error {
	return s.provider.Respond(request)
}

// Interrupt abandons the named completions, or every completion in flight when given
// none, so the agent can stop mid-sentence.
func (s *Session) Interrupt(completionIDs ...string) error {
	return s.provider.Interrupt(completionIDs...)
}

// Events returns the provider's events. The channel closes when the session closes.
func (s *Session) Events() <-chan llm.Event { return s.events }

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Reasoning reports whether the model streams its thinking before answering.
func (s *Session) Reasoning() bool { return s.provider.Reasoning() }

// Price is what this session's provider charges, so a caller can report a cost without
// reaching for the router's config.
func (s *Session) Price() routing.Price { return s.config.Price }

// LLM exposes the underlying provider so callers can reach provider-specific features.
func (s *Session) LLM() llm.LLM { return s.provider }

// Close ends the session. The event channel closes once the provider's events are drained.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() { err = s.provider.Close() })
	return err
}

// forward relays provider events and records statistics as they pass.
func (s *Session) forward() {
	defer close(s.events)

	for event := range s.provider.Events() {
		s.observe(event)
		s.events <- event
	}
}

func (s *Session) observe(event llm.Event) {
	switch typed := event.(type) {
	case llm.CompletionStarted:
		s.mu.Lock()
		s.inFlight[typed.CompletionID] = &turn{startedAt: typed.At}
		s.mu.Unlock()

	case llm.CompletionComplete:
		// One completion is one unit of billable work, the way one synthesis is for
		// text-to-speech. An interrupted one still generated tokens and still cost money.
		settled := s.settle(typed.CompletionID)
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: settled.startedAt,
			Usage: routing.Usage{
				InputTokens:       typed.InputTokens,
				CachedInputTokens: typed.CachedInputTokens,
				OutputTokens:      typed.OutputTokens,
			},
			// Time to first token is what the caller actually waited for; the rest of the
			// answer arrives while they are already reading or hearing it.
			LatencyMs: typed.TimeToFirstTokenMs,
			Success:   settled.errorCode == "",
			ErrorCode: settled.errorCode,
		})

	case llm.Error:
		// A failure that names a completion is settled by that completion, so one turn is
		// still one row. Anything else is a session-level failure.
		if typed.CompletionID != "" && s.fail(typed.CompletionID, errorCode(typed)) {
			return
		}
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: time.Now().UTC(),
			Success:   false,
			ErrorCode: errorCode(typed),
		})
	}
}

// fail marks a completion as failed, reporting whether it was still in flight.
func (s *Session) fail(completionID, code string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.inFlight[completionID]
	if !ok {
		return false
	}
	// The first failure is the one that explains the completion.
	if current.errorCode == "" {
		current.errorCode = code
	}
	return true
}

// settle takes a completion out of flight, so a late duplicate cannot bill twice. A
// completion the session never saw start is stamped with now.
func (s *Session) settle(completionID string) turn {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.inFlight[completionID]
	if !ok {
		return turn{startedAt: time.Now().UTC()}
	}
	delete(s.inFlight, completionID)
	return turn{startedAt: current.startedAt.UTC(), errorCode: current.errorCode}
}

func errorCode(failure llm.Error) string {
	if failure.Fatal {
		return "provider_fatal"
	}
	if failure.Context != "" {
		return failure.Context
	}
	return "provider_error"
}

// Await waits for the one completion an off-conversation pass asked for.
//
// It is for the model calls nobody is streaming: a reviewer summarising a call, a judge
// ruling on one, a model asked to rewrite a scenario. Every one of them sends a request and
// wants the whole answer, so they all wrote this loop until it was moved here.
func Await(ctx context.Context, session *Session, id string) (string, error) {
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case event, open := <-session.Events():
			if !open {
				return "", errors.New("llmrouter: the model closed before answering")
			}
			switch typed := event.(type) {
			case llm.CompletionComplete:
				if typed.CompletionID == id {
					return typed.Text, nil
				}
			case llm.Error:
				if typed.CompletionID == "" || typed.CompletionID == id {
					return "", typed.Err
				}
			}
		}
	}
}
