package ttsrouter

import (
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// Session is a live voice attached to one customer. It forwards the provider's events
// untouched and records a stat row per completed synthesis on the way past.
type Session struct {
	provider tts.TTS
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder

	events chan tts.Event

	mu sync.Mutex
	// inFlight tracks utterances that have not been settled into a stat row yet.
	inFlight map[string]*utterance

	closeOnce sync.Once
}

// utterance is what the session remembers about one synthesis until it completes.
type utterance struct {
	// startedAt is when the provider accepted it, so the stat row is stamped with when
	// the customer asked rather than when the audio finished.
	startedAt time.Time
	// errorCode is set when the provider reported a failure for this synthesis. It turns
	// the completion into a failed row rather than adding a second one.
	errorCode string
}

func newSession(
	provider tts.TTS,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	session := &Session{
		provider: provider,
		config:   config,
		owner:    owner,
		recorder: recorder,
		events:   make(chan tts.Event, 64),
		inFlight: map[string]*utterance{},
	}
	go session.forward()
	return session
}

// Synthesize sends text to the selected provider.
func (s *Session) Synthesize(request tts.Request) error {
	return s.provider.Synthesize(request)
}

// Interrupt drops audio in flight so the agent can stop mid-sentence.
func (s *Session) Interrupt() error { return s.provider.Interrupt() }

// Events returns the provider's events. The channel closes when the session closes.
func (s *Session) Events() <-chan tts.Event { return s.events }

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Streaming reports whether the provider accepts partial text deltas.
func (s *Session) Streaming() bool { return s.provider.Streaming() }

// Price is what this session's provider charges, so a caller can report a cost without
// reaching for the router's config.
func (s *Session) Price() routing.Price { return s.config.Price }

// TTS exposes the underlying provider so callers can reach provider-specific features.
func (s *Session) TTS() tts.TTS { return s.provider }

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

func (s *Session) observe(event tts.Event) {
	switch typed := event.(type) {
	case tts.SynthesisStarted:
		s.mu.Lock()
		s.inFlight[typed.SynthesisID] = &utterance{startedAt: typed.At}
		s.mu.Unlock()

	case tts.SynthesisComplete:
		// One utterance is one unit of billable work, the way one turn is for
		// speech-to-text. An interrupted one still produced audio and still cost money.
		settled := s.settle(typed.SynthesisID)
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: settled.startedAt,
			Usage: routing.Usage{
				Characters: typed.Characters,
				AudioMs:    int64(typed.AudioDurationMs),
			},
			// Time to first byte is what the listener actually waited for; the rest of
			// the audio arrives while they are already hearing it.
			LatencyMs: typed.TimeToFirstByteMs,
			Success:   settled.errorCode == "",
			ErrorCode: settled.errorCode,
		})

	case tts.Error:
		// A failure that names an utterance is settled by that utterance's completion, so
		// one synthesis is still one row. Anything else is a session-level failure.
		if typed.SynthesisID != "" && s.fail(typed.SynthesisID, errorCode(typed)) {
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

// fail marks an utterance as failed, reporting whether it was still in flight.
func (s *Session) fail(synthesisID, code string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.inFlight[synthesisID]
	if !ok {
		return false
	}
	// The first failure is the one that explains the utterance.
	if current.errorCode == "" {
		current.errorCode = code
	}
	return true
}

// settle takes an utterance out of flight, so a late duplicate completion cannot bill
// twice. An utterance the session never saw start is stamped with now.
func (s *Session) settle(synthesisID string) utterance {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.inFlight[synthesisID]
	if !ok {
		return utterance{startedAt: time.Now().UTC()}
	}
	delete(s.inFlight, synthesisID)
	return utterance{startedAt: current.startedAt.UTC(), errorCode: current.errorCode}
}

func errorCode(failure tts.Error) string {
	if failure.Fatal {
		return "provider_fatal"
	}
	if failure.Context != "" {
		return failure.Context
	}
	return "provider_error"
}
