package stsrouter

import (
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

// Session is a live conversation attached to one customer. It forwards the provider's
// events and records a stat row per completed reply on the way past.
//
// It also keeps the one promise the contract makes about barge-in: once a reply has been
// reported interrupted, or the caller has asked for it to be, no more of its audio gets
// through. A provider learns of an interrupt one round trip after the caller, and the
// chunks in that gap would otherwise play as a tail on the words the caller cut off.
type Session struct {
	provider sts.STS
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder

	events chan sts.Event

	mu sync.Mutex
	// inFlight tracks replies that have not been settled into a stat row yet.
	inFlight map[string]*reply
	// heardMs is the caller's audio forwarded since the last reply settled, which is the
	// audio the next reply is billed against.
	heardMs float64
	// current is the generation of the reply in flight, and staleBelow the highest one
	// whose audio is no longer wanted.
	current    int
	staleBelow int

	closeOnce sync.Once
}

// reply is what the session remembers about one response until it completes.
type reply struct {
	// startedAt is when the model began, so the stat row is stamped with when the customer
	// was answered rather than when the audio finished.
	startedAt time.Time
	// errorCode is set when the provider reported a failure for this reply. It turns the
	// completion into a failed row rather than adding a second one.
	errorCode string
}

func newSession(
	provider sts.STS,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	session := &Session{
		provider: provider,
		config:   config,
		owner:    owner,
		recorder: recorder,
		events:   make(chan sts.Event, sts.EmitterBuffer),
		inFlight: map[string]*reply{},
	}
	go session.forward()
	return session
}

// ProcessAudio streams the caller's speech to the selected provider.
func (s *Session) ProcessAudio(pcm sts.PcmData, participant sts.Participant) error {
	s.mu.Lock()
	s.heardMs += pcm.DurationMs()
	s.mu.Unlock()
	return s.provider.ProcessAudio(pcm, participant)
}

// SendText injects a typed turn.
func (s *Session) SendText(text string, participant sts.Participant) error {
	return s.provider.SendText(text, participant)
}

// SendFrame offers the model a still image.
func (s *Session) SendFrame(frame llm.ImagePart) error { return s.provider.SendFrame(frame) }

// SetInstructions changes the system prompt, where the model allows it.
func (s *Session) SetInstructions(text string) error { return s.provider.SetInstructions(text) }

// SetTools replaces what the model may call, where the model allows it.
func (s *Session) SetTools(tools []llm.Tool) error { return s.provider.SetTools(tools) }

// Answer returns what a tool produced.
func (s *Session) Answer(callID string, output string, err error) error {
	return s.provider.Answer(callID, output, err)
}

// Prompt asks the model to speak, guided by the text.
func (s *Session) Prompt(text string) error { return s.provider.Prompt(text) }

// Interrupt stops the reply in flight. Its audio stops being forwarded at once, before the
// provider has heard about it, since the caller has already stopped listening.
func (s *Session) Interrupt(playedMs int) error {
	s.mu.Lock()
	if s.current > s.staleBelow {
		s.staleBelow = s.current
	}
	s.mu.Unlock()
	return s.provider.Interrupt(playedMs)
}

// Events returns the provider's events. The channel closes when the session closes.
func (s *Session) Events() <-chan sts.Event { return s.events }

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// SampleRate is the rate the model speaks at.
func (s *Session) SampleRate() int { return s.provider.SampleRate() }

// Capabilities is what the model serving this session can be asked for.
func (s *Session) Capabilities() sts.Capabilities { return s.provider.Capabilities() }

// Price is what this session's provider charges, so a caller can report a cost without
// reaching for the router's config.
func (s *Session) Price() routing.Price { return s.config.Price }

// STS exposes the underlying provider so callers can reach provider-specific features.
func (s *Session) STS() sts.STS { return s.provider }

// Close ends the session. The event channel closes once the provider's events are drained.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() { err = s.provider.Close() })
	return err
}

// forward relays provider events, recording statistics and dropping stale audio as they
// pass.
func (s *Session) forward() {
	defer close(s.events)

	for event := range s.provider.Events() {
		if s.observe(event) {
			s.events <- event
		}
	}
}

// observe records what an event means for the books and reports whether it is forwarded.
func (s *Session) observe(event sts.Event) bool {
	switch typed := event.(type) {
	case sts.ResponseStarted:
		s.mu.Lock()
		s.inFlight[typed.ResponseID] = &reply{startedAt: typed.At}
		s.current = typed.Generation
		s.mu.Unlock()

	case sts.AudioChunk:
		s.mu.Lock()
		stale := typed.Generation <= s.staleBelow
		s.mu.Unlock()
		return !stale

	case sts.ResponseComplete:
		// One reply is one unit of billable work, the way one utterance is for a voice.
		// An interrupted one still cost the model's time and still cost money.
		settled, heardMs := s.settle(typed)
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: settled.startedAt,
			Usage: routing.Usage{
				AudioMs:           int64(heardMs),
				InputTokens:       typed.Usage.InputTokens,
				CachedInputTokens: typed.Usage.CachedInputTokens,
				OutputTokens:      typed.Usage.OutputTokens,
			},
			// Time to first byte is what the caller actually waited for; the rest of the
			// reply arrives while they are already hearing it.
			LatencyMs: typed.TimeToFirstByteMs,
			Success:   settled.errorCode == "",
			ErrorCode: settled.errorCode,
		})

	case sts.Error:
		// A failure that names a reply is settled by that reply's completion, so one
		// reply is still one row. Anything else is a session-level failure.
		if typed.ResponseID != "" && s.fail(typed.ResponseID, errorCode(typed)) {
			return true
		}
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: time.Now().UTC(),
			Success:   false,
			ErrorCode: errorCode(typed),
		})
	}
	return true
}

// fail marks a reply as failed, reporting whether it was still in flight.
func (s *Session) fail(responseID, code string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.inFlight[responseID]
	if !ok {
		return false
	}
	// The first failure is the one that explains the reply.
	if current.errorCode == "" {
		current.errorCode = code
	}
	return true
}

// settle takes a reply out of flight, so a late duplicate completion cannot bill twice, and
// claims the caller's audio heard since the last one. An interrupted reply also marks its
// generation stale, so audio still arriving for it is dropped. A reply the session never
// saw start is stamped with now.
func (s *Session) settle(complete sts.ResponseComplete) (reply, float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	heardMs := s.heardMs
	s.heardMs = 0
	if complete.Interrupted && complete.Generation > s.staleBelow {
		s.staleBelow = complete.Generation
	}

	current, ok := s.inFlight[complete.ResponseID]
	if !ok {
		return reply{startedAt: time.Now().UTC()}, heardMs
	}
	delete(s.inFlight, complete.ResponseID)
	return reply{startedAt: current.startedAt.UTC(), errorCode: current.errorCode}, heardMs
}

func errorCode(failure sts.Error) string {
	if failure.Fatal {
		return "provider_fatal"
	}
	if failure.Context != "" {
		return failure.Context
	}
	return "provider_error"
}
