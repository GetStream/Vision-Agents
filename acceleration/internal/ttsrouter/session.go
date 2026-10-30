package ttsrouter

import (
	"log/slog"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/google/uuid"
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
	logger   *slog.Logger

	events chan tts.Event
	// forwarded closes once every event the provider sent has been recorded, and stopped once
	// the session is closed.
	forwarded chan struct{}
	stopped   chan struct{}

	mu sync.Mutex
	// inFlight tracks utterances that have not been settled into a stat row yet.
	inFlight map[string]*utterance
	// sent is the characters handed to the provider for each utterance not settled yet,
	// which is what it is billed for, whatever the adapter counted.
	sent map[string]int64
	// reserved is the usage id each utterance not settled yet holds its characters under,
	// which its row is recorded under.
	reserved map[string]uuid.UUID
	// finished is the most recent utterances settled, oldest first in finishedOrder, so a
	// completion a voice repeats for one of them bills nothing. Text sent for one after it
	// settled is reserved and counted as an utterance of its own, since a voice that speaks it
	// bills it; what a voice drops (an interrupted utterance takes no more) stays reserved until
	// the session closes. Only the last finishedKept are kept.
	finished      map[string]struct{}
	finishedOrder []string

	closeOnce sync.Once
}

// finishedKept is how many settled utterances a session remembers. Late text belongs to the
// last few, well inside it.
const finishedKept = 256

// miscounted is the provider and model names whose adapter was seen counting characters other
// than those sent, each logged once a process.
var miscounted sync.Map

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
	logger *slog.Logger,
) *Session {
	// A voice that takes each sentence as its own request works on them side by side, and
	// heard as it arrives their audio is spliced together.
	if !provider.Streaming() {
		provider = tts.InOrder(provider)
	}
	session := &Session{
		provider:  provider,
		config:    config,
		owner:     owner,
		recorder:  recorder,
		logger:    logger,
		events:    make(chan tts.Event, 64),
		forwarded: make(chan struct{}),
		stopped:   make(chan struct{}),
		inFlight:  map[string]*utterance{},
		sent:      map[string]int64{},
		reserved:  map[string]uuid.UUID{},
		finished:  map[string]struct{}{},
	}
	go session.forward()
	return session
}

// Synthesize sends text to the selected provider. A request with no id is given one here
// rather than by the provider, so what is sent can be counted against the utterance it
// settles as.
//
// What an utterance's text costs is reserved before it is sent, under the id its row is
// recorded under, and raised as more of its text follows. A refusal of its first text is
// answered and nothing is sent; a refusal of more text for an utterance already speaking
// keeps what it reserved and lets it finish.
func (s *Session) Synthesize(request tts.Request) error {
	if request.ID == "" {
		request.ID = tts.NewSynthesisID()
	}
	characters := int64(utf8.RuneCountInString(request.Text))
	s.mu.Lock()
	reserved, speaking := s.reserved[request.ID]
	// Counted before it is sent, so a completion that comes straight back finds it.
	s.sent[request.ID] += characters
	if !speaking {
		reserved = uuid.Must(uuid.NewV7())
		s.reserved[request.ID] = reserved
	}
	total := s.sent[request.ID]
	s.mu.Unlock()

	var err error
	if characters > 0 {
		err = s.recorder.Reserve(s.config, s.owner, reserved, pricing.Quantities{pricing.Characters: total}, routing.ReserveTTL)
		if err != nil && speaking {
			err = nil
		}
	}
	if err == nil {
		err = s.provider.Synthesize(request)
	}
	if err != nil && characters > 0 {
		// Refused, it was never sent: what is held goes back to what was.
		s.mu.Lock()
		if s.sent[request.ID] -= characters; s.sent[request.ID] <= 0 {
			delete(s.sent, request.ID)
		}
		left := s.sent[request.ID]
		if !speaking || left <= 0 {
			delete(s.reserved, request.ID)
		}
		s.mu.Unlock()
		if !speaking || left <= 0 {
			s.recorder.Release(reserved)
		} else {
			_ = s.recorder.Reserve(s.config, s.owner, reserved, pricing.Quantities{pricing.Characters: left}, routing.ReserveTTL)
		}
	}
	return err
}

// Interrupt drops audio in flight so the agent can stop mid-sentence.
func (s *Session) Interrupt() error { return s.provider.Interrupt() }

// Events returns the provider's events. The channel closes when the session closes.
func (s *Session) Events() <-chan tts.Event { return s.events }

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Voice is the voice speaking, the provider's default included. Empty when the provider
// cannot say.
func (s *Session) Voice() string {
	if voiced, ok := s.provider.(tts.Voiced); ok {
		return voiced.Voice()
	}
	return ""
}

// Streaming reports whether the provider accepts partial text deltas.
func (s *Session) Streaming() bool { return s.provider.Streaming() }

// Prompt is what the model writing this voice's lines should be told about it.
func (s *Session) Prompt() string { return s.provider.Prompt() }

// Performs reports whether the voice acts bracketed directions rather than reading them.
func (s *Session) Performs() bool { return s.provider.Performs() }

// Quote is what that usage of this session's provider costs, priced as its row is, so a
// caller can report a cost without reaching for the router's prices.
func (s *Session) Quote(usage routing.Usage) pricing.Quote { return s.recorder.Quote(s.config, usage) }

// Close ends the session. The event channel closes once the provider's events are drained.
// What is still reserved for utterances that never completed is let go once they are.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() {
		close(s.stopped)
		err = s.provider.Close()
		go func() {
			<-s.forwarded
			s.mu.Lock()
			reserved := s.reserved
			s.reserved = map[string]uuid.UUID{}
			s.mu.Unlock()
			for _, id := range reserved {
				s.recorder.Release(id)
			}
		}()
	})
	return err
}

// forward relays provider events and records statistics as they pass.
func (s *Session) forward() {
	defer close(s.forwarded)
	defer close(s.events)

	for event := range s.provider.Events() {
		s.observe(event)
		select {
		case s.events <- event:
			continue
		default:
		}
		// Once the session is closed a caller that stopped reading drops what is left, which
		// is still recorded, rather than holding the provider's events, and what they settle,
		// forever.
		select {
		case s.events <- event:
		case <-s.stopped:
		}
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
		settled, repeated := s.settle(typed.SynthesisID)
		if repeated {
			return
		}
		characters := typed.Characters
		if settled.sent {
			characters = settled.characters
			if characters != typed.Characters {
				s.warnMiscounted(typed.Characters, characters)
			}
		}
		s.recorder.Record(s.config, routing.Stat{
			Owner:       s.owner,
			UsageID:     settled.reserved,
			StartedAt:   settled.startedAt,
			OperationID: typed.SynthesisID,
			Purpose:     "speech",
			DurationMs:  float64(time.Since(settled.startedAt).Microseconds()) / 1000,
			Usage: routing.Usage{
				Characters: characters,
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

// settled is an utterance taken out of flight, and the characters sent for it.
type settled struct {
	utterance
	characters int64
	// sent is false for an utterance that never went through Synthesize here, whose
	// characters only the adapter counted.
	sent bool
	// reserved is the usage id its characters were reserved under, nil when none were.
	reserved uuid.UUID
}

// settle takes an utterance out of flight, so a late duplicate completion cannot bill
// twice: one for an utterance already settled that has not started again, and was sent no
// text since, is reported repeated, and records nothing. An utterance the session never saw
// start is stamped with now.
func (s *Session) settle(synthesisID string) (settled, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.inFlight[synthesisID]
	characters, sent := s.sent[synthesisID]
	if _, known := s.finished[synthesisID]; known && !ok && characters == 0 {
		return settled{}, true
	}
	delete(s.sent, synthesisID)
	reserved := s.reserved[synthesisID]
	delete(s.reserved, synthesisID)
	if _, known := s.finished[synthesisID]; !known {
		s.finished[synthesisID] = struct{}{}
		s.finishedOrder = append(s.finishedOrder, synthesisID)
		if len(s.finishedOrder) > finishedKept {
			delete(s.finished, s.finishedOrder[0])
			s.finishedOrder = s.finishedOrder[1:]
		}
	}
	if !ok {
		return settled{utterance: utterance{startedAt: time.Now().UTC()}, characters: characters, sent: sent, reserved: reserved}, false
	}
	delete(s.inFlight, synthesisID)
	return settled{
		utterance:  utterance{startedAt: current.startedAt.UTC(), errorCode: current.errorCode},
		characters: characters,
		sent:       sent,
		reserved:   reserved,
	}, false
}

// warnMiscounted logs, once a process for each provider and model, an adapter counting
// characters other than those it was sent.
func (s *Session) warnMiscounted(counted, sent int64) {
	if _, logged := miscounted.LoadOrStore(s.config.Name(), true); logged {
		return
	}
	s.logger.Warn("the voice counted characters other than those it was sent; the row bills what was sent",
		"provider", s.config.Provider, "model", s.config.Model, "counted", counted, "sent", sent)
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
