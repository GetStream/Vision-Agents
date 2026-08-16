package ttsrouter

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// stubTTS stands in for a real provider so a session can be driven without credentials.
type stubTTS struct {
	emitter *tts.Emitter
	said    []tts.Request
	// interrupts counts barge-ins, so a session can be checked to forward them.
	interrupts int
	closed     bool
}

func newStubTTS() *stubTTS {
	return &stubTTS{emitter: tts.NewEmitter(64)}
}

func (s *stubTTS) Start(context.Context) error { return nil }

func (s *stubTTS) Synthesize(request tts.Request) error {
	s.said = append(s.said, request)
	return nil
}

func (s *stubTTS) Interrupt() error { s.interrupts++; return nil }

func (s *stubTTS) Events() <-chan tts.Event { return s.emitter.Events() }

func (s *stubTTS) Close() error {
	s.closed = true
	s.emitter.Close()
	return nil
}

func (s *stubTTS) Provider() string { return "stub" }
func (s *stubTTS) Model() string    { return "stub-model" }
func (s *stubTTS) Streaming() bool  { return true }

type TTSRouterSuite struct {
	suite.Suite
	ctx context.Context
}

func TestTTSRouterSuite(t *testing.T) {
	suite.Run(t, new(TTSRouterSuite))
}

func (s *TTSRouterSuite) SetupTest() {
	s.ctx = context.Background()
}

// newRouter routes over the built-in text-to-speech config, which is what a deployment
// gets when it sets no config file.
func (s *TTSRouterSuite) newRouter() *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	router, err := New(Options{Config: config[routing.TTS], Registry: DefaultRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// newSession returns a session over a stub provider, so event handling is the only thing
// under test.
func (s *TTSRouterSuite) newSession() (*Session, *stubTTS) {
	config := routing.ProviderConfig{
		Provider: "stub",
		Model:    "stub-model",
		Price:    routing.Price{PerMillionChars: 50},
	}
	return s.sessionFor(config)
}

func (s *TTSRouterSuite) sessionFor(config routing.ProviderConfig) (*Session, *stubTTS) {
	provider := newStubTTS()
	recorder := routing.NewRecorder(routing.TTS, nil, nil, slog.Default())
	session := newSession(provider, config, routing.Owner{CustomerID: "acme"}, recorder)

	s.T().Cleanup(func() {
		_ = session.Close()
		recorder.Close()
	})
	return session, provider
}

// drain reads the session's forwarded events until the channel closes.
func (s *TTSRouterSuite) drain(session *Session) []tts.Event {
	var events []tts.Event
	for {
		select {
		case event, open := <-session.Events():
			if !open {
				return events
			}
			events = append(events, event)
		case <-time.After(5 * time.Second):
			s.FailNow("timed out draining the session")
			return events
		}
	}
}

func (s *TTSRouterSuite) TestRouterServesTheTextToSpeechModality() {
	s.Equal(routing.TTS, s.newRouter().Modality())
}

func (s *TTSRouterSuite) TestEveryShortcutResolvesToAProvider() {
	router := s.newRouter()

	for alias := range router.Config().Aliases {
		candidates, err := router.Resolve(s.ctx, alias, nil)
		s.Require().NoErrorf(err, "alias %s", alias)
		s.NotEmptyf(candidates, "alias %s", alias)
	}
}

func (s *TTSRouterSuite) TestLowLatencyAndQualityShortcutsPickDifferentModels() {
	router := s.newRouter()

	fast, err := router.Resolve(s.ctx, "en-low-latency", nil)
	s.Require().NoError(err)
	good, err := router.Resolve(s.ctx, "en-high-accuracy", nil)
	s.Require().NoError(err)

	s.Require().NotEmpty(fast)
	s.Require().NotEmpty(good)
	for _, candidate := range fast {
		s.Equal(routing.LowLatency, candidate.Config.Tier)
	}
	for _, candidate := range good {
		s.Equal(routing.HighQuality, candidate.Config.Tier)
	}
}

func (s *TTSRouterSuite) TestUnservableLanguageIsRejected() {
	_, err := s.newRouter().Resolve(s.ctx, "en-low-latency", []string{"tlh"})
	s.ErrorContains(err, "no provider satisfies")
}

func (s *TTSRouterSuite) TestRegistryKnowsEveryConfiguredProvider() {
	router := s.newRouter()
	registry := DefaultRegistry()

	for _, provider := range router.Config().Providers {
		s.Truef(registry.Has(provider.Provider),
			"%s is configured but has no factory, so it can never serve a request", provider.Provider)
	}
}

func (s *TTSRouterSuite) TestRegistryPassesTheVoiceAndLanguageToTheProvider() {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	registry := DefaultRegistry()

	built, err := registry.Build("elevenlabs", routing.Spec{
		Model:         "eleven_multilingual_v2",
		Voice:         "chosen-voice",
		LanguageHints: []string{"es", "fr"},
	})
	s.Require().NoError(err)
	s.Equal("eleven_multilingual_v2", built.Model())
}

func (s *TTSRouterSuite) TestStartRequiresACustomer() {
	_, err := s.newRouter().Start(s.ctx, Request{Target: "en-low-latency"})
	s.ErrorContains(err, "customer id is required")
}

func (s *TTSRouterSuite) TestSessionReportsTheRoutingIdentityRatherThanTheProvidersOwn() {
	session, _ := s.sessionFor(routing.ProviderConfig{Provider: "fish", Model: "s2-pro"})

	s.Equal("fish", session.Provider(), "stats are keyed by the configured name")
	s.Equal("s2-pro", session.Model())
}

func (s *TTSRouterSuite) TestSessionForwardsTextAndBargeInToTheProvider() {
	session, provider := s.newSession()

	s.Require().NoError(session.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.Require().NoError(session.Interrupt())

	s.Require().Len(provider.said, 1)
	s.Equal("hello", provider.said[0].Text)
	s.Equal(1, provider.interrupts)
	s.True(session.Streaming())
}

func (s *TTSRouterSuite) TestSessionForwardsProviderEventsUntouched() {
	session, provider := s.newSession()

	pcm := audio.PcmData{Samples: make([]int16, 2400), SampleRate: 24_000, Channels: 1}
	provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	provider.emitter.Send(tts.AudioChunk{SynthesisID: "u1", Index: 0, Audio: pcm})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "u1", Characters: 11, AudioDurationMs: 100})
	s.Require().NoError(session.Close())

	events := s.drain(session)
	s.Require().Len(events, 3)
	chunk, ok := events[1].(tts.AudioChunk)
	s.Require().True(ok)
	s.Equal(2400, len(chunk.Audio.Samples), "audio should reach the caller unchanged")
}

func (s *TTSRouterSuite) TestSessionClosesItsEventChannelWithTheProvider() {
	session, _ := s.newSession()

	s.Require().NoError(session.Close())

	s.Empty(s.drain(session))
}

func (s *TTSRouterSuite) TestCloseIsIdempotent() {
	session, provider := s.newSession()

	s.Require().NoError(session.Close())
	s.Require().NoError(session.Close())
	s.True(provider.closed)
}

func (s *TTSRouterSuite) TestAnUtteranceIsStampedWithWhenTheCustomerAsked() {
	session, _ := s.newSession()
	askedAt := time.Now().Add(-2 * time.Second)

	session.observe(tts.SynthesisStarted{SynthesisID: "u1", At: askedAt})
	settled := session.settle("u1")

	s.WithinDuration(askedAt.UTC(), settled.startedAt, time.Millisecond,
		"a stat row should cover the whole wait, not just the audio")
}

func (s *TTSRouterSuite) TestAnUtteranceTheSessionNeverSawStartIsStillTimed() {
	session, _ := s.newSession()

	settled := session.settle("never-seen")

	s.WithinDuration(time.Now().UTC(), settled.startedAt, time.Second)
	s.Empty(settled.errorCode)
}

func (s *TTSRouterSuite) TestAFailedUtteranceIsOneFailedRowRatherThanTwo() {
	session, _ := s.newSession()

	session.observe(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	session.observe(tts.Error{SynthesisID: "u1", Err: context.Canceled, Context: "audio"})

	// The failure is remembered, not recorded, so the completion is what settles it.
	settled := session.settle("u1")
	s.Equal("audio", settled.errorCode,
		"the completion should report the failure instead of adding a second row")
}

func (s *TTSRouterSuite) TestTheFirstFailureIsTheOneThatExplainsAnUtterance() {
	session, _ := s.newSession()

	session.observe(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	session.observe(tts.Error{SynthesisID: "u1", Err: context.Canceled, Context: "request"})
	session.observe(tts.Error{SynthesisID: "u1", Err: context.Canceled, Context: "audio"})

	s.Equal("request", session.settle("u1").errorCode)
}

func (s *TTSRouterSuite) TestSettlingAnUtteranceTwiceCannotBillItTwice() {
	session, _ := s.newSession()
	askedAt := time.Now().Add(-time.Second)

	session.observe(tts.SynthesisStarted{SynthesisID: "u1", At: askedAt})
	s.WithinDuration(askedAt.UTC(), session.settle("u1").startedAt, time.Millisecond)

	// The second settle finds nothing in flight and so cannot reuse the utterance.
	s.WithinDuration(time.Now().UTC(), session.settle("u1").startedAt, time.Second)
}

func (s *TTSRouterSuite) TestAFatalErrorIsGradedWorseThanAContextualOne() {
	s.Equal("provider_fatal", errorCode(tts.Error{Fatal: true, Context: "read"}))
	s.Equal("read", errorCode(tts.Error{Context: "read"}))
	s.Equal("provider_error", errorCode(tts.Error{}))
}

func (s *TTSRouterSuite) TestSessionExposesThePriceItWillBeBilledAt() {
	session, _ := s.newSession()

	s.EqualValues(50_000, session.Price().CostMicros(routing.Usage{Characters: 1_000}),
		"a thousand characters is five cents")
}
