package ttsrouter

import (
	"context"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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
	// performs and prompt are what a voice that acts stage directions reports.
	performs bool
	prompt   string
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
func (s *stubTTS) Performs() bool   { return s.performs }
func (s *stubTTS) Prompt() string   { return s.prompt }

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

// newStubbedRouter routes over the built-in config with every real voice replaced by a
// stub, so where a request lands can be asked of the deployment's own config without a
// key for any of the vendors in it. Every spec built is recorded, which is how what a
// provider was asked for is checked.
func (s *TTSRouterSuite) newStubbedRouter(built *[]routing.Spec) *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	registry := routing.NewRegistry[tts.TTS]()
	for _, provider := range config[routing.TTS].Providers {
		registry.Register(provider.Provider, func(spec routing.Spec) (tts.TTS, error) {
			*built = append(*built, spec)
			return newStubTTS(), nil
		})
	}

	router, err := New(Options{Config: config[routing.TTS], Registry: registry})
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

func (s *TTSRouterSuite) TestAPriorityListIsTriedInTheOrderItWasWritten() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Options:    options.TTS{Providers: []string{"inworld", "cartesia"}},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("inworld", session.Provider(),
		"a caller who wrote an order wants the second name only once the first is down")
}

func (s *TTSRouterSuite) TestAVendorNamedForALiveCallGetsTheirStreamingModel() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	// ElevenLabs has four models here, and eleven_v3 is the one that returns a file
	// rather than streaming. A socket asking for the vendor by name must not get it.
	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Options:    options.TTS{Providers: []string{"elevenlabs"}},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs", session.Provider())
	s.NotEqual("eleven_v3", session.Model(), "the batch model cannot serve a live socket")
}

func (s *TTSRouterSuite) TestAVoiceThatTrainsOnWhatItIsSentIsNotAskedToSpeakForACallerWhoRefused() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	no := false

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "en-low-latency",
		Options:    options.TTS{DataPolicy: options.DataPolicy{AllowTraining: &no}},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	config, ok := router.Config().Provider(session.Provider() + "/" + session.Model())
	s.Require().True(ok)
	s.Equal(options.ClaimNo, config.DataPolicy.TrainsOnData)
}

func (s *TTSRouterSuite) TestAPolicyNoVoiceMeetsIsRefusedRatherThanServedAnyway() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	no := false

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Options: options.TTS{
			Providers:  []string{"elevenlabs"},
			DataPolicy: options.DataPolicy{AllowTraining: &no},
		},
	})

	s.Error(err, "speaking somewhere the caller ruled out is worse than not speaking")
	s.Empty(built, "nothing should have been built for a request nobody may serve")
}

func (s *TTSRouterSuite) TestAnOverwriteReachesTheVendorItNames() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      "library-voice",
		Options: options.TTS{
			Providers: []string{"inworld"},
			Overwrites: map[string]json.RawMessage{
				"inworld":    json.RawMessage(`{"delivery_mode":"STABLE"}`),
				"elevenlabs": json.RawMessage(`{"voice_id":"el-1"}`),
			},
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Require().Len(built, 1)
	s.JSONEq(`{"delivery_mode":"STABLE"}`, string(built[0].Overwrites),
		"a vendor is handed its own block and nobody else's")
}

func (s *TTSRouterSuite) TestRegistryReadsTheDeliveryModeAndVoiceFromOverwrites() {
	var inworldSaid inworldSettings
	s.Require().NoError(routing.Spec{
		Overwrites: json.RawMessage(`{"delivery_mode":"CREATIVE","voice_id":"Ashley"}`),
	}.Settings(&inworldSaid))
	s.Equal("CREATIVE", inworldSaid.DeliveryMode)
	s.Equal("Ashley", inworldSaid.VoiceID)

	var elevenlabsSaid elevenlabsSettings
	s.Require().NoError(routing.Spec{
		Overwrites: json.RawMessage(`{"voice_id":"el-1"}`),
	}.Settings(&elevenlabsSaid))
	s.Equal("el-1", elevenlabsSaid.VoiceID)
}

func (s *TTSRouterSuite) TestAVendorsOwnVoiceIdWinsOverTheOneTheRequestAsked() {
	// A voice id from one library means nothing at another, so a config that routes
	// between vendors names the voice per vendor and the one chosen is the one read.
	s.Equal("el-1", voiceOr("el-1", "founder"))
	s.Equal("founder", voiceOr("", "founder"))
}

func (s *TTSRouterSuite) TestRegistryRefusesAnOverwriteTheVendorHasNoFieldFor() {
	registry := DefaultRegistry()
	s.T().Setenv("INWORLD_API_KEY", "test-key")

	_, err := registry.Build("inworld", routing.Spec{
		Model:      "inworld-tts-2-flash",
		Overwrites: json.RawMessage(`{"delivery_moode":"STABLE"}`),
	})

	s.ErrorContains(err, "delivery_moode",
		"a misspelt setting has to be reported, since the alternative is silently not sending it")
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
