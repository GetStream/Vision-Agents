package ttsrouter

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
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
	// perSentence is a voice that takes each sentence as its own request.
	perSentence bool
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
func (s *stubTTS) Streaming() bool  { return !s.perSentence }
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
	return s.newStubbedRouterResolving(built, nil)
}

// newStubbedRouterResolving is newStubbedRouter with a voice resolver in front of every
// candidate, as cmd/router builds one.
func (s *TTSRouterSuite) newStubbedRouterResolving(built *[]routing.Spec, resolver routing.VoiceResolver) *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	registry := routing.NewRegistry[tts.TTS]()
	for _, provider := range config[routing.TTS].Providers {
		registry.Register(provider.Provider, func(spec routing.Spec) (tts.TTS, error) {
			*built = append(*built, spec)
			return newStubTTS(), nil
		})
	}

	router, err := New(Options{Config: config[routing.TTS], Registry: registry, Voices: resolver})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// newRouterOverRealElevenLabs is newStubbedRouter with ElevenLabs' own factory in front of
// its stub, so what that factory refuses is refused here, while nothing is dialled. Every
// candidate built is recorded as provider/model.
func (s *TTSRouterSuite) newRouterOverRealElevenLabs(built *[]string) *Router {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	vendor := DefaultRegistry()
	registry := routing.NewRegistry[tts.TTS]()
	for _, provider := range config[routing.TTS].Providers {
		registry.Register(provider.Provider, func(spec routing.Spec) (tts.TTS, error) {
			*built = append(*built, provider.Provider+"/"+spec.Model)
			if provider.Provider == "elevenlabs" {
				if _, err := vendor.Build("elevenlabs", spec); err != nil {
					return nil, err
				}
			}
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
	return s.sessionOver(newStubTTS(), config)
}

func (s *TTSRouterSuite) sessionOver(provider *stubTTS, config routing.ProviderConfig) (*Session, *stubTTS) {
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
	RegisterMock(registry)

	for _, provider := range router.Config().Providers {
		s.Truef(registry.Has(provider.Provider),
			"%s is configured but has no factory, so it can never serve a request", provider.Provider)
	}
}

func (s *TTSRouterSuite) TestAVendorThatRecordsStillSpeaksLive() {
	// Probed on da97ee28: served by cartesia/sonic-preview.
	var built []routing.Spec
	session, err := s.newStubbedRouter(&built).Start(s.ctx, Request{CustomerID: "acme", Target: "cartesia/sonic-preview"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("cartesia/sonic-preview", session.Provider()+"/"+session.Model())
}

func (s *TTSRouterSuite) TestALiveCartesiaVoiceAskedForAFormatIsRefusedRatherThanStreamingRawAudio() {
	// On 41cb93cba both were refused before a voice was built, sonic-preview declaring no
	// format: "routing: cartesia/sonic-preview does not take format" and "routing: no
	// provider can express format".
	s.T().Setenv("CARTESIA_API_KEY", "test-key")
	for _, target := range []string{"cartesia/sonic-preview", "tts-fast"} {
		s.Run(target, func() {
			_, err := s.newRouter().Start(s.ctx, Request{
				CustomerID: "acme",
				Target:     target,
				Options:    options.TTS{Format: "pcm_16000"},
			})

			s.Require().ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
			s.EqualError(err, "cartesia/sonic-preview: a format is taken only when recording, not by a live voice")
		})
	}
}

func (s *TTSRouterSuite) TestALiveInworldVoiceAskedForAFormatIsRefusedRatherThanStreamingRawAudio() {
	// On b5a81650b both were refused before a voice was built, neither declaring format:
	// "routing: inworld/inworld-tts-2 does not take format".
	s.T().Setenv("INWORLD_API_KEY", "test-key")
	for _, target := range []string{"inworld/inworld-tts-2", "inworld/inworld-tts-2-flash"} {
		s.Run(target, func() {
			_, err := s.newRouter().Start(s.ctx, Request{
				CustomerID: "acme",
				Target:     target,
				Options:    options.TTS{Format: "pcm_16000"},
			})

			s.Require().ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
			s.EqualError(err, target+": a format is taken only when recording, not by a live voice")
		})
	}
}

func (s *TTSRouterSuite) TestALiveShortcutAskedForAFormatIsStillRefusedByCartesia() {
	// Probed on b5a81650b: both refused in these words.
	s.T().Setenv("CARTESIA_API_KEY", "test-key")
	s.T().Setenv("INWORLD_API_KEY", "test-key")
	for _, target := range []string{"tts-fast", "tts-quality"} {
		s.Run(target, func() {
			_, err := s.newRouter().Start(s.ctx, Request{CustomerID: "acme", Target: target, Options: options.TTS{Format: "pcm_16000"}})

			s.EqualError(err, "cartesia/sonic-preview: a format is taken only when recording, not by a live voice")
		})
	}
}

func (s *TTSRouterSuite) TestTheMockIsBuiltOnlyWhereItWasAskedFor() {
	s.False(DefaultRegistry().Has("mock"),
		"a deployment that did not ask for mocks should fail to build one rather than route to it")
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

// sarahOnly is an ElevenLabs library that lists Sarah and has no other voice, and counts
// the lookups it was asked for. With err set, a lookup fails rather than answering.
type sarahOnly struct {
	lookups *int
	err     error
}

func (sarahOnly) List(context.Context) ([]voices.Library, error) {
	return []voices.Library{{ID: "EXAVITQu4vr4xnSDxMaL", Name: "Sarah - Mature, Reassuring, Confident"}}, nil
}

func (sarahOnly) Preview(context.Context, string) (voices.Speech, error) {
	return voices.Speech{}, nil
}

func (l sarahOnly) Exists(context.Context, string) (bool, error) {
	if l.lookups != nil {
		*l.lookups++
	}
	return false, l.err
}

// Shaped is ElevenLabs' own rule.
func (sarahOnly) Shaped(id string) bool { return (*voices.ElevenLabs)(nil).Shaped(id) }

// listed is a library that lists its voices and cannot look any other up, as Cartesia's and
// Inworld's cannot.
type listed []voices.Library

func (l listed) List(context.Context) ([]voices.Library, error) { return l, nil }

func (listed) Preview(context.Context, string) (voices.Speech, error) {
	return voices.Speech{}, nil
}

// vendorVoices resolves voices as cmd/router does with every vendor's key: ElevenLabs'
// library looks a voice up, Cartesia's and Inworld's only list theirs.
func vendorVoices(elevenlabs sarahOnly, cartesia, inworld voices.Lister) routing.VoiceResolver {
	catalogue := voices.NewCatalogue()
	catalogue.Register("elevenlabs", elevenlabs)
	catalogue.Register("cartesia", cartesia)
	catalogue.Register("inworld", inworld)
	return voices.NewResolver(nil, catalogue)
}

// unknownVoice is the id WWSD asked staging for on 2026-10-16.
const unknownVoice = "ZZZZnotARealVoice123"

// cartesiaVoice is a voice Cartesia lists.
const cartesiaVoice = "a0e99841-438c-4a64-b679-ae501e7d6091"

// stockVoices is a Cartesia and an Inworld library that hold neither unknownVoice nor Sara.
var stockVoices = [2]listed{
	{{ID: cartesiaVoice, Name: "Barbershop Man"}},
	{{ID: "Ashley", Name: "Ashley"}},
}

func (s *TTSRouterSuite) TestAShortcutDoesNotFallBackToAVendorThatCannotKnowAVoiceElevenLabsRefused() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built, vendorVoices(sarahOnly{}, stockVoices[0], stockVoices[1]))

	_, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "tts-quality", Voice: unknownVoice})

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.ErrorContains(err, `elevenlabs/eleven_v4: voices: elevenlabs has no voice "ZZZZnotARealVoice123"; `+
		`the closest in its library is "Sarah - Mature, Reassuring, Confident" (EXAVITQu4vr4xnSDxMaL)`)
	s.NotContains(err.Error(), "every candidate")
	s.Empty(built, "Cartesia and Inworld would start and fail on the first line")
}

func (s *TTSRouterSuite) TestAVoiceElevenLabsRefusedIsRefusedWhenAnotherVendorIsTriedFirst() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built, vendorVoices(sarahOnly{}, stockVoices[0], stockVoices[1]))

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      unknownVoice,
		Options:    options.TTS{Providers: []string{"cartesia", "inworld", "elevenlabs"}},
	})

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.ErrorContains(err, `elevenlabs has no voice "ZZZZnotARealVoice123"`)
	s.Empty(built)
}

func (s *TTSRouterSuite) TestAVoiceElevenLabsRefusedIsSpokenByAVendorWhoseLibraryListsIt() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built,
		vendorVoices(sarahOnly{}, stockVoices[0], listed{{ID: unknownVoice, Name: unknownVoice}}))

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "tts-quality", Voice: unknownVoice})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("inworld", session.Provider())
	s.Require().Len(built, 1, "nothing was built for ElevenLabs or Cartesia")
	s.Equal(unknownVoice, built[0].Voice)
}

func (s *TTSRouterSuite) TestACartesiaVoiceOnAShortcutIsNeverLookedUpAtElevenLabs() {
	for _, voice := range []string{cartesiaVoice, "5619d38c-cf51-4d8e-9575-48f61a280413"} {
		for _, target := range []string{"tts-quality", "tts-fast"} {
			var built []routing.Spec
			lookups := 0
			router := s.newStubbedRouterResolving(&built,
				vendorVoices(sarahOnly{lookups: &lookups}, stockVoices[0], stockVoices[1]))

			session, err := router.Start(s.ctx, Request{
				CustomerID: "acme", Target: target, Voice: voice,
				Options: options.TTS{Providers: []string{"cartesia", "elevenlabs"}},
			})
			s.Require().NoError(err, "%s on %s, listed or not", voice, target)
			s.T().Cleanup(func() { _ = session.Close() })

			s.Equal("cartesia", session.Provider(), voice)
			s.Equal(0, lookups, "%s on %s: no ElevenLabs lookup before Cartesia speaks", voice, target)
		}
	}
}

func (s *TTSRouterSuite) TestANameNoListingCanBeReadForIsNotLookedUpOrRefused() {
	var built []routing.Spec
	lookups := 0
	router := s.newStubbedRouterResolving(&built,
		vendorVoices(sarahOnly{lookups: &lookups}, stockVoices[0], failingListing{}))

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme", Target: "tts-quality", Voice: "Ashley",
		Options: options.TTS{Providers: []string{"inworld", "elevenlabs"}},
	})
	s.Require().NoError(err, "Inworld's key may not list voices, and Ashley is one of them")
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("inworld", session.Provider())
	s.Equal(0, lookups)
}

func (s *TTSRouterSuite) TestAVoiceElevenLabsRefusedIsStillRefusedAfterACartesiaVoice() {
	var built []routing.Spec
	lookups := 0
	router := s.newStubbedRouterResolving(&built,
		vendorVoices(sarahOnly{lookups: &lookups}, stockVoices[0], stockVoices[1]))
	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "tts-fast", Voice: cartesiaVoice})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })
	s.Require().Equal(0, lookups, "ElevenLabs answers a Cartesia id with an error, so it is never asked")

	_, err = router.Start(s.ctx, Request{CustomerID: "acme", Target: "tts-quality", Voice: unknownVoice})

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.ErrorContains(err, `elevenlabs has no voice "ZZZZnotARealVoice123"; the closest in its library is`)
	s.Len(built, 1, "only the Cartesia voice was built")
}

// failingListing is a library that cannot be listed, as Inworld's answered staging's key
// on 2026-10-16: 403, "api key does not have required scopes".
type failingListing struct{ listed }

func (failingListing) List(context.Context) ([]voices.Library, error) {
	return nil, errors.New("the inworld key cannot list voices (http 403): api key does not have required scopes")
}

func (s *TTSRouterSuite) TestAShortcutSpeaksAVoiceElevenLabsHasOnElevenLabs() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built, vendorVoices(sarahOnly{}, stockVoices[0], stockVoices[1]))

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "tts-quality", Voice: "EXAVITQu4vr4xnSDxMaL"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs/eleven_v4", session.Provider()+"/"+session.Model())
}

func (s *TTSRouterSuite) TestAVoiceNoCandidateCanLookUpGoesWhereItAlwaysWent() {
	var built []routing.Spec
	lookups := 0
	router := s.newStubbedRouterResolving(&built,
		vendorVoices(sarahOnly{lookups: &lookups}, stockVoices[0], stockVoices[1]))

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      unknownVoice,
		Options:    options.TTS{Providers: []string{"cartesia", "inworld"}},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("cartesia", session.Provider())
	s.Equal(0, lookups)
}

func (s *TTSRouterSuite) TestAShortcutGoesWhereItAlwaysWentWhenElevenLabsCannotBeAsked() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built,
		vendorVoices(sarahOnly{err: errors.New("gateway")}, stockVoices[0], stockVoices[1]))

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "tts-quality", Voice: unknownVoice})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs/eleven_v4", session.Provider()+"/"+session.Model())
}

func (s *TTSRouterSuite) TestAShortcutSpeaksAnElevenLabsOverwriteWhateverTheVoice() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built, vendorVoices(sarahOnly{}, stockVoices[0], stockVoices[1]))

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Voice:      unknownVoice,
		Options: options.TTS{
			Overwrites: map[string]json.RawMessage{"elevenlabs": json.RawMessage(`{"voice_id":"EXAVITQu4vr4xnSDxMaL"}`)},
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs/eleven_v4", session.Provider()+"/"+session.Model())
}

// checkingVoices resolves voices against ElevenLabs' library alone, so another vendor's
// voices go through unchecked.
func checkingVoices(lookups *int) routing.VoiceResolver {
	catalogue := voices.NewCatalogue()
	catalogue.Register("elevenlabs", sarahOnly{lookups: lookups})
	return voices.NewResolver(nil, catalogue)
}

func (s *TTSRouterSuite) TestAVoiceOneVendorDoesNotHaveIsSpokenByTheNextInThePriorityList() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built, vendorVoices(sarahOnly{}, listed{{ID: "Sara"}}, listed{}))

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      "Sara",
		Options:    options.TTS{Providers: []string{"elevenlabs", "cartesia"}},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("cartesia", session.Provider(), "a voice is the caller's, so another vendor may still have it")
	s.Len(built, 1, "nothing was built for the vendor that said it has no such voice")
}

func (s *TTSRouterSuite) TestAVoiceNoVendorHasIsLookedUpOnceAndRefusedOnceAcrossItsModels() {
	var built []routing.Spec
	lookups := 0
	router := s.newStubbedRouterResolving(&built, checkingVoices(&lookups))

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      "Sara",
		Options:    options.TTS{Providers: []string{"elevenlabs"}},
	})

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.Equal(1, lookups, "every ElevenLabs model asks about the same voice")
	s.Equal(1, strings.Count(err.Error(), `has no voice "Sara"`), "the caller reads the refusal once: %s", err)
}

func (s *TTSRouterSuite) TestAVoiceAnOverwriteReplacesIsNotCheckedAndTheOverwriteSpeaks() {
	var built []routing.Spec
	lookups := 0
	router := s.newStubbedRouterResolving(&built, checkingVoices(&lookups))

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      "Sara",
		Options: options.TTS{
			Providers:  []string{"elevenlabs"},
			Overwrites: map[string]json.RawMessage{"elevenlabs": json.RawMessage(`{"voice_id":"EXAVITQu4vr4xnSDxMaL"}`)},
		},
	})
	s.Require().NoError(err, "the voice that is spoken is the overwrite's, and it was never asked about")
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal(0, lookups)
	s.Require().NotEmpty(built)
	s.JSONEq(`{"voice_id":"EXAVITQu4vr4xnSDxMaL"}`, string(built[0].Overwrites))
}

func (s *TTSRouterSuite) TestAVoiceNoVendorHasIsRefusedNamingTheClosest() {
	var built []routing.Spec
	router := s.newStubbedRouterResolving(&built, checkingVoices(nil))

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      "Sara",
		Options:    options.TTS{Providers: []string{"elevenlabs"}},
	})

	s.Require().ErrorIs(err, routing.ErrRequest, "the caller's mistake, not every candidate failing")
	s.Contains(err.Error(), `"Sarah - Mature, Reassuring, Confident" (EXAVITQu4vr4xnSDxMaL)`)
	s.Empty(built)
}

func (s *TTSRouterSuite) TestAVendorNamedForALiveCallGetsTheirStreamingModel() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	// ElevenLabs has six models here, and eleven_v3 is the one that returns a file
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

func (s *TTSRouterSuite) TestASpeedIsOnlyAskedOfAVoiceThatCanChangeIt() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	speed := 0.9

	// The dialogue models are listed first, and none of them can be sped up.
	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Options: options.TTS{
			Providers: []string{"elevenlabs/eleven_v3_conversational", "elevenlabs/eleven_flash_v2_5"},
			Speed:     &speed,
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("eleven_flash_v2_5", session.Model())
	s.Require().Len(built, 1)
	s.Equal(&speed, built[0].TTS.Speed)
}

func (s *TTSRouterSuite) TestASpeedNoVoiceCanChangeIsRefused() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	speed := 0.9

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Options: options.TTS{
			Providers: []string{"elevenlabs/eleven_v3_conversational"},
			Speed:     &speed,
		},
	})

	s.Error(err, "a voice that ignores the speed asked of it is worse than one that says it cannot")
	s.Empty(built)
}

func (s *TTSRouterSuite) TestRegistryRefusesASpeedOutsideTheVendorsRange() {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	speed := 1.5

	_, err := DefaultRegistry().Build("elevenlabs", routing.Spec{
		Model: "eleven_flash_v2_5",
		TTS:   options.TTS{Speed: &speed},
	})
	s.ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
	s.ErrorContains(err, "outside 0.7 to 1.2")
}

func (s *TTSRouterSuite) TestASpeedElevenLabsWillNotTakeIsRefusedAsTheCallersMistake() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)
	// ElevenLabs takes 0.7 to 1.2
	// (https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input,
	// opened 2026-10-16).
	outside := 1.5

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "en-low-latency",
		Options:    options.TTS{Speed: &outside},
	})

	s.Require().ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
	s.ErrorContains(err, "speed 1.5 is outside 0.7 to 1.2")
	s.NotContains(err.Error(), "every candidate", "the caller's mistake is not an outage")
	s.Equal([]string{"elevenlabs/eleven_flash_v2_5"}, built, "v4 Turbo takes no speed, so only Flash is built")
}

func (s *TTSRouterSuite) TestASpeedADialogueModelIsHandedIsRefusedAsTheCallersMistake() {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	speed := 0.9

	// Routing never hands a dialogue model a speed, since none declares one; the factory
	// refuses it all the same rather than let the socket drop it.
	_, err := DefaultRegistry().Build("elevenlabs", routing.Spec{Model: "eleven_v4", TTS: options.TTS{Speed: &speed}})

	s.ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
	s.ErrorContains(err, "eleven_v4 takes no speed")
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

func (s *TTSRouterSuite) TestAnOverwriteElevenLabsHasNoFieldForIsRefusedRatherThanSpokenByAnotherVoice() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options: options.TTS{Overwrites: map[string]json.RawMessage{
			"elevenlabs": json.RawMessage(`{"voice_settings":{"speed":1.2}}`),
		}},
	})

	s.Require().ErrorIs(err, routing.ErrOverwrite)
	s.ErrorContains(err, "elevenlabs/eleven_v4")
	// voice_settings has a style and nothing else: a speed is a shared option, so naming one
	// here is a field ElevenLabs' block does not have.
	s.ErrorContains(err, `unknown field "speed"`)
	s.Equal([]string{"elevenlabs/eleven_v4", "elevenlabs/eleven_v4_turbo"}, built,
		"cartesia never reads the elevenlabs block, so serving with it would drop the overwrite in silence")
}

func (s *TTSRouterSuite) TestAVoiceIdOverwriteStillReachesElevenLabsOnTheQualityShortcut() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options: options.TTS{Overwrites: map[string]json.RawMessage{
			"elevenlabs": json.RawMessage(`{"voice_id":"el-1"}`),
		}},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs", session.Provider())
	s.Equal("eleven_v4", session.Model())
}

func (s *TTSRouterSuite) TestAStabilityOnTheQualityShortcutIsServedByElevenLabs() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	stability := 0.4

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options:    options.TTS{Stability: &stability},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs", session.Provider())
	s.Equal("eleven_v4", session.Model())
	s.Require().Len(built, 1)
	s.Equal(&stability, built[0].TTS.Stability)
}

func (s *TTSRouterSuite) TestASimilarityOnTheQualityShortcutIsRefusedAtRouting() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)
	similarity := 0.7

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options:    options.TTS{Similarity: &similarity},
	})

	// Refused at routing, as on the commit before this PR (which named the term
	// stability), since the dialogue socket has no similarity and nothing else on the
	// shortcut declares one.
	s.ErrorContains(err, "routing: no provider can express similarity")
	s.Empty(built, "nothing is built, so nothing is counted against a model")
}

func (s *TTSRouterSuite) TestASimilarityOnAModelThatTakesNoneIsRefusedNamingTheModel() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)
	similarity := 0.7

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "elevenlabs/eleven_v4",
		Options:    options.TTS{Similarity: &similarity},
	})

	s.ErrorContains(err, "routing: elevenlabs/eleven_v4 does not take similarity")
	s.Empty(built)
}

func (s *TTSRouterSuite) TestASimilarityOnTheDefaultShortcutSkipsTheDialogueModel() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)
	similarity := 0.7

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "en-low-latency",
		Options:    options.TTS{Similarity: &similarity},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("eleven_flash_v2_5", session.Model())
	s.Equal([]string{"elevenlabs/eleven_flash_v2_5"}, built, "v4 Turbo takes no similarity, so it is never built")
}

func (s *TTSRouterSuite) TestAStabilityElevenLabsWillNotTakeIsRefusedAsTheCallersMistake() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)
	outside := 1.5

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options:    options.TTS{Stability: &outside},
	})

	s.Require().ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
	s.ErrorContains(err, "stability 1.5 is outside 0 to 1")
	s.NotContains(err.Error(), "every candidate", "the caller's mistake is not an outage")
}

func (s *TTSRouterSuite) TestRegistryHandsTheStabilityAndSimilarityToElevenLabs() {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	outside := 1.5

	// Out of range is the one answer the provider gives that shows it was handed the value.
	_, err := DefaultRegistry().Build("elevenlabs", routing.Spec{Model: "eleven_v4", TTS: options.TTS{Stability: &outside}})
	s.ErrorContains(err, "stability 1.5 is outside 0 to 1")
	_, err = DefaultRegistry().Build("elevenlabs", routing.Spec{Model: "eleven_flash_v2_5", TTS: options.TTS{Similarity: &outside}})
	s.ErrorContains(err, "similarity 1.5 is outside 0 to 1")
}

func (s *TTSRouterSuite) TestRegistryHandsTheStyleOverwriteToElevenLabs() {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")

	// Out of range is the one answer the provider gives that shows it was handed the value.
	_, err := DefaultRegistry().Build("elevenlabs", routing.Spec{
		Model:      "eleven_flash_v2_5",
		Overwrites: json.RawMessage(`{"voice_settings":{"style":1.5}}`),
	})

	s.ErrorIs(err, routing.ErrOverwrite, "the style came in the overwrite block, so the block is what was refused")
	s.ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
	s.ErrorContains(err, "style 1.5 is outside 0 to 1")
}

func (s *TTSRouterSuite) TestAStyleOverwriteIsSpokenByAModelThatTakesOne() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "en-low-latency",
		Options: options.TTS{Overwrites: map[string]json.RawMessage{
			"elevenlabs": json.RawMessage(`{"voice_settings":{"style":0.4}}`),
		}},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("eleven_flash_v2_5", session.Model())
	s.Equal([]string{"elevenlabs/eleven_v4_turbo", "elevenlabs/eleven_flash_v2_5"}, built,
		"v4 Turbo has no style, so the request stays with ElevenLabs and Flash speaks it")
}

func (s *TTSRouterSuite) TestAStyleOverwriteOnTheQualityShortcutIsRefusedRatherThanSpokenByAnotherVoice() {
	var built []string
	router := s.newRouterOverRealElevenLabs(&built)

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options: options.TTS{Overwrites: map[string]json.RawMessage{
			"elevenlabs": json.RawMessage(`{"voice_settings":{"style":0.4}}`),
		}},
	})

	s.Require().ErrorIs(err, routing.ErrOverwrite)
	s.ErrorContains(err, "style is not taken by eleven_v4")
	s.Equal([]string{"elevenlabs/eleven_v4", "elevenlabs/eleven_v4_turbo"}, built,
		"cartesia never reads the elevenlabs block, so serving with it would drop the style in silence")
}

// onlyCartesiasVoice asks a voice of Cartesia alone, on a request that names no other.
var onlyCartesiasVoice = map[string]json.RawMessage{"cartesia": json.RawMessage(`{"voice_id":"` + cartesiaVoice + `"}`)}

// newStubbedRouterWithDownVendor is newStubbedRouter where one vendor fails to start.
func (s *TTSRouterSuite) newStubbedRouterWithDownVendor(built *[]routing.Spec, down string) *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	registry := routing.NewRegistry[tts.TTS]()
	for _, provider := range config[routing.TTS].Providers {
		registry.Register(provider.Provider, func(spec routing.Spec) (tts.TTS, error) {
			*built = append(*built, spec)
			if provider.Provider == down {
				return nil, errors.New(down + " is down")
			}
			return newStubTTS(), nil
		})
	}

	router, err := New(Options{Config: config[routing.TTS], Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *TTSRouterSuite) TestAVoiceAskedOnlyOfOneVendorIsSpokenByThatVendorInThatVoice() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	// On b1e563f98 ElevenLabs v4 served this with no voice, so in its default.
	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options:    options.TTS{Overwrites: onlyCartesiasVoice},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("cartesia", session.Provider())
	s.Require().Len(built, 1)
	s.Equal(cartesiaVoice, built[0].Voice, "Cartesia's factory reads no overwrites, so the voice comes as the spec's")
}

func (s *TTSRouterSuite) TestAVoiceAskedOfAVendorThatFailsIsNotSpokenByAnotherInItsDefault() {
	var built []routing.Spec
	router := s.newStubbedRouterWithDownVendor(&built, "cartesia")

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Options:    options.TTS{Overwrites: onlyCartesiasVoice},
	})

	s.Require().Error(err)
	s.ErrorContains(err, "cartesia/sonic-preview: cartesia is down")
	s.Require().Len(built, 1, "no other vendor was started without the voice")
	s.Equal("sonic-preview", built[0].Model)
}

func (s *TTSRouterSuite) TestAVoiceAskedOfAVendorTheTargetDoesNotReachIsRefused() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Options:    options.TTS{Providers: []string{"elevenlabs"}, Overwrites: onlyCartesiasVoice},
	})

	s.Require().ErrorIs(err, routing.ErrRequest, "the caller can fix it, and no model's health pays for it")
	s.ErrorContains(err, "a voice was asked for only in overwrites.cartesia, and no candidate for this request is cartesia")
	s.Empty(built)
}

func (s *TTSRouterSuite) TestWithNoVoiceAskedTheQualityShortcutSpeaksInElevenLabsDefaultAsBefore() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	// Probed on b1e563f98: elevenlabs/eleven_v4, built with no voice.
	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "tts-quality"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs/eleven_v4", session.Provider()+"/"+session.Model())
	s.Require().Len(built, 1)
	s.Empty(built[0].Voice)
}

func (s *TTSRouterSuite) TestATopLevelVoiceIsStillTheOneCartesiaIsBuiltWithBesideItsOwnVoiceId() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	// Probed on b1e563f98: built with top-voice, which Cartesia's factory reads from the spec.
	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Voice:      "top-voice",
		Options:    options.TTS{Providers: []string{"cartesia"}, Overwrites: onlyCartesiasVoice},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("cartesia", session.Provider())
	s.Require().Len(built, 1)
	s.Equal("top-voice", built[0].Voice, "a top-level voice is not replaced by an overwrite, as on base")
}

func (s *TTSRouterSuite) TestAVoiceAtTheTopLevelStillGoesToEveryVendorBesideAnOverwrite() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	// Probed on b1e563f98: elevenlabs/eleven_v4, built with library-voice.
	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "tts-quality",
		Voice:      "library-voice",
		Options:    options.TTS{Overwrites: onlyCartesiasVoice},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("elevenlabs/eleven_v4", session.Provider()+"/"+session.Model())
	s.Require().Len(built, 1)
	s.Equal("library-voice", built[0].Voice)
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

func (s *TTSRouterSuite) TestAVoiceThatTakesOneSentenceAtATimeIsHeardInOrder() {
	provider := newStubTTS()
	provider.perSentence = true
	session, _ := s.sessionOver(provider, routing.ProviderConfig{Provider: "stub", Model: "stub-model"})
	s.Require().NoError(session.Synthesize(tts.Request{ID: "first", Text: "Take your time.", Final: true}))
	s.Require().NoError(session.Synthesize(tts.Request{ID: "second", Text: "I'm right here.", Final: true}))

	pcm := audio.PcmData{Samples: make([]int16, 2400), SampleRate: 24_000, Channels: 1}
	provider.emitter.Send(tts.AudioChunk{SynthesisID: "second", Index: 0, Audio: pcm})
	provider.emitter.Send(tts.AudioChunk{SynthesisID: "first", Index: 0, Audio: pcm})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "second"})
	provider.emitter.Send(tts.AudioChunk{SynthesisID: "first", Index: 1, Audio: pcm})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "first"})
	s.Require().NoError(session.Close())

	var heard []string
	for _, event := range s.drain(session) {
		switch typed := event.(type) {
		case tts.AudioChunk:
			heard = append(heard, typed.SynthesisID)
		case tts.SynthesisComplete:
			heard = append(heard, typed.SynthesisID+" done")
		}
	}
	s.Equal([]string{"first", "first", "first done", "second", "second done"}, heard)
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
