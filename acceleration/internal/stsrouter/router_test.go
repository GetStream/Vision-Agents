package stsrouter

import (
	"context"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

// stubSTS stands in for a real provider so a session can be driven without credentials.
type stubSTS struct {
	emitter      *sts.Emitter
	capabilities sts.Capabilities
	heard        []sts.PcmData
	typed        []string
	answers      []string
	interrupts   []int
	closed       bool
}

func newStubSTS() *stubSTS {
	return &stubSTS{emitter: sts.NewEmitter(sts.EmitterBuffer), capabilities: sts.Capabilities{Text: true, Tools: true}}
}

func (s *stubSTS) Start(context.Context) error { return nil }

func (s *stubSTS) ProcessAudio(pcm sts.PcmData, _ sts.Participant) error {
	s.heard = append(s.heard, pcm)
	return nil
}

func (s *stubSTS) SendText(text string, _ sts.Participant) error {
	s.typed = append(s.typed, text)
	return nil
}

func (s *stubSTS) SendFrame(llm.ImagePart) error { return sts.ErrNoImages }
func (s *stubSTS) SetInstructions(string) error  { return nil }
func (s *stubSTS) SetTools([]llm.Tool) error     { return nil }
func (s *stubSTS) Prompt(text string) error      { s.typed = append(s.typed, text); return nil }
func (s *stubSTS) Answer(callID, output string, _ error) error {
	s.answers = append(s.answers, callID+"="+output)
	return nil
}
func (s *stubSTS) Interrupt(playedMs int) error {
	s.interrupts = append(s.interrupts, playedMs)
	return nil
}
func (s *stubSTS) Events() <-chan sts.Event { return s.emitter.Events() }

func (s *stubSTS) Close() error {
	s.closed = true
	s.emitter.Close()
	return nil
}

func (s *stubSTS) Provider() string               { return "stub" }
func (s *stubSTS) Model() string                  { return "stub-model" }
func (s *stubSTS) SampleRate() int                { return 24_000 }
func (s *stubSTS) Capabilities() sts.Capabilities { return s.capabilities }

type STSRouterSuite struct {
	suite.Suite
	ctx context.Context
}

func TestSTSRouterSuite(t *testing.T) {
	suite.Run(t, new(STSRouterSuite))
}

func (s *STSRouterSuite) SetupTest() {
	s.ctx = context.Background()
}

// newRouter routes over the built-in speech-to-speech config, which is what a deployment
// gets when it sets no config file.
func (s *STSRouterSuite) newRouter() *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	router, err := New(Options{Config: config[routing.STS], Registry: DefaultRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// newStubbedRouter routes over the built-in config with every real model replaced by a
// stub, so where a request lands can be asked of the deployment's own config without a key
// for any of the vendors in it. Every spec built is recorded, which is how what a provider
// was asked for is checked.
func (s *STSRouterSuite) newStubbedRouter(built *[]routing.Spec) *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	registry := NewRegistry()
	for _, provider := range config[routing.STS].Providers {
		capabilities, _ := capabilitiesFor(provider.Provider, provider.Model)
		registry.Register(provider.Provider, func(spec routing.Spec) (sts.STS, error) {
			*built = append(*built, spec)
			stub := newStubSTS()
			stub.capabilities = capabilities
			return stub, nil
		})
	}

	router, err := New(Options{Config: config[routing.STS], Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// newSession returns a session over a stub provider, so event handling is the only thing
// under test.
func (s *STSRouterSuite) newSession() (*Session, *stubSTS) {
	provider := newStubSTS()
	recorder := routing.NewRecorder(routing.STS, nil, nil, slog.Default())
	config := routing.ProviderConfig{Provider: "stub", Model: "stub-model"}
	session := newSession(provider, config, routing.Owner{CustomerID: "acme"}, recorder)

	s.T().Cleanup(func() {
		_ = session.Close()
		recorder.Close()
	})
	return session, provider
}

// drain reads the session's forwarded events until the channel closes.
func (s *STSRouterSuite) drain(session *Session) []sts.Event {
	var events []sts.Event
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

func chunk(generation, index int) sts.AudioChunk {
	return sts.AudioChunk{
		ResponseID: "r",
		Generation: generation,
		Index:      index,
		Audio:      audio.PcmData{Samples: make([]int16, 240), SampleRate: 24_000, Channels: 1},
	}
}

func chunks(events []sts.Event) []sts.AudioChunk {
	var found []sts.AudioChunk
	for _, event := range events {
		if typed, ok := event.(sts.AudioChunk); ok {
			found = append(found, typed)
		}
	}
	return found
}

func (s *STSRouterSuite) TestRouterServesTheSpeechToSpeechModality() {
	s.Equal(routing.STS, s.newRouter().Modality())
}

func (s *STSRouterSuite) TestEveryShortcutResolvesToAProvider() {
	router := s.newRouter()

	for alias := range router.Config().Aliases {
		candidates, err := router.Resolve(s.ctx, alias, nil)
		s.Require().NoErrorf(err, "alias %s", alias)
		s.NotEmptyf(candidates, "alias %s", alias)
	}
}

func (s *STSRouterSuite) TestRegistryKnowsEveryConfiguredProvider() {
	router := s.newRouter()
	registry := DefaultRegistry()

	for _, provider := range router.Config().Providers {
		s.Truef(registry.Has(provider.Provider),
			"%s is configured but has no factory, so it can never serve a request", provider.Provider)
	}
}

func (s *STSRouterSuite) TestAConfigPromisingWhatItsProviderCannotSendIsRefusedAtBoot() {
	config := routing.ModalityConfig{Providers: []routing.ProviderConfig{{
		Provider:   "qwen",
		Model:      "qwen3.5-omni-plus-realtime",
		Languages:  []string{"en"},
		Realtime:   true,
		Terms:      []options.Term{options.Tools},
		DataPolicy: options.DataHandling{TrainsOnData: options.ClaimUnknown, Retention: options.RetentionUnknown},
	}}}

	_, err := New(Options{Config: config, Registry: DefaultRegistry()})
	s.ErrorContains(err, "declares tools, which its provider cannot express",
		"a term declared and never sent is the one thing terms exist to prevent")
}

func (s *STSRouterSuite) TestSemanticTurnsRouteOnlyToAModelThatReadsTheWords() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "sts-fast",
		Options:    options.STS{TurnDetection: options.TurnSemantic},
	})
	s.ErrorContains(err, "semantic_turns",
		"a caller who asked for semantic turns and got a silence timer could not hear the difference")

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "openai/gpt-realtime-2",
		Options:    options.STS{TurnDetection: options.TurnSemantic},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })
	s.Equal("openai", session.Provider())
	s.Equal(options.TurnSemantic, built[0].STS.TurnDetection)
}

func (s *STSRouterSuite) TestFramesRouteOnlyToAModelThatSees() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	sees := true

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "xai/grok-voice-think-fast-2.0",
		Options:    options.STS{Images: &sees},
	})
	s.ErrorContains(err, "no provider accepts image input")

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "sts-vision",
		Options:    options.STS{Images: &sees},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })
	s.True(session.Capabilities().Accepts(options.ModalityImage))
}

func (s *STSRouterSuite) TestOverwritesReachOnlyTheVendorTheyName() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	overwrites := map[string]json.RawMessage{
		"openai": json.RawMessage(`{"eagerness":"high"}`),
		"gemini": json.RawMessage(`{"thinking_level":"low"}`),
	}

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "openai/gpt-realtime-2",
		Options:    options.STS{Overwrites: overwrites},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Require().Len(built, 1)
	s.JSONEq(`{"eagerness":"high"}`, string(built[0].Overwrites),
		"the model should be handed its own block and nobody else's")
}

func (s *STSRouterSuite) TestToolsInstructionsAndVoiceReachTheFactory() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)
	tools := []llm.Tool{{Name: "get_weather", Description: "Weather in a city."}}

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "gemini/gemini-3.1-flash-live-preview",
		Tools:      tools,
		Options:    options.STS{Instructions: "Be brief.", Voice: "Kore"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Require().Len(built, 1)
	s.Equal(tools, built[0].Tools, "a model that takes tools only at setup has to get them at start")
	s.Equal("Be brief.", built[0].STS.Instructions)
	s.Equal("Kore", built[0].Voice)
}

func (s *STSRouterSuite) TestACustomVoiceIsRefusedUpFront() {
	var built []routing.Spec
	router := s.newStubbedRouter(&built)

	_, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "sts-fast",
		Options:    options.STS{Voice: options.OwnVoicePrefix + "receptionist"},
	})
	s.ErrorContains(err, "no voices of its own",
		"none of these models takes a cloned voice, so the honest answer is a refusal, not a lookup at every candidate")
	s.Empty(built, "nothing should have been built for a request that could never be served")
}

func (s *STSRouterSuite) TestAudioFromAnInterruptedReplyIsDroppedOnceItSettles() {
	session, provider := s.newSession()

	provider.emitter.Send(sts.ResponseStarted{ResponseID: "r", Generation: 1, At: time.Now()})
	provider.emitter.Send(chunk(1, 0))
	provider.emitter.Send(sts.ResponseComplete{ResponseID: "r", Generation: 1, Interrupted: true})
	// The model learns of the barge-in a round trip late, so this chunk is the tail the
	// caller talked over.
	provider.emitter.Send(chunk(1, 1))
	provider.emitter.Send(sts.ResponseStarted{ResponseID: "r2", Generation: 2, At: time.Now()})
	provider.emitter.Send(chunk(2, 0))
	s.Require().NoError(session.Close())

	forwarded := chunks(s.drain(session))
	s.Require().Len(forwarded, 2)
	s.Equal(1, forwarded[0].Generation)
	s.Equal(0, forwarded[0].Index)
	s.Equal(2, forwarded[1].Generation, "the next reply's audio must not be dropped with the last one's")
}

func (s *STSRouterSuite) TestInterruptMutesTheReplyBeforeTheProviderHearsOfIt() {
	session, provider := s.newSession()

	provider.emitter.Send(sts.ResponseStarted{ResponseID: "r", Generation: 1, At: time.Now()})
	provider.emitter.Send(chunk(1, 0))
	// Let the start and the chunk through before interrupting, so the interrupt is
	// unambiguously about a reply in flight.
	s.Eventually(func() bool {
		session.mu.Lock()
		defer session.mu.Unlock()
		return session.current == 1
	}, time.Second, 10*time.Millisecond)

	s.Require().NoError(session.Interrupt(350))
	provider.emitter.Send(chunk(1, 1))
	s.Require().NoError(session.Close())

	s.Equal([]int{350}, provider.interrupts, "the provider should be told how much the listener heard")
	forwarded := chunks(s.drain(session))
	s.Require().Len(forwarded, 1, "audio after the caller cut in must not reach them")
	s.Equal(0, forwarded[0].Index)
}

func (s *STSRouterSuite) TestTheCallersAudioIsBilledAgainstTheReplyThatAnsweredIt() {
	session, provider := s.newSession()
	spoken := audio.PcmData{Samples: make([]int16, 1600), SampleRate: 16_000, Channels: 1}

	s.Require().NoError(session.ProcessAudio(spoken, sts.Participant{ID: "alice"}))
	s.Require().NoError(session.ProcessAudio(spoken, sts.Participant{ID: "alice"}))
	s.Require().Len(provider.heard, 2)

	session.mu.Lock()
	heard := session.heardMs
	session.mu.Unlock()
	s.InDelta(200, heard, 0.001)

	provider.emitter.Send(sts.ResponseStarted{ResponseID: "r", Generation: 1, At: time.Now()})
	provider.emitter.Send(sts.ResponseComplete{ResponseID: "r", Generation: 1})
	s.Eventually(func() bool {
		session.mu.Lock()
		defer session.mu.Unlock()
		return session.heardMs == 0 && len(session.inFlight) == 0
	}, time.Second, 10*time.Millisecond, "the reply should have claimed the audio it answered and left flight")
}

func (s *STSRouterSuite) TestAFailureNamingAReplyIsSettledByItsCompletion() {
	session, provider := s.newSession()

	provider.emitter.Send(sts.ResponseStarted{ResponseID: "r", Generation: 1, At: time.Now()})
	provider.emitter.Send(sts.Error{ResponseID: "r", Err: context.DeadlineExceeded, Context: "response"})
	s.Eventually(func() bool {
		session.mu.Lock()
		defer session.mu.Unlock()
		inFlight, ok := session.inFlight["r"]
		return ok && inFlight.errorCode == "response"
	}, time.Second, 10*time.Millisecond, "the failure is remembered, not recorded, so the completion is what settles it")

	provider.emitter.Send(sts.ResponseComplete{ResponseID: "r", Generation: 1})
	s.Eventually(func() bool {
		session.mu.Lock()
		defer session.mu.Unlock()
		return len(session.inFlight) == 0
	}, time.Second, 10*time.Millisecond)
	s.Require().NoError(session.Close())

	var errors, completes int
	for _, event := range s.drain(session) {
		switch event.(type) {
		case sts.Error:
			errors++
		case sts.ResponseComplete:
			completes++
		}
	}
	s.Equal(1, errors, "the failure is still forwarded")
	s.Equal(1, completes)
}

func (s *STSRouterSuite) TestTheSessionForwardsWhatItIsAskedTo() {
	session, provider := s.newSession()

	s.Require().NoError(session.SendText("hello", sts.Participant{ID: "alice"}))
	s.Require().NoError(session.Prompt("greet the caller"))
	s.Require().NoError(session.Answer("call_1", `{"ok":true}`, nil))
	s.ErrorIs(session.SendFrame(llm.ImagePart{}), sts.ErrNoImages)

	s.Equal([]string{"hello", "greet the caller"}, provider.typed)
	s.Equal([]string{`call_1={"ok":true}`}, provider.answers)
	s.Equal(24_000, session.SampleRate())
	s.True(session.Capabilities().Tools)
}
