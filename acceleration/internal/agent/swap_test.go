package agent

import (
	"context"
	"errors"
	"log/slog"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// stubSTS is a speech-to-speech model that hears whatever it is sent and says nothing.
type stubSTS struct {
	emitter      *sts.Emitter
	instructions string

	mu     sync.Mutex
	heard  int
	closed bool
}

func newStubSTS(instructions string) *stubSTS {
	return &stubSTS{emitter: sts.NewEmitter(sts.EmitterBuffer), instructions: instructions}
}

func (s *stubSTS) Start(context.Context) error { return nil }

func (s *stubSTS) ProcessAudio(sts.PcmData, sts.Participant) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.heard++
	return nil
}

func (s *stubSTS) SendText(string, sts.Participant) error { return nil }
func (s *stubSTS) SendFrame(llm.ImagePart) error          { return sts.ErrNoImages }
func (s *stubSTS) SetInstructions(string) error           { return nil }
func (s *stubSTS) SetTools([]llm.Tool) error              { return nil }
func (s *stubSTS) Answer(string, string, error) error     { return nil }
func (s *stubSTS) Prompt(string) error                    { return nil }
func (s *stubSTS) Interrupt(int) error                    { return nil }
func (s *stubSTS) Events() <-chan sts.Event               { return s.emitter.Events() }

func (s *stubSTS) Close() error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.emitter.Close()
	return nil
}

func (s *stubSTS) Provider() string { return "openai" }
func (s *stubSTS) Model() string    { return "gpt-realtime-2" }
func (s *stubSTS) SampleRate() int  { return 24_000 }

func (s *stubSTS) Capabilities() sts.Capabilities {
	return sts.Capabilities{
		InputModalities: []string{"image"},
		Text:            true, Tools: true, InputTranscript: true, OutputTranscript: true,
		SemanticTurns: true, ManualTurns: true, Endpointing: true,
		InstructionsMidSession: true, ToolsMidSession: true, Usage: true,
	}
}

func (s *stubSTS) hearing() (int, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.heard, s.closed
}

// swappable is what a session that moves between models opened: every provider is a
// fresh stub per session, by provider name, so a test can tell the old ones from the new.
type swappable struct {
	mu     sync.Mutex
	opens  map[string]int
	models map[string][]*stubLLM
	voices map[string][]*stubTTS
	ears   []*stubSTT
	speech []*stubSTS

	// refuses is the error a provider's conversation models answer with, so a test can
	// offer a model that a session opens onto but that never replies.
	refuses map[string]error
}

// refuse makes every conversation model opened on a provider from now on answer with err.
func (w *swappable) refuse(provider string, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.refuses[provider] = err
}

func (w *swappable) model(provider string) *stubLLM {
	w.mu.Lock()
	defer w.mu.Unlock()
	opened := w.models[provider]
	if len(opened) == 0 {
		return nil
	}
	return opened[len(opened)-1]
}

func (w *swappable) voice(provider string) *stubTTS {
	w.mu.Lock()
	defer w.mu.Unlock()
	opened := w.voices[provider]
	if len(opened) == 0 {
		return nil
	}
	return opened[len(opened)-1]
}

func (w *swappable) listener() *stubSTT {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.ears) == 0 {
		return nil
	}
	return w.ears[len(w.ears)-1]
}

func (w *swappable) conversing() *stubSTS {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.speech) == 0 {
		return nil
	}
	return w.speech[len(w.speech)-1]
}

// swapConfig routes two providers, so a session can be moved from one to the other.
func swapConfig() routing.ModalityConfig {
	return routing.ModalityConfig{Providers: []routing.ProviderConfig{
		{Provider: "stub", Model: "stub-model", Languages: []string{"en"}, Realtime: true},
		{Provider: "other", Model: "other-model", Languages: []string{"en"}, Realtime: true},
	}}
}

// joinSwappable joins a cascade on the stub providers, with the other providers and a
// speech-to-speech model available to move it onto.
func (s *AgentSuite) joinSwappable() *swappable {
	w := &swappable{
		opens:   map[string]int{},
		models:  map[string][]*stubLLM{},
		voices:  map[string][]*stubTTS{},
		refuses: map[string]error{},
	}
	logger := slog.New(slog.DiscardHandler)

	// Each cascade opens its conversation model and then its flow controller, so the opens
	// alternate between the two.
	reasoning := llmrouter.NewRegistry()
	transcription := sttrouter.NewRegistry()
	speaking := ttsrouter.NewRegistry()
	for _, provider := range []string{"stub", "other"} {
		reasoning.Register(provider, func(routing.Spec) (llmrouter.Provider, error) {
			opened := newStubLLM()
			w.mu.Lock()
			defer w.mu.Unlock()
			if w.opens[provider]%2 == 0 {
				opened.reply = []string{"Hello there."}
				opened.refuses = w.refuses[provider]
				w.models[provider] = append(w.models[provider], opened)
			} else {
				opened.reply = []string{`{"disposition":"respond","floor":"stop"}`}
			}
			w.opens[provider]++
			return opened, nil
		})
		transcription.Register(provider, func(routing.Spec) (stt.STT, error) {
			opened := newStubSTT()
			w.mu.Lock()
			defer w.mu.Unlock()
			w.ears = append(w.ears, opened)
			return opened, nil
		})
		speaking.Register(provider, func(routing.Spec) (tts.TTS, error) {
			opened := newStubTTS(true)
			w.mu.Lock()
			defer w.mu.Unlock()
			w.voices[provider] = append(w.voices[provider], opened)
			return opened, nil
		})
	}
	reasoner, err := llmrouter.New(llmrouter.Options{Config: swapConfig(), Registry: reasoning, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)
	transcriber, err := sttrouter.New(sttrouter.Options{Config: swapConfig(), Registry: transcription, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: swapConfig(), Registry: speaking, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	models := stsrouter.NewRegistry()
	models.Register("openai", func(spec routing.Spec) (sts.STS, error) {
		opened := newStubSTS(spec.STS.Instructions)
		w.mu.Lock()
		defer w.mu.Unlock()
		w.speech = append(w.speech, opened)
		return opened, nil
	})
	conversing, err := stsrouter.New(stsrouter.Options{Config: config[routing.STS], Registry: models, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(conversing.Close)

	s.edge = newLoopbackEdge()
	s.agent, err = New(Options{
		Edge:         s.edge,
		Instructions: "be brief",
		CustomerID:   "acme",
		LLM:          reasoner,
		LLMTarget:    "stub/stub-model",
		STT:          transcriber,
		STTTarget:    "stub/stub-model",
		TTS:          speaker,
		TTSTarget:    "stub/stub-model",
		STS:          conversing,
		Logger:       logger,
	})
	s.Require().NoError(err)
	s.Require().NoError(s.agent.Join(s.ctx))
	s.events = collect(s.agent)
	s.T().Cleanup(func() { <-s.events.done })
	s.T().Cleanup(func() { _ = s.agent.Close() })
	return w
}

// hears has a participant say something the transcriber settles on.
func (s *AgentSuite) hears(w *swappable, participant stt.Participant, text string, answered int) {
	s.speak(participant)
	// A move closes the listeners it replaces, so the words go to the one the agent holds
	// now, which is the newest one opened.
	s.eventually(func() bool {
		s.agent.mu.Lock()
		_, listening := s.agent.listeners[participant.ID]
		s.agent.mu.Unlock()
		return listening && len(w.listener().transcribed()) > 0
	}, "nobody listened")
	w.listener().emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeFinal, Text: text, Language: "en", Confidence: 1})
	s.eventually(func() bool { return countOf[Spoke](s.reported()) >= answered }, "the turn was never answered")
}

var cascadeOnStub = Settings{LLMTarget: "stub/stub-model", STTTarget: "stub/stub-model", TTSTarget: "stub/stub-model"}

func (s *AgentSuite) TestMovingASessionOntoOtherModelsAnswersTheNextTurnOnThem() {
	w := s.joinSwappable()
	alice := stt.Participant{ID: "alice"}
	s.hears(w, alice, "hello", 1)
	first, firstVoice := w.model("stub"), w.voice("stub")
	spoken := len(firstVoice.spoken())

	s.Require().NoError(s.agent.SetSettings(s.ctx, Settings{
		LLMTarget: "other/other-model", STTTarget: "stub/stub-model", TTSTarget: "other/other-model", Voice: "ada",
	}))
	s.hears(w, alice, "and again", 2)

	s.Len(first.turns(), 1, "the old model must not answer once the session moved")
	s.Require().Len(w.model("other").turns(), 1)
	input := w.model("other").turns()[0].Input
	s.Equal("hello", input[0].Content, "the new model is handed the conversation so far")
	s.Equal("and again", input[len(input)-1].Content)
	s.NotEmpty(w.voice("other").spoken(), "the new voice speaks the reply")
	s.Len(firstVoice.spoken(), spoken, "the old voice says nothing more")
	changed, ok := firstOf[ModelsChanged](s.reported())
	s.Require().True(ok)
	s.False(changed.Native)
	s.Equal("ada", changed.Voice)
	s.Empty(changed.STS)
}

func (s *AgentSuite) TestAMoveThatCannotOpenEverythingLeavesTheSessionAsItWas() {
	w := s.joinSwappable()
	alice := stt.Participant{ID: "alice"}
	model := s.agent.LLM()

	err := s.agent.SetSettings(s.ctx, Settings{
		LLMTarget: "other/other-model", STTTarget: "stub/stub-model", TTSTarget: "nobody/nothing",
	})

	s.Require().Error(err)
	s.Same(model, s.agent.LLM())
	s.hears(w, alice, "hello", 1)
	s.Len(w.model("stub").turns(), 1, "the session still answers on the model it had")
	s.Empty(w.model("other").turns())
	s.Zero(countOf[ModelsChanged](s.reported()))
}

func (s *AgentSuite) TestAMoveOntoAModelThatAnswersNothingIsRefusedRatherThanGoingQuiet() {
	w := s.joinSwappable()
	alice := stt.Participant{ID: "alice"}
	w.refuse("other", errors.New("403 Forbidden"))

	err := s.agent.SetSettings(s.ctx, Settings{
		LLMTarget: "other/other-model", STTTarget: "stub/stub-model", TTSTarget: "stub/stub-model",
	})

	s.Require().Error(err)
	s.Contains(err.Error(), "403 Forbidden", "the caller is told why the model would not do")
	s.hears(w, alice, "hello", 1)
	s.Len(w.model("stub").turns(), 1, "the session keeps answering on the model that works")
	s.Zero(countOf[ModelsChanged](s.reported()))
}

func (s *AgentSuite) TestASessionMovesOntoASpeechToSpeechModelAndBackWithoutLeavingTheCall() {
	w := s.joinSwappable()
	alice := stt.Participant{ID: "alice"}
	s.hears(w, alice, "my name is Ada", 1)

	native := cascadeOnStub
	native.STSTarget = "openai/gpt-realtime-2"
	s.Require().NoError(s.agent.SetSettings(s.ctx, native))

	s.True(s.agent.Native())
	s.Nil(s.agent.LLM())
	speech := w.conversing()
	s.Require().NotNil(speech)
	s.Contains(speech.instructions, "Caller: my name is Ada", "the model taking over is told the conversation so far")
	s.Contains(speech.instructions, "You: Hello there.")
	s.speak(alice)
	s.eventually(func() bool { heard, _ := speech.hearing(); return heard > 0 }, "the caller's audio never reached the new model")

	s.Require().NoError(s.agent.SetSettings(s.ctx, cascadeOnStub))

	s.False(s.agent.Native())
	_, closed := speech.hearing()
	s.True(closed, "the speech-to-speech model is closed once the session moved off it")
	s.hears(w, alice, "are you still there", 2)
	requests := w.model("stub").turns()
	s.Require().Len(requests, 1)
	s.Equal("my name is Ada", requests[0].Input[0].Content, "the cascade picks the conversation back up")
	s.Equal(2, countOf[ModelsChanged](s.reported()))
	s.edge.mu.Lock()
	defer s.edge.mu.Unlock()
	s.False(s.edge.left, "the call is never left")
}
