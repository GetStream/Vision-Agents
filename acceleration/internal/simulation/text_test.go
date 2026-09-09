package simulation

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// answer is one thing the agent under test says back, and how long it takes about it.
type answer struct {
	text  string
	after time.Duration
}

// agentModel is the model inside the agent being tested. It answers what the test queued,
// which is how a turn that arrives in several pieces is arranged.
type agentModel struct {
	mu      sync.Mutex
	answers []answer
	// writing are the replies still being written, so closing the model settles them and
	// whoever is draining one is let go of.
	writing []*llmtest.Script
}

func (m *agentModel) Start(context.Context) error { return nil }

func (m *agentModel) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	script := llmtest.New(llm.StreamOptions{
		ResponseID: params.ID,
		Provider:   m.Provider(),
		Model:      m.Model(),
	})

	m.mu.Lock()
	queued := m.answers
	m.answers = nil
	m.writing = append(m.writing, script)
	m.mu.Unlock()

	// Nothing queued is a reply that never comes, which is a turn the test leaves hanging.
	if len(queued) == 0 {
		return script.Stream(), nil
	}
	go func() {
		for _, said := range queued {
			if said.after > 0 {
				time.Sleep(said.after)
			}
			script.OutputText(said.text)
		}
		script.Done()
	}()
	return script.Stream(), nil
}

func (m *agentModel) Provider() string               { return "scripted" }
func (m *agentModel) Model() string                  { return "scripted-model" }
func (m *agentModel) Capabilities() llm.Capabilities { return llm.Capabilities{} }

func (m *agentModel) Close() error {
	m.mu.Lock()
	writing := append([]*llmtest.Script(nil), m.writing...)
	m.mu.Unlock()

	for _, script := range writing {
		script.Done()
	}
	return nil
}

func (m *agentModel) says(answers ...answer) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.answers = answers
}

// deafSTT and mutedTTS are what a manager insists on having even for a conversation with no
// sound in it.
type deafSTT struct{ emitter *stt.Emitter }

func (s *deafSTT) Start(context.Context) error                     { return nil }
func (s *deafSTT) ProcessAudio(stt.PcmData, stt.Participant) error { return nil }
func (s *deafSTT) Events() <-chan stt.Event                        { return s.emitter.Events() }
func (s *deafSTT) Provider() string                                { return "scripted" }
func (s *deafSTT) Model() string                                   { return "scripted-model" }

func (s *deafSTT) Close() error {
	s.emitter.Close()
	return nil
}

type mutedTTS struct{ emitter *tts.Emitter }

func (t *mutedTTS) Start(context.Context) error { return nil }

func (t *mutedTTS) Synthesize(request tts.Request) error {
	if request.Text != "" {
		t.emitter.Send(tts.AudioChunk{
			SynthesisID: request.ID,
			Audio:       audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1},
		})
	}
	if request.Final {
		t.emitter.Send(tts.SynthesisComplete{SynthesisID: request.ID, AudioDurationMs: 10})
	}
	return nil
}

func (t *mutedTTS) Interrupt() error         { return nil }
func (t *mutedTTS) Events() <-chan tts.Event { return t.emitter.Events() }
func (t *mutedTTS) Provider() string         { return "scripted" }
func (t *mutedTTS) Model() string            { return "scripted-model" }
func (t *mutedTTS) Streaming() bool          { return false }
func (t *mutedTTS) Performs() bool           { return false }
func (t *mutedTTS) Prompt() string           { return "" }

func (t *mutedTTS) Close() error {
	t.emitter.Close()
	return nil
}

type TextSuite struct {
	suite.Suite
	ctx     context.Context
	manager *session.Manager
	model   *agentModel
}

func TestTextSuite(t *testing.T) {
	suite.Run(t, new(TextSuite))
}

func (s *TextSuite) SetupTest() {
	s.ctx = context.Background()
	logger := slog.New(slog.DiscardHandler)

	transcription := sttrouter.NewRegistry()
	transcription.Register("scripted", func(routing.Spec) (stt.STT, error) {
		return &deafSTT{emitter: stt.NewEmitter(64)}, nil
	})
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: scriptedConfig(), Registry: transcription, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	// The agent opens a voice model and a flow controller, in that order, and each needs
	// its own emitter: two sessions on one channel would each consume the other's events.
	s.model = &agentModel{}
	var opened int
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("scripted", func(routing.Spec) (llmrouter.Provider, error) {
		defer func() { opened++ }()
		if opened == 0 {
			return s.model, nil
		}
		return &agentModel{}, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: scriptedConfig(), Registry: reasoning, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	speech := ttsrouter.NewRegistry()
	speech.Register("scripted", func(routing.Spec) (tts.TTS, error) {
		return &mutedTTS{emitter: tts.NewEmitter(64)}, nil
	})
	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: scriptedConfig(), Registry: speech, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	manager, err := session.NewManager(session.ManagerOptions{
		LLM: reasoner, STT: transcriber, TTS: speaker, Logger: logger,
		// A conversation in writing joins nothing, so this is never reached. It is here
		// because a manager insists on knowing how it would have joined.
		Edge: func(session.Spec, *slog.Logger) (agent.Edge, error) {
			return nil, errors.New("there is no call to join")
		},
	})
	s.Require().NoError(err)
	s.manager = manager
	s.T().Cleanup(func() { _ = manager.Shutdown() })
}

// talks opens a conversation in writing against the scripted agent.
func (s *TextSuite) talks(greeting string) *written {
	runner := &Runner{sessions: s.manager, logger: slog.New(slog.DiscardHandler)}

	over, err := runner.speak(s.ctx, session.Spec{
		Text: true, CustomerID: "customer-1", Greeting: greeting,
		LLMTarget: "scripted/scripted-model", SubagentTarget: "scripted/scripted-model",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = over.Close() })

	held := over.(*written)
	held.within = 2 * time.Second
	return held
}

func (s *TextSuite) TestATurnIsWaitedForUntilTheAgentHasFinishedHavingItsSay() {
	held := s.talks("")
	s.model.says(
		answer{text: "Let me check that."},
		answer{text: " Yes, we have it.", after: 150 * time.Millisecond},
	)

	reply, err := held.Say(s.ctx, "Do you have pasta?")

	s.Require().NoError(err)
	// Both halves come back as one answer. Returning on the first would have the caller
	// talk over the second.
	s.Equal("Let me check that. Yes, we have it.", reply.Text)
}

func (s *TextSuite) TestATurnThatSaidNothingIsNotAnAnswer() {
	held := s.talks("")
	s.model.says(answer{text: ""}, answer{text: "Ordered."})

	reply, err := held.Say(s.ctx, "One pizza.")

	s.Require().NoError(err)
	s.Equal("Ordered.", reply.Text)
}

func (s *TextSuite) TestAnAgentThatNeverAnswersEndsTheTurnRatherThanTheConversation() {
	held := s.talks("")
	s.model.says()

	_, err := held.Say(s.ctx, "Hello?")

	s.Require().Error(err)
	s.Contains(err.Error(), "did not answer")
	// The conversation is still there to be judged on, and to be asked the next thing.
	s.Equal(session.Live, held.Session().State())
}

func (s *TextSuite) TestOneTurnIsNotAnsweredWithTheOneBeforeIt() {
	held := s.talks("")
	s.model.says(answer{text: "One bolognese."})
	first, err := held.Say(s.ctx, "A pasta bolognese please.")
	s.Require().NoError(err)
	s.Equal("One bolognese.", first.Text)

	s.model.says(answer{text: "Changed to pepperoni."})
	second, err := held.Say(s.ctx, "Change it to a pepperoni pizza.")

	s.Require().NoError(err)
	s.Equal("Changed to pepperoni.", second.Text)
}

func (s *TextSuite) TestTheGreetingIsSaidByTheConversationRatherThanBeforeItIsWatched() {
	held := s.talks("Northwind, how can I help?")

	// The manager says a greeting inside Create, before anything can attach to watch for
	// it, so a conversation that left it there would start halfway through.
	s.Equal("Northwind, how can I help?", held.Opening())

	s.model.says(answer{text: "Certainly."})
	reply, err := held.Say(s.ctx, "One pizza.")

	s.Require().NoError(err)
	s.Equal("Certainly.", reply.Text)
}
