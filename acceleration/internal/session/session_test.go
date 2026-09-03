package session

import (
	"context"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// settleFor is how long a test waits for something crossing goroutines to become true.
const settleFor = 3 * time.Second

// quietEdge is a call with no network in it. A session does not push audio, so what a test
// needs from an edge is only that joining and leaving work.
type quietEdge struct {
	inbound chan agent.InboundAudio

	mu     sync.Mutex
	joined bool
	left   bool
	// unheard is speech that has been published but has not gone out yet, which is what a
	// real edge holds while a reply is being spoken.
	unheard bool
}

func newQuietEdge() *quietEdge {
	return &quietEdge{inbound: make(chan agent.InboundAudio, 4)}
}

func (e *quietEdge) Join(context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.joined = true
	return nil
}

func (e *quietEdge) Audio() <-chan agent.InboundAudio { return e.inbound }

func (e *quietEdge) PublishAudio(audio.PcmData) error { return nil }

func (e *quietEdge) SpeechPending() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.unheard
}

func (e *quietEdge) DropSpeech() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.unheard = false
}

// holdSpeech makes the edge report published speech as still on its way out, the way a real
// one does while a reply is being spoken.
func (e *quietEdge) holdSpeech(unheard bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.unheard = unheard
}

func (e *quietEdge) Leave() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.left {
		return nil
	}
	e.left = true
	close(e.inbound)
	return nil
}

func (e *quietEdge) gone() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.left
}

// stubSTT hears nothing, which is what a session under test does until a transcript is
// pushed into it.
type stubSTT struct{ emitter *stt.Emitter }

func (s *stubSTT) Start(context.Context) error                     { return nil }
func (s *stubSTT) ProcessAudio(stt.PcmData, stt.Participant) error { return nil }
func (s *stubSTT) Events() <-chan stt.Event                        { return s.emitter.Events() }
func (s *stubSTT) Provider() string                                { return "stub" }
func (s *stubSTT) Model() string                                   { return "stub-stt" }

func (s *stubSTT) Close() error {
	s.emitter.Close()
	return nil
}

// stubLLM answers with a fixed reply and, on the first turn, whatever tool the test wants.
type stubLLM struct {
	emitter *llm.Emitter

	mu    sync.Mutex
	asked []llm.Request
	reply string
	calls []llm.ToolCall
}

func (s *stubLLM) Start(context.Context) error { return nil }

func (s *stubLLM) Respond(request llm.Request) error {
	s.mu.Lock()
	s.asked = append(s.asked, request)
	first := len(s.asked) == 1
	reply := s.reply
	var calls []llm.ToolCall
	if first {
		calls = append([]llm.ToolCall(nil), s.calls...)
	}
	s.mu.Unlock()

	if reply == "" && len(calls) == 0 {
		return nil
	}
	go func() {
		s.emitter.Send(llm.CompletionStarted{CompletionID: request.ID, At: time.Now()})
		if reply != "" {
			s.emitter.Send(llm.TextDelta{CompletionID: request.ID, Text: reply})
		}
		s.emitter.Send(llm.CompletionComplete{
			CompletionID: request.ID,
			Text:         reply,
			ToolCalls:    calls,
		})
	}()
	return nil
}

func (s *stubLLM) Interrupt(completionIDs ...string) error {
	for _, id := range completionIDs {
		s.emitter.Send(llm.CompletionComplete{CompletionID: id, Interrupted: true})
	}
	return nil
}

func (s *stubLLM) Events() <-chan llm.Event { return s.emitter.Events() }
func (s *stubLLM) Provider() string         { return "stub" }
func (s *stubLLM) Model() string            { return "stub-llm" }
func (s *stubLLM) Reasoning() bool          { return false }

func (s *stubLLM) Close() error {
	s.emitter.Close()
	return nil
}

func (s *stubLLM) requests() []llm.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.Request(nil), s.asked...)
}

// stubTTS produces one chunk of audio per piece of text.
type stubTTS struct {
	emitter *tts.Emitter

	mu   sync.Mutex
	said []tts.Request
}

func (s *stubTTS) Start(context.Context) error { return nil }

func (s *stubTTS) Synthesize(request tts.Request) error {
	s.mu.Lock()
	s.said = append(s.said, request)
	s.mu.Unlock()

	if request.Text != "" {
		s.emitter.Send(tts.AudioChunk{
			SynthesisID: request.ID,
			Audio:       audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1},
		})
	}
	if request.Final {
		s.emitter.Send(tts.SynthesisComplete{SynthesisID: request.ID, AudioDurationMs: 10})
	}
	return nil
}

func (s *stubTTS) Interrupt() error         { return nil }
func (s *stubTTS) Events() <-chan tts.Event { return s.emitter.Events() }
func (s *stubTTS) Provider() string         { return "stub" }
func (s *stubTTS) Model() string            { return "stub-tts" }
func (s *stubTTS) Streaming() bool          { return false }
func (s *stubTTS) Performs() bool           { return false }
func (s *stubTTS) Prompt() string           { return "" }

func (s *stubTTS) Close() error {
	s.emitter.Close()
	return nil
}

func (s *stubTTS) spoken() []tts.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]tts.Request(nil), s.said...)
}

// stubMemory records the scope it was asked under, which is what a memory filter has to
// reach for it to mean anything.
type stubMemory struct {
	mu    sync.Mutex
	scope memory.Scope
}

func (m *stubMemory) Recall(_ context.Context, query memory.Query) ([]memory.Memory, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.scope = query.Scope
	return nil, nil
}

func (m *stubMemory) Remember(context.Context, memory.Scope, []llm.Message) error { return nil }
func (m *stubMemory) Provider() string                                            { return "stub" }
func (m *stubMemory) Close() error                                                { return nil }

func (m *stubMemory) scopedTo() memory.Scope {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.scope
}

// stubConfig is one provider per modality, which is all these tests need from routing.
func stubConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{
			Provider:  "stub",
			Model:     "stub-model",
			Languages: []string{"en"},
			Realtime:  true,
		}},
		Aliases: map[string]routing.Alias{
			"en-low-latency": {Languages: []string{"en"}, RequireRealtime: true},
		},
	}
}

type SessionSuite struct {
	suite.Suite
	ctx context.Context

	manager *Manager
	edges   []*quietEdge
	model   *stubLLM
	voice   *stubTTS
	// remembers is the memory store the manager was built with, when a test wants one.
	remembers *stubMemory
}

func TestSessionSuite(t *testing.T) {
	suite.Run(t, new(SessionSuite))
}

func (s *SessionSuite) SetupTest() {
	s.ctx = context.Background()
	s.edges = nil
	s.remembers = nil
}

// manages builds a manager over stub providers. It is called by each test rather than in
// setup, because a test that wants memory has to say so before the manager is built.
func (s *SessionSuite) manages() {
	logger := slog.New(slog.DiscardHandler)

	ears := &stubSTT{emitter: stt.NewEmitter(64)}
	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return ears, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: stubConfig(), Registry: transcription, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	// The agent opens a voice model and a flow controller, in that order, and each needs
	// its own emitter: two sessions on one channel would each consume the other's events.
	s.model = &stubLLM{emitter: llm.NewEmitter(64), reply: "Hello."}
	var opened int
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llm.LLM, error) {
		defer func() { opened++ }()
		if opened == 0 {
			return s.model, nil
		}
		return &stubLLM{emitter: llm.NewEmitter(64)}, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: stubConfig(), Registry: reasoning, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	s.voice = &stubTTS{emitter: tts.NewEmitter(64)}
	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return s.voice, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: stubConfig(), Registry: speech, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	// A typed nil would make the interface non-nil, which is not the same as having no
	// memory at all.
	var remembering memory.Store
	if s.remembers != nil {
		remembering = s.remembers
	}

	manager, err := NewManager(ManagerOptions{
		LLM:    reasoner,
		STT:    transcriber,
		TTS:    speaker,
		Memory: remembering,
		Logger: logger,
		Edge: func(Spec, *slog.Logger) (agent.Edge, error) {
			edge := newQuietEdge()
			s.edges = append(s.edges, edge)
			return edge, nil
		},
	})
	s.Require().NoError(err)
	s.manager = manager
	s.T().Cleanup(func() { manager.Shutdown() })
}

// joins creates a session for a customer, defaulting everything a test did not name.
func (s *SessionSuite) joins(spec Spec) *Session {
	if spec.CallID == "" {
		spec.CallID = "call-1"
	}
	if spec.CustomerID == "" {
		spec.CustomerID = "acme"
	}
	if spec.LLMTarget == "" {
		spec.LLMTarget = "en-low-latency"
		spec.STTTarget = "en-low-latency"
		spec.TTSTarget = "en-low-latency"
	}

	created, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	return created
}

func (s *SessionSuite) eventually(condition func() bool, message string) {
	s.Require().Eventually(condition, settleFor, 5*time.Millisecond, message)
}

func (s *SessionSuite) TestASessionIsListedAndFoundByTheCustomerRunningIt() {
	s.manages()

	created := s.joins(Spec{CallID: "call-7"})

	s.Equal(Live, created.State())
	s.Equal("call-7", created.Spec().CallID)
	s.Equal("call-7", created.Spec().AgentID, "an agent id defaults to the call id")

	found, ok := s.manager.Get(created.ID(), "acme")
	s.Require().True(ok)
	s.Same(created, found)
	s.Len(s.manager.List("acme"), 1)
}

func (s *SessionSuite) TestRejoiningACallEndsTheSessionTheAgentLeftBehind() {
	// A session outlives the connection that asked for it, so a restarted agent is still
	// in its call. Both would hear each other and answer each other for the rest of it.
	s.manages()
	left := s.joins(Spec{CallID: "call-1"})

	rejoined := s.joins(Spec{CallID: "call-1"})

	s.Equal(Ended, left.State())
	s.Equal(Live, rejoined.State())
	s.Require().Len(s.edges, 2)
	s.True(s.edges[0].gone(), "the agent left behind is still in the call")
	s.False(s.edges[1].gone())

	listed := s.manager.List("acme")
	s.Require().Len(listed, 1)
	s.Same(rejoined, listed[0])
}

func (s *SessionSuite) TestASecondAgentCanJoinTheSameCall() {
	// Only an agent replacing itself supersedes. Two different agents in one call is a
	// conversation somebody meant to have.
	s.manages()
	first := s.joins(Spec{CallID: "call-1", UserID: "support"})

	second := s.joins(Spec{CallID: "call-1", UserID: "supervisor"})

	s.Equal(Live, first.State())
	s.Equal(Live, second.State())
	s.Len(s.manager.List("acme"), 2)
}

func (s *SessionSuite) TestTheRecordedCallSaysWhatItWasRunWith() {
	// A config decides these until a session overrides one, so neither answers "what
	// spoke on this call" alone. The row is what a finished call is read back from.
	created := &Session{
		id:      "session-9",
		created: time.Now(),
		spec: Spec{
			CustomerID:     "acme",
			CallID:         "call-9",
			AgentID:        "call-9",
			LLMTarget:      "gemini/gemini-3.5-flash-lite",
			STTTarget:      "gemini/gemini-3.5-transcribe-live",
			TTSTarget:      "elevenlabs/eleven_v3_conversational",
			SubagentTarget: "openai/gpt-5.6-sol",
			Instructions:   "Keep it short.",
		},
		skills: harness.Skills{Skills: []harness.Skill{
			{Name: "think", Description: "work it through", Instructions: "Reason."},
		}},
	}

	recorded := row(created)

	s.Equal("gemini/gemini-3.5-flash-lite", recorded.LLM)
	s.Equal("gemini/gemini-3.5-transcribe-live", recorded.STT)
	s.Equal("elevenlabs/eleven_v3_conversational", recorded.TTS)
	s.Equal("openai/gpt-5.6-sol", recorded.Subagent)
	s.Equal("Keep it short.", recorded.Instructions)
	s.Equal([]string{"think"}, recorded.Skills,
		"the row carries the names; the instructions behind them are in the registry")
}

func (s *SessionSuite) TestACallThatDelegatesNothingRecordsNoSkills() {
	// Without a subagent there is nobody to hand work to, so listing skills on the row
	// would claim the call could do something it could not.
	s.manages()

	created := s.joins(Spec{CallID: "call-10"})

	recorded := row(created)

	s.Empty(recorded.Subagent)
	s.Empty(recorded.Skills)
}

func (s *SessionSuite) TestACallThatNamesNoSkillsRecordsTheBuiltInSet() {
	s.manages()

	created := s.joins(Spec{CallID: "call-11", SubagentTarget: "en-low-latency"})

	s.Contains(row(created).Skills, "think")
}

// writes creates a session that holds the conversation in writing.
func (s *SessionSuite) writes(spec Spec) *Session {
	spec.Text = true
	if spec.CustomerID == "" {
		spec.CustomerID = "acme"
	}
	if spec.LLMTarget == "" {
		spec.LLMTarget = "en-low-latency"
	}

	created, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	return created
}

func (s *SessionSuite) TestATextSessionOpensNoEdgeBecauseThereIsNoCall() {
	s.manages()

	created := s.writes(Spec{})

	s.Equal(Live, created.State())
	s.Empty(created.Spec().CallID)
	s.NotEmpty(created.Spec().AgentID, "a text session is keyed by an agent id of its own")
	s.Empty(s.edges, "a conversation held in writing joins nothing")
}

func (s *SessionSuite) TestATextSessionCannotAlsoJoinACall() {
	s.manages()

	_, err := s.manager.Create(s.ctx, Spec{Text: true, CallID: "call-1", CustomerID: "acme"})

	s.ErrorContains(err, "holds no call")
}

func (s *SessionSuite) TestATextSessionAnswersInWriting() {
	s.manages()
	created := s.writes(Spec{})

	events, detach := created.Watch()
	defer detach()
	s.Require().NoError(created.Respond(s.ctx, "hello"))

	s.Equal("Hello.", awaitReply(events))
	s.Empty(s.voice.spoken(), "nothing is synthesised for a reader")
}

func (s *SessionSuite) TestAnotherCustomersSessionDoesNotExist() {
	// Two customers sharing a router must not be able to reach each other's calls, and a
	// session that reported itself as forbidden would confirm the id was real.
	s.manages()
	created := s.joins(Spec{})

	_, ok := s.manager.Get(created.ID(), "other")

	s.False(ok)
	s.Empty(s.manager.List("other"))

	closed, err := s.manager.Close(created.ID(), "other")
	s.Require().NoError(err)
	s.False(closed)
	s.Equal(Live, created.State(), "another customer ended the call")
}

func (s *SessionSuite) TestClosingASessionLeavesTheCallAndForgetsIt() {
	s.manages()
	created := s.joins(Spec{})

	closed, err := s.manager.Close(created.ID(), "acme")

	s.Require().NoError(err)
	s.True(closed)
	s.Equal(Ended, created.State())
	s.Require().Len(s.edges, 1)
	s.True(s.edges[0].gone(), "the agent stayed in the call")
	s.Empty(s.manager.List("acme"))
}

func (s *SessionSuite) TestAGreetingIsSpokenWithoutGoingThroughTheModel() {
	s.manages()

	s.joins(Spec{Greeting: "Hi, I'm listening."})

	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "nothing was said")
	s.Equal("Hi, I'm listening.", s.voice.spoken()[0].Text)
	s.Empty(s.model.requests(), "a greeting the caller wrote does not need a model")
}

func (s *SessionSuite) TestAWatcherSeesTheConversationAndStopsWhenItDetaches() {
	s.manages()
	created := s.joins(Spec{})

	events, detach := created.Watch()
	s.Require().NoError(created.Say(s.ctx, "something"))

	s.Require().NotNil(<-events, "the watcher saw nothing")

	detach()
	_, open := <-events
	s.False(open, "detaching left the channel open")
}

func (s *SessionSuite) TestACallersToolIsAskedForAndItsAnswerReachesTheModel() {
	s.manages()
	s.model.calls = []llm.ToolCall{{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`}}
	created := s.joins(Spec{
		Tools: []harness.Tool{{
			Name:        "lookup_order",
			Description: "find an order by its number",
		}},
	})

	events, detach := created.Watch()
	defer detach()
	s.Require().NoError(created.Respond(s.ctx, "where is my order"))

	asked := awaitToolCall(events)
	s.Require().NotNil(asked)
	s.Equal("lookup_order", asked.Name)
	s.Equal(`{"order":"12"}`, asked.Arguments)

	s.True(created.ResolveTool(asked.ID, "it ships tomorrow", ""))

	s.eventually(func() bool {
		for _, message := range created.voiceAgent.History() {
			if message.ToolCallID == asked.ID && message.Content == "it ships tomorrow" {
				return true
			}
		}
		return false
	}, "the answer never reached the conversation")
}

func (s *SessionSuite) TestAToolNobodyIsConnectedToRunFailsRatherThanWaiting() {
	// The model asked mid-reply and the caller is listening to the gap. Waiting out the
	// timeout against a disconnected watcher would be a pause nobody could explain.
	s.manages()
	created := s.joins(Spec{
		Tools: []harness.Tool{{Name: "lookup_order", Description: "find an order"}},
	})

	started := time.Now()
	result, err := created.tools.Run(s.ctx, llm.ToolCall{ID: "call-1", Name: "lookup_order"})

	s.Require().Error(err)
	s.Empty(result)
	s.ErrorContains(err, "nobody is connected")
	s.Less(time.Since(started), time.Second, "it waited for a tool nobody could run")
}

func (s *SessionSuite) TestAToolNobodyAnswersGivesUpRatherThanHangingTheTurn() {
	s.manages()
	created := s.joins(Spec{
		ToolTimeoutMs: 50,
		Tools:         []harness.Tool{{Name: "lookup_order", Description: "find an order"}},
	})

	events, detach := created.Watch()
	defer detach()

	_, err := created.tools.Run(s.ctx, llm.ToolCall{ID: "call-1", Name: "lookup_order"})

	s.Require().Error(err)
	s.ErrorContains(err, "did not answer")
	s.Require().NotNil(awaitToolCall(events), "the caller was never asked in the first place")
}

func (s *SessionSuite) TestAnAnswerToAToolNobodyIsWaitingOnIsDropped() {
	// The commonest cause is a caller answering a call that already timed out, which is
	// not worth failing anything over.
	s.manages()
	created := s.joins(Spec{})

	s.False(created.ResolveTool("call-nobody-asked-for", "here you go", ""))
}

func (s *SessionSuite) TestEndingACallStopsAToolWaitingOnAnAnswer() {
	s.manages()
	created := s.joins(Spec{
		ToolTimeoutMs: 60_000,
		Tools:         []harness.Tool{{Name: "lookup_order", Description: "find an order"}},
	})

	events, detach := created.Watch()
	defer detach()

	failed := make(chan error, 1)
	go func() {
		_, err := created.tools.Run(context.Background(),
			llm.ToolCall{ID: "call-1", Name: "lookup_order"})
		failed <- err
	}()
	s.Require().NotNil(awaitToolCall(events), "the caller was never asked")

	s.Require().NoError(created.Close())

	select {
	case err := <-failed:
		s.Require().Error(err)
		s.ErrorContains(err, "the call ended")
	case <-time.After(settleFor):
		s.Fail("the tool was still waiting on a call that had ended")
	}
}

func (s *SessionSuite) TestEndingACallWaitsForTheReplyToBeHeard() {
	// Leaving the call discards the audio that has not gone out yet, and a voice streams a
	// reply far faster than it is spoken. Ending the moment the provider stops sending
	// would cut the last of the reply off mid-word.
	s.manages()
	created := s.joins(Spec{})
	edge := s.edges[len(s.edges)-1]
	edge.holdSpeech(true)

	ended := make(chan error, 1)
	go func() { ended <- created.Close() }()

	select {
	case <-ended:
		s.Fail("the call was left while the reply was still on its way out")
	case <-time.After(100 * time.Millisecond):
	}

	edge.holdSpeech(false)

	select {
	case err := <-ended:
		s.Require().NoError(err)
		s.True(edge.gone(), "the call is left once the reply has been heard")
	case <-time.After(settleFor):
		s.Fail("the call never ended")
	}
}

func (s *SessionSuite) TestTheCallersMemoryFilterScopesWhatIsRecalled() {
	s.remembers = &stubMemory{}
	s.manages()

	s.joins(Spec{Memory: MemorySpec{
		UserID: "222",
		AppID:  "router",
		Filter: map[string]string{"company_id": "12312"},
	}})

	scope := s.remembers.scopedTo()
	s.Equal("222", scope.UserID, "the customer was recalled instead of the caller's user")
	s.Equal("router", scope.AppID)
	s.Equal(map[string]string{"company_id": "12312"}, scope.Extra)
}

func (s *SessionSuite) TestChangingTheInstructionsAppliesToTheNextTurn() {
	s.manages()
	created := s.joins(Spec{Instructions: "be brief"})

	created.SetInstructions("be thorough")
	s.Require().NoError(created.Respond(s.ctx, "hello"))

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("be thorough", s.model.requests()[0].Instructions)
}

func (s *SessionSuite) TestASessionNeedsACallToJoin() {
	s.manages()

	_, err := s.manager.Create(s.ctx, Spec{CustomerID: "acme"})

	s.ErrorContains(err, "call id is required")
}

func (s *SessionSuite) TestASandboxNobodyHasIsRefusedRatherThanIgnored() {
	// Silently running without one would leave the subagent doing arithmetic in its head
	// on a call whose caller asked for a sandbox precisely because that goes wrong.
	s.manages()

	_, err := s.manager.Create(s.ctx, Spec{CallID: "call-1", CustomerID: "acme", Sandbox: "docker"})

	s.ErrorContains(err, "docker")
}

func (s *SessionSuite) TestASandboxIsOnlyOpenedWhenTheDeploymentHasAKeyForIt() {
	s.T().Setenv("DAYTONA_API_KEY", "")
	s.manages()

	_, err := s.manager.Create(s.ctx, Spec{CallID: "call-1", CustomerID: "acme", Sandbox: "daytona"})

	s.ErrorContains(err, "DAYTONA_API_KEY")
}

func (s *SessionSuite) TestASessionCarriesItsSandboxToTheSubagent() {
	s.T().Setenv("DAYTONA_API_KEY", "key-1")
	s.manages()

	created := s.joins(Spec{Sandbox: "daytona", SubagentTarget: "en-low-latency"})

	s.Equal("daytona", created.Spec().Sandbox)
}

func (s *SessionSuite) TestNamingABuiltInSkillResolvesItWithoutDefiningIt() {
	s.manages()

	skills, err := s.manager.skills(s.ctx, Spec{
		CustomerID:     "acme",
		SubagentTarget: "en-low-latency",
		SkillNames:     []string{"think"},
	})

	s.Require().NoError(err)
	s.Require().Len(skills.Skills, 1, "a config naming one skill must not get the whole set")
	s.Equal("think", skills.Skills[0].Name)
	s.NotEmpty(skills.Skills[0].Instructions)
}

func (s *SessionSuite) TestASkillNothingDefinesIsRefused() {
	// Dropping it would leave the model told to hand work to a colleague who is not
	// there, and the caller waiting through small talk for an answer nobody is writing.
	s.manages()

	_, err := s.manager.skills(s.ctx, Spec{
		CustomerID:     "acme",
		SubagentTarget: "en-low-latency",
		SkillNames:     []string{"refund"},
	})

	s.ErrorContains(err, "refund")
}

func (s *SessionSuite) TestNamingNoSkillsTakesTheBuiltInSet() {
	s.manages()

	skills, err := s.manager.skills(s.ctx, Spec{
		CustomerID:     "acme",
		SubagentTarget: "en-low-latency",
	})

	s.Require().NoError(err)
	s.Greater(len(skills.Skills), 1)
}

func (s *SessionSuite) TestSkillsMeanNothingWithoutASubagentToRunThem() {
	s.manages()

	skills, err := s.manager.skills(s.ctx, Spec{
		CustomerID: "acme",
		SkillNames: []string{"nothing-defines-this"},
	})

	s.Require().NoError(err)
	s.Empty(skills.Skills)
}

func (s *SessionSuite) TestASkillSpelledOutWithoutADeadlineGetsOne() {
	// A zero deadline would abandon the work the instant it started.
	s.manages()

	skills, err := s.manager.skills(s.ctx, Spec{
		CustomerID:     "acme",
		SubagentTarget: "en-low-latency",
		Skills: &harness.Skills{Skills: []harness.Skill{{
			Name:         "refund",
			Description:  "work out what a caller is owed",
			Instructions: "read the order and the policy",
		}}},
	})

	s.Require().NoError(err)
	s.Require().Len(skills.Skills, 1)
	s.Positive(skills.Skills[0].Deadline)
}

func (s *SessionSuite) TestWhatWasSaidIsKeptSoTheCallCanBeReviewed() {
	s.manages()
	created := s.joins(Spec{})

	s.Require().NoError(created.Respond(s.ctx, "where is my order"))

	s.eventually(func() bool { return len(created.conversation()) > 0 },
		"the call left nothing to review")
	said := created.conversation()
	s.True(said[0].agent, "the agent's own reply is the agent's")
	s.Equal("Hello.", said[0].text)
}

func (s *SessionSuite) TestShutdownEndsEveryCallRatherThanDroppingIt() {
	s.manages()
	first := s.joins(Spec{CallID: "call-1"})
	second := s.joins(Spec{CallID: "call-2", CustomerID: "other"})

	s.Require().NoError(s.manager.Shutdown())

	s.Equal(Ended, first.State())
	s.Equal(Ended, second.State())
	for _, edge := range s.edges {
		s.True(edge.gone(), "a call was dropped rather than left")
	}
}

// awaitToolCall waits for the model to ask for a tool, skipping the conversation events
// that arrive alongside it.
// awaitReply returns the text of the first finished reply a watcher sees, or empty if the
// session said nothing before the deadline.
func awaitReply(events <-chan Event) string {
	deadline := time.After(settleFor)
	for {
		select {
		case event, open := <-events:
			if !open {
				return ""
			}
			if answered, ok := event.(agent.Responded); ok {
				return answered.Text
			}
		case <-deadline:
			return ""
		}
	}
}

func awaitToolCall(events <-chan Event) *ToolCall {
	deadline := time.After(settleFor)
	for {
		select {
		case event, open := <-events:
			if !open {
				return nil
			}
			if asked, ok := event.(ToolCall); ok {
				return &asked
			}
		case <-deadline:
			return nil
		}
	}
}
