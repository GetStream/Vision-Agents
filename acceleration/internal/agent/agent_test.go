package agent

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// settleFor is how long a test waits for an expectation to become true. The flow crosses
// several goroutines, so the alternative to waiting is asserting on a race.
const settleFor = 3 * time.Second

// loopbackEdge is a call with no network in it: audio is pushed in from the test and
// whatever the agent says is kept for inspection.
// It also satisfies Roster, so a test can say somebody arrived without anybody speaking,
// which is what an agent answering a call has to react to.
type loopbackEdge struct {
	inbound   chan InboundAudio
	attending chan Attendance

	mu sync.Mutex
	// published is what the agent has handed over and the caller has not heard yet, so it
	// is both the log of what was said and the queue DropSpeech throws away.
	published []audio.PcmData
	joined    bool
	left      bool
	// unheard is speech that has been published but has not gone out yet, which is what a
	// real edge holds while a reply is being spoken.
	unheard bool
}

func newLoopbackEdge() *loopbackEdge {
	return &loopbackEdge{
		inbound:   make(chan InboundAudio, 16),
		attending: make(chan Attendance, 16),
	}
}

func (e *loopbackEdge) Join(context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.joined = true
	return nil
}

func (e *loopbackEdge) Audio() <-chan InboundAudio { return e.inbound }

func (e *loopbackEdge) Attendance() <-chan Attendance { return e.attending }

func (e *loopbackEdge) PublishAudio(pcm audio.PcmData) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.published = append(e.published, pcm)
	return nil
}

func (e *loopbackEdge) SpeechPending() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.unheard
}

// DropSpeech throws away speech that has been published but not heard, the way a real edge
// empties its playout queue when the caller takes the floor.
func (e *loopbackEdge) DropSpeech() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.published = nil
	e.unheard = false
}

// holdSpeech makes the edge report published speech as still on its way out, the way a real
// one does while a reply is being spoken.
func (e *loopbackEdge) holdSpeech(unheard bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.unheard = unheard
}

func (e *loopbackEdge) Leave() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.left {
		return nil
	}
	e.left = true
	close(e.inbound)
	close(e.attending)
	return nil
}

// heard returns how much speech the agent published.
func (e *loopbackEdge) heard() []audio.PcmData {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]audio.PcmData(nil), e.published...)
}

// stubSTT turns whatever the test pushes into whatever transcript the test wants.
type stubSTT struct {
	emitter *stt.Emitter

	mu    sync.Mutex
	heard []audio.PcmData
}

func newStubSTT() *stubSTT { return &stubSTT{emitter: stt.NewEmitter(64)} }

func (s *stubSTT) Start(context.Context) error { return nil }

func (s *stubSTT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.heard = append(s.heard, pcm)
	return nil
}

func (s *stubSTT) Events() <-chan stt.Event { return s.emitter.Events() }

func (s *stubSTT) Close() error {
	s.emitter.Close()
	return nil
}

func (s *stubSTT) Provider() string { return "stub" }
func (s *stubSTT) Model() string    { return "stub-stt" }

func (s *stubSTT) transcribed() []audio.PcmData {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]audio.PcmData(nil), s.heard...)
}

// stubLLM answers with whatever deltas the test has queued.
type stubLLM struct {
	emitter *llm.Emitter

	mu         sync.Mutex
	asked      []llm.Request
	interrupts int
	// abandoned is every completion id an interrupt named, so a test can tell a targeted
	// interrupt from one that stopped everything.
	abandoned []string
	// reply is streamed one delta per element for each request, unless the test drives the
	// emitter itself.
	reply []string
	// then replaces reply from the second request onward, so a model that asked for help
	// on the first turn does not ask for it again once the answer has come back.
	then []string
	// calls are the tools the first reply asks for. Later replies ask for nothing, so a
	// model that transferred a caller does not transfer them again.
	calls []llm.ToolCall
	// keepCalling asks for them on every reply instead, which is the model that answers
	// a tool that failed by reaching for it again.
	keepCalling bool
}

func newStubLLM() *stubLLM { return &stubLLM{emitter: llm.NewEmitter(64)} }

func (s *stubLLM) Start(context.Context) error { return nil }

func (s *stubLLM) Respond(request llm.Request) error {
	s.mu.Lock()
	s.asked = append(s.asked, request)
	reply := append([]string(nil), s.reply...)
	first := len(s.asked) == 1
	if !first && s.then != nil {
		reply = append([]string(nil), s.then...)
	}
	var calls []llm.ToolCall
	if first || s.keepCalling {
		calls = append([]llm.ToolCall(nil), s.calls...)
	}
	s.mu.Unlock()

	if len(reply) == 0 && len(calls) == 0 {
		return nil
	}

	go func() {
		s.emitter.Send(llm.CompletionStarted{CompletionID: request.ID, At: time.Now()})
		var whole string
		for index, delta := range reply {
			whole += delta
			s.emitter.Send(llm.TextDelta{CompletionID: request.ID, Index: index, Text: delta})
		}
		s.emitter.Send(llm.CompletionComplete{
			CompletionID:       request.ID,
			Text:               whole,
			ToolCalls:          calls,
			InputTokens:        12,
			OutputTokens:       8,
			TimeToFirstTokenMs: 42,
		})
	}()
	return nil
}

func (s *stubLLM) Interrupt(completionIDs ...string) error {
	s.mu.Lock()
	s.interrupts++
	s.abandoned = append(s.abandoned, completionIDs...)
	s.mu.Unlock()

	// A real provider settles an abandoned completion rather than dropping it, because
	// the tokens it had already generated were still billed.
	for _, id := range completionIDs {
		s.emitter.Send(llm.CompletionComplete{CompletionID: id, Interrupted: true})
	}
	return nil
}

func (s *stubLLM) Events() <-chan llm.Event { return s.emitter.Events() }

func (s *stubLLM) Close() error {
	s.emitter.Close()
	return nil
}

func (s *stubLLM) Provider() string { return "stub" }
func (s *stubLLM) Model() string    { return "stub-llm" }
func (s *stubLLM) Reasoning() bool  { return false }

func (s *stubLLM) requests() []llm.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.Request(nil), s.asked...)
}

func (s *stubLLM) interrupted() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.interrupts
}

// stubTTS produces one chunk of audio per piece of text it is given.
type stubTTS struct {
	emitter   *tts.Emitter
	streaming bool
	// performs and prompt are what a voice that acts stage directions reports, so a test
	// can have the agent treat the stub as one.
	performs bool
	prompt   string

	mu         sync.Mutex
	said       []tts.Request
	interrupts int
	// silent stops the stub producing audio, so a test can hold a turn open.
	silent bool
}

func newStubTTS(streaming bool) *stubTTS {
	return &stubTTS{emitter: tts.NewEmitter(64), streaming: streaming}
}

func (s *stubTTS) Start(context.Context) error { return nil }

func (s *stubTTS) Synthesize(request tts.Request) error {
	s.mu.Lock()
	s.said = append(s.said, request)
	silent := s.silent
	s.mu.Unlock()

	if silent {
		return nil
	}
	if request.Text != "" {
		s.emitter.Send(tts.AudioChunk{
			SynthesisID: request.ID,
			Audio:       audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1},
		})
	}
	if request.Final {
		s.emitter.Send(tts.SynthesisComplete{
			SynthesisID:       request.ID,
			Characters:        int64(len(request.Text)),
			AudioDurationMs:   10,
			TimeToFirstByteMs: 5,
		})
	}
	return nil
}

func (s *stubTTS) Interrupt() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.interrupts++
	return nil
}

func (s *stubTTS) Events() <-chan tts.Event { return s.emitter.Events() }

func (s *stubTTS) Close() error {
	s.emitter.Close()
	return nil
}

func (s *stubTTS) Provider() string { return "stub" }
func (s *stubTTS) Model() string    { return "stub-tts" }
func (s *stubTTS) Streaming() bool  { return s.streaming }
func (s *stubTTS) Performs() bool   { return s.performs }
func (s *stubTTS) Prompt() string   { return s.prompt }

func (s *stubTTS) spoken() []tts.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]tts.Request(nil), s.said...)
}

func (s *stubTTS) interrupted() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.interrupts
}

// stubMemory is a memory store with no network in it: it returns whatever the test has
// put in it and keeps whatever the agent hands over.
type stubMemory struct {
	mu        sync.Mutex
	knows     []memory.Memory
	scope     memory.Scope
	learned   [][]llm.Message
	recallErr error
}

func (m *stubMemory) Recall(_ context.Context, query memory.Query) ([]memory.Memory, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.scope = query.Scope
	if m.recallErr != nil {
		return nil, m.recallErr
	}
	return m.knows, nil
}

func (m *stubMemory) Remember(_ context.Context, _ memory.Scope, messages []llm.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.learned = append(m.learned, messages)
	return nil
}

func (m *stubMemory) Provider() string { return "stub" }
func (m *stubMemory) Close() error     { return nil }

func (m *stubMemory) remembered() [][]llm.Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([][]llm.Message(nil), m.learned...)
}

func (m *stubMemory) scopedTo() memory.Scope {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.scope
}

// stubConfig is one provider per modality, which is all a flow test needs from routing.
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

// collector drains an agent's events for the life of one test, because the emitter applies
// backpressure on a reader that stops.
type collector struct {
	mu     sync.Mutex
	events []Event
	done   chan struct{}
}

func collect(agent *Agent) *collector {
	drained := &collector{done: make(chan struct{})}
	go func() {
		defer close(drained.done)
		for event := range agent.Events() {
			drained.mu.Lock()
			drained.events = append(drained.events, event)
			drained.mu.Unlock()
		}
	}()
	return drained
}

func (c *collector) seen() []Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Event(nil), c.events...)
}

type AgentSuite struct {
	suite.Suite
	ctx context.Context

	edge  *loopbackEdge
	voice *stubTTS
	model *stubLLM
	flow  *stubLLM
	ears  *stubSTT
	// opened counts the transcription sessions the agent asked for, which is how a test
	// tells a transcriber that was replaced from one that was never reopened.
	opened atomic.Int64
	// subagent is the second model, present only when a test asks for delegation.
	subagent *stubLLM
	// skills are what the model may hand over, when a test gives the agent a subagent.
	skills harness.Skills
	// line is what the agent may do to the call, present only when a test gives it
	// telephony.
	line *stubTelephony
	// tools are what the model may run, which mean nothing without something to run them.
	tools harness.Tools
	// runner carries out the tools this package does not own, present only when a test
	// gives the agent one.
	runner *stubToolRunner
	// duplex is how the agent listens and talks at the same time, off unless a test says
	// otherwise.
	duplex DuplexOptions
	// remembers is the memory store the agent joins with, when a test gives it one.
	remembers *stubMemory
	// knows is what the agent may look things up in, when a test gives it a knowledge
	// base, and namespace is which one it reads.
	knows     *stubKnowledge
	namespace string
	// finds is what the agent may find out about now, when a test gives it a search
	// provider.
	finds *stubSearch
	// performing is what a voice that acts stage directions asks to have said about it.
	// It is set before joining, because the stub voice is built there.
	performing string
	// records is where turns are written, when a test gives the agent a database.
	records *store.Store
	// agentID names the agent, so a test writing turns can find its own rows.
	agentID string

	agent  *Agent
	events *collector
}

func TestAgentSuite(t *testing.T) {
	suite.Run(t, new(AgentSuite))
}

func (s *AgentSuite) SetupTest() {
	s.ctx = context.Background()
	s.remembers = nil
	s.knows = nil
	s.namespace = ""
	s.finds = nil
	s.subagent = nil
	s.skills = harness.Skills{}
	s.line = nil
	s.tools = harness.Tools{}
	s.runner = nil
	s.duplex = DuplexOptions{}
	s.performing = ""
	if s.agentID == "" {
		s.agentID = "agent-1"
	}
}

// delegates gives the agent a subagent and one skill to hand work to it with.
func (s *AgentSuite) delegates() {
	s.subagent = newStubLLM()
	s.skills = harness.Skills{Skills: []harness.Skill{{
		Name:         "think",
		Description:  "hard questions",
		Instructions: "think it through",
		Deadline:     time.Minute,
	}}}
}

// reasoner is the model router the agent answers through. The agent opens the voice
// model, the flow controller and the optional subagent from it, in that order.
func (s *AgentSuite) reasoner(logger *slog.Logger) *llmrouter.Router {
	models := []llm.LLM{s.model, s.flow}
	if s.subagent != nil {
		models = append(models, s.subagent)
	}
	var opened int
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llm.LLM, error) {
		provider := models[min(opened, len(models)-1)]
		opened++
		return provider, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: stubConfig(), Registry: reasoning, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)
	return reasoner
}

// joinText builds an agent that holds the conversation in writing. It is given the model
// alone: a text agent joins no call and opens neither speech router.
func (s *AgentSuite) joinText() {
	s.model = newStubLLM()
	s.flow = newStubLLM()
	s.model.reply = []string{"Hello there. ", "How are you?"}

	logger := slog.New(slog.DiscardHandler)

	var subagentTarget string
	if s.subagent != nil {
		subagentTarget = "en-low-latency"
	}
	var reading knowledge.Store
	if s.knows != nil {
		reading = s.knows
	}

	agent, err := New(Options{
		Text:               true,
		Instructions:       "be brief",
		CustomerID:         "acme",
		AgentID:            s.agentID,
		SubagentTarget:     subagentTarget,
		Skills:             s.skills,
		Tools:              s.tools,
		LLM:                s.reasoner(logger),
		LLMTarget:          "en-low-latency",
		Knowledge:          reading,
		KnowledgeNamespace: s.namespace,
		Logger:             logger,
	})
	s.Require().NoError(err)
	s.agent = agent

	s.Require().NoError(agent.Join(s.ctx))

	s.events = collect(agent)
	s.T().Cleanup(func() { <-s.events.done })
	s.T().Cleanup(func() { _ = agent.Close() })
}

// join builds an agent over the stubs and joins the loopback call.
func (s *AgentSuite) join(streamingVoice bool) {
	s.edge = newLoopbackEdge()
	s.ears = newStubSTT()
	s.opened.Store(0)
	s.model = newStubLLM()
	s.flow = newStubLLM()
	s.voice = newStubTTS(streamingVoice)
	if s.performing != "" {
		s.voice.performs = true
		s.voice.prompt = s.performing
	}
	s.model.reply = []string{"Hello there. ", "How are you?"}
	s.flow.reply = []string{`{"disposition":"respond","floor":"stop"}`}

	logger := slog.New(slog.DiscardHandler)

	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) {
		s.opened.Add(1)
		return s.ears, nil
	})
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: stubConfig(), Registry: transcription, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	reasoner := s.reasoner(logger)

	var subagentTarget string
	if s.subagent != nil {
		subagentTarget = "en-low-latency"
	}

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
	var calling Telephony
	if s.line != nil {
		calling = s.line
	}
	var running ToolRunner
	if s.runner != nil {
		running = s.runner
	}
	var reading knowledge.Store
	if s.knows != nil {
		reading = s.knows
	}

	var finding *searchrouter.Router
	searchTarget := ""
	if s.finds != nil {
		finding = s.searchRouter()
		searchTarget = "stub/now"
	}

	agent, err := New(Options{
		Edge:               s.edge,
		Instructions:       "be brief",
		CustomerID:         "acme",
		AgentID:            s.agentID,
		AppID:              "router",
		Store:              s.records,
		SubagentTarget:     subagentTarget,
		Skills:             s.skills,
		Telephony:          calling,
		ToolRunner:         running,
		Tools:              s.tools,
		Duplex:             s.duplex,
		LLM:                reasoner,
		LLMTarget:          "en-low-latency",
		STT:                transcriber,
		STTTarget:          "en-low-latency",
		TTS:                speaker,
		TTSTarget:          "en-low-latency",
		Memory:             remembering,
		Knowledge:          reading,
		KnowledgeNamespace: s.namespace,
		Search:             finding,
		SearchTarget:       searchTarget,
		Logger:             logger,
	})
	s.Require().NoError(err)
	s.agent = agent

	s.Require().NoError(agent.Join(s.ctx))

	// The collector is waited on last, after the agent has closed its event channel, so no
	// goroutine from this test outlives it.
	s.events = collect(agent)
	s.T().Cleanup(func() { <-s.events.done })
	s.T().Cleanup(func() { _ = agent.Close() })
}

// speak pushes a chunk of a participant's audio into the call.
func (s *AgentSuite) speak(participant stt.Participant) {
	s.edge.inbound <- InboundAudio{
		Participant: participant,
		Audio:       audio.PcmData{Samples: make([]int16, 320), SampleRate: 16_000, Channels: 1},
	}
}

// says makes the transcriber report a settled turn.
// synthesises has the voice send one chunk of audio for a turn, the way a provider does
// while a reply is being spoken.
func (s *AgentSuite) synthesises(turnID string) {
	s.voice.emitter.Send(tts.AudioChunk{
		SynthesisID: turnID,
		Audio:       audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1},
	})
}

func (s *AgentSuite) says(participant stt.Participant, text string) {
	s.saysAfter(participant, text, 0)
}

// saysAfter reports a settled turn the transcriber spent the given time deciding on.
func (s *AgentSuite) saysAfter(participant stt.Participant, text string, processingMs float64) {
	s.ears.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             stt.ModeFinal,
		Text:             text,
		Language:         "en",
		Confidence:       1,
		ProcessingTimeMs: processingMs,
	})
}

// mumbles reports a settled turn the transcriber was not sure it heard right.
func (s *AgentSuite) mumbles(participant stt.Participant, text string, confidence float64) {
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeFinal,
		Text:        text,
		Language:    "en",
		Confidence:  confidence,
	})
}

// mutters reports a revision of a turn that has not finished, the way a transcriber does
// while someone is still talking.
func (s *AgentSuite) mutters(participant stt.Participant, text string) {
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeReplacement,
		Text:        text,
		Language:    "en",
	})
}

// eventually waits for a condition, which is how a test asserts on a flow that crosses
// goroutines without sleeping for a fixed time.
func (s *AgentSuite) eventually(condition func() bool, message string) {
	s.Require().Eventually(condition, settleFor, 5*time.Millisecond, message)
}

// reported returns the events seen so far.
func (s *AgentSuite) reported() []Event { return s.events.seen() }

func countOf[E Event](events []Event) int {
	var count int
	for _, event := range events {
		if _, ok := event.(E); ok {
			count++
		}
	}
	return count
}

func firstOf[E Event](events []Event) (E, bool) {
	for _, event := range events {
		if typed, ok := event.(E); ok {
			return typed, true
		}
	}
	var zero E
	return zero, false
}

func (s *AgentSuite) TestAnEdgeIsRequiredToHoldTheConversationOnACall() {
	_, err := New(Options{
		CustomerID: "acme",
		LLM:        &llmrouter.Router{},
		STT:        &sttrouter.Router{},
		TTS:        &ttsrouter.Router{},
	})

	s.ErrorContains(err, "edge")
}

func (s *AgentSuite) TestAConversationInWritingNeedsNoCallAndNoVoice() {
	_, err := New(Options{Text: true, CustomerID: "acme", LLM: &llmrouter.Router{}})

	s.NoError(err)
}

func (s *AgentSuite) TestEveryModalityIsRequired() {
	_, err := New(Options{Edge: newLoopbackEdge(), CustomerID: "acme"})

	s.ErrorContains(err, "llm")
}

func (s *AgentSuite) TestACustomerIsRequiredBecauseSomeoneIsBilled() {
	_, err := New(Options{
		Edge: newLoopbackEdge(),
		LLM:  &llmrouter.Router{},
		STT:  &sttrouter.Router{},
		TTS:  &ttsrouter.Router{},
	})

	s.ErrorContains(err, "customer")
}

func (s *AgentSuite) TestJoiningEntersTheCall() {
	s.join(true)

	s.eventually(func() bool { return countOf[Joined](s.reported()) == 1 }, "the agent never joined")
	s.True(s.edge.joined)
}

func (s *AgentSuite) TestJoiningTwiceIsRejected() {
	s.join(true)

	s.ErrorContains(s.agent.Join(s.ctx), "already joined")
}

func (s *AgentSuite) TestSomebodyArrivingIsReportedBeforeTheySpeak() {
	// This is what an agent answering a phone waits on. Nobody has said anything, so
	// audio is not evidence the caller is there.
	s.join(true)

	s.edge.attending <- Attendance{
		Participant: stt.Participant{ID: "session-1", UserID: "sip-+15550001111"},
		Joined:      true,
	}

	s.eventually(func() bool { return countOf[ParticipantJoined](s.reported()) == 1 },
		"the arrival was never reported")
	arrived, _ := firstOf[ParticipantJoined](s.reported())
	s.Equal("sip-+15550001111", arrived.Participant.UserID)
	s.False(arrived.At.IsZero())
}

func (s *AgentSuite) TestSomebodyLeavingIsReportedSeparately() {
	s.join(true)
	participant := stt.Participant{ID: "session-1", UserID: "sip-+15550001111"}
	s.edge.attending <- Attendance{Participant: participant, Joined: true}

	s.edge.attending <- Attendance{Participant: participant, Joined: false}

	s.eventually(func() bool { return countOf[ParticipantLeft](s.reported()) == 1 },
		"the caller hanging up was never reported")
	s.Equal(1, countOf[ParticipantJoined](s.reported()),
		"leaving must not read as arriving again")
}

func (s *AgentSuite) TestParticipantAudioIsTranscribed() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}

	s.speak(participant)

	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 },
		"the participant's audio never reached the transcriber")
}

func (s *AgentSuite) TestASettledTurnIsAnsweredAndSpoken() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	request := s.model.requests()[0]
	s.Equal("be brief", request.Instructions)
	s.Require().Len(request.Messages, 1)
	s.Equal("hello", request.Messages[0].Content)
	s.Equal(llm.User, request.Messages[0].Role)

	s.eventually(func() bool { return len(s.edge.heard()) > 0 },
		"the reply was never published to the call")
}

func (s *AgentSuite) TestARevisionReplacesTheWordsBeforeCadenceActs() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.ears.emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeReplacement, Text: "hel"})
	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("hello", s.model.requests()[0].Messages[0].Content)
}

func (s *AgentSuite) TestStablePartialSpeechIsAnsweredWithoutAFinalEvent() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mutters(participant, "can you help me")

	s.eventually(func() bool { return len(s.model.requests()) == 1 },
		"cadence should act without waiting for a provider final event")
	s.Equal("can you help me", s.model.requests()[0].Messages[0].Content)
}

func (s *AgentSuite) TestSettlingOnAnAnsweredUtteranceDoesNotAnswerItTwice() {
	// A transcriber restates an utterance when it settles on it, which lands after the
	// agent has started answering those same words. Answering the restatement as well is
	// what makes the agent reply twice to one thing.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mutters(participant, "how is your day going")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the model was never asked")
	s.says(participant, "How is your day going?")

	// The restatement is given longer than a cadence gap to become a second question,
	// because becoming one is the bug.
	time.Sleep(3 * defaultCadenceGap)
	s.Len(s.flow.requests(), 1, "the transcriber settling is not a second thing to answer")

	s.mutters(participant, "and what is the weather")
	s.eventually(func() bool { return len(s.flow.requests()) == 2 },
		"the agent stopped listening after the restatement")
}

func (s *AgentSuite) TestBackgroundSpeechIsIgnored() {
	s.join(true)
	s.flow.reply = []string{`{"disposition":"ignore","floor":"continue"}`}
	participant := stt.Participant{ID: "child", Name: "Child"}
	s.speak(participant)

	s.mutters(participant, "mom where is my backpack")

	s.eventually(func() bool { return len(s.flow.requests()) == 1 },
		"the flow controller never considered the speech")
	s.Empty(s.model.requests(), "speech addressed to somebody else should stay in the background")
	s.Empty(s.agent.History())
	s.Zero(countOf[Heard](s.reported()))
}

func (s *AgentSuite) TestAmbiguousSpeechAsksAClarifyingQuestion() {
	s.join(true)
	s.flow.reply = []string{`{"disposition":"clarify","floor":"continue"}`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mutters(participant, "do it like last time")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Contains(s.model.requests()[0].Instructions, "ambiguous")
}

func (s *AgentSuite) TestAnEmptyTurnIsNotAnswered() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "   ")

	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 }, "audio never arrived")
	s.Empty(s.model.requests(), "silence is not a question")
}

func (s *AgentSuite) TestAReplyIsSpokenSentenceBySentence() {
	// A streaming voice takes a turn's sentences as deltas of one utterance, so the turn
	// stays one billed synthesis.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	spoken := s.voice.spoken()
	s.Require().GreaterOrEqual(len(spoken), 2)
	s.Equal("Hello there.", spoken[0].Text, "the first sentence is sent before the reply ends")
	s.False(spoken[0].Final)
	s.True(spoken[len(spoken)-1].Final, "the utterance is closed once the model stops")

	ids := map[string]struct{}{}
	for _, request := range spoken {
		ids[request.ID] = struct{}{}
	}
	s.Len(ids, 1, "one turn is one utterance for a streaming voice")
}

func (s *AgentSuite) TestANonStreamingVoiceGetsOneRequestPerSentence() {
	s.join(false)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.voice.spoken()) == 2 },
		"both sentences should have been synthesised")

	for _, request := range s.voice.spoken() {
		s.True(request.Final, "a voice that cannot take deltas gets whole sentences")
	}
	s.NotEqual(s.voice.spoken()[0].ID, s.voice.spoken()[1].ID,
		"each sentence is its own synthesis, so each settles on its own")
}

func (s *AgentSuite) TestTheConversationIsRemembered() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	history := s.agent.History()
	s.Require().Len(history, 2)
	s.Equal(llm.Message{Role: llm.User, Content: "hello"}, history[0])
	s.Equal(llm.Assistant, history[1].Role)
	s.Equal("Hello there. How are you?", history[1].Content)
}

func (s *AgentSuite) TestASecondTurnCarriesTheWholeConversation() {
	// The history travels with every request, so a failover to another provider
	// mid-conversation loses nothing.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the first reply never finished")
	s.says(participant, "and again")

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the second turn was never asked")
	second := s.model.requests()[1]
	s.Require().Len(second.Messages, 3)
	s.Equal("hello", second.Messages[0].Content)
	s.Equal(llm.Assistant, second.Messages[1].Role)
	s.Equal("and again", second.Messages[2].Content)
}

func (s *AgentSuite) TestCompactionKeepsNewerTurnsVerbatim() {
	prefix := []llm.Message{
		{Role: llm.User, Content: "first question"},
		{Role: llm.Assistant, Content: "first answer"},
	}
	recent := []llm.Message{
		{Role: llm.User, Content: "new question"},
		{Role: llm.Assistant, Content: "new answer"},
	}
	emitter := NewEmitter(eventBuffer)
	defer emitter.Close()
	agent := &Agent{
		emitter: emitter,
		history: append(append([]llm.Message(nil), prefix...), recent...),
	}

	agent.applyCompaction(harness.Compacted{Prefix: prefix, Summary: "The first answer was established."})

	history := agent.History()
	s.Require().Len(history, 3)
	s.Equal(llm.System, history[0].Role)
	s.Contains(history[0].Content, "The first answer was established.")
	s.Equal(recent, history[1:])
	event, ok := (<-emitter.Events()).(ConversationCompacted)
	s.Require().True(ok)
	s.Equal(4, event.Before)
	s.Equal(3, event.After)
}

func (s *AgentSuite) TestLateCompactionCannotReplaceAChangedPrefix() {
	emitter := NewEmitter(eventBuffer)
	defer emitter.Close()
	agent := &Agent{
		emitter: emitter,
		history: []llm.Message{{Role: llm.User, Content: "corrected question"}},
	}

	agent.applyCompaction(harness.Compacted{
		Prefix:  []llm.Message{{Role: llm.User, Content: "old question"}},
		Summary: "stale summary",
	})

	s.Equal("corrected question", agent.History()[0].Content)
	select {
	case <-emitter.Events():
		s.Fail("a stale compaction was reported")
	default:
	}
}

func (s *AgentSuite) TestAFinishedTurnReportsWhatTheParticipantWaitedFor() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.saysAfter(participant, "hello", 30)

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never measured")
	turn, _ := firstOf[Turn](s.reported())

	s.Equal(participant, turn.Participant)
	s.False(turn.Interrupted)
	s.Equal(float64(30), turn.STTLatencyMs)
	s.Equal(float64(42), turn.LLMTTFTMs, "the model's own time to first token")
	s.Equal(float64(5), turn.TTSTTFBMs, "the voice's own time to first byte")
	s.Positive(turn.RoundtripMs, "the settled transcript to the first audio published")
	s.InDelta(turn.RoundtripMs+30, turn.SpeechEndToAudioMs, 0.001,
		"voice in to voice out includes the time the transcriber spent settling the turn")
	s.Equal(float64(10), turn.AudioOutMs)
}

func (s *AgentSuite) TestANonStreamingVoiceStillReportsOneTurnPerExchange() {
	// Each sentence is its own synthesis, but the participant had one exchange, so the
	// audio of all of them belongs to one turn.
	s.join(false)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never measured")
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(float64(20), turn.AudioOutMs, "both sentences count towards the one turn")
}

func (s *AgentSuite) TestAnInterruptedTurnIsStillMeasured() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, "no turn was started")

	s.says(participant, "actually, stop")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 },
		"an abandoned turn still happened and is still worth reporting")
	turn, _ := firstOf[Turn](s.reported())
	s.True(turn.Interrupted)
	s.Zero(turn.RoundtripMs, "the participant never heard anything")
}

func (s *AgentSuite) TestWithoutAMemoryStoreTheAgentOnlyKnowsItsInstructions() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("be brief", s.model.requests()[0].Instructions)
}

func (s *AgentSuite) TestWhatIsRememberedIsToldToTheModel() {
	s.remembers = &stubMemory{knows: []memory.Memory{{Text: "Prefers to be called Al"}}}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	instructions := s.model.requests()[0].Instructions
	s.Contains(instructions, "Prefers to be called Al")
	s.Contains(instructions, "be brief")
}

func (s *AgentSuite) TestAVoiceThatActsDirectionsSaysSoInTheInstructions() {
	s.performing = "You may write stage directions in square brackets."
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	instructions := s.model.requests()[0].Instructions
	s.Contains(instructions, "be brief", "the agent's own instructions come first")
	s.Contains(instructions, "square brackets", "and how to write for the voice after them")
}

func (s *AgentSuite) TestAVoiceThatCannotActDirectionsAsksForNone() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("be brief", s.model.requests()[0].Instructions,
		"a voice that would read a direction out must not be offered one")
}

func (s *AgentSuite) TestADirectionIsSpokenButNeverRead() {
	// The direction is for the voice to act. A caller reading the transcript would see
	// stage notes nobody said, and the model would read its own back on the next turn.
	s.performing = "You may write stage directions in square brackets."
	s.join(true)
	s.model.reply = []string{"[laughs] That ", "is a good one."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "tell me a joke")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	s.Contains(said(s.voice.spoken()), "[laughs]", "the voice is given the direction to act")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("That is a good one.", responded.Text)

	history := s.agent.History()
	s.Require().Len(history, 2)
	s.Equal("That is a good one.", history[1].Content)
}

func (s *AgentSuite) TestADirectionIsTakenOutOfWhatAVoiceCannotAct() {
	// Left in, the caller would hear the word "laughs" read out mid-sentence.
	s.join(true)
	s.model.reply = []string{"[laughs] That ", "is a good one."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "tell me a joke")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	s.NotContains(said(s.voice.spoken()), "laughs")
	s.Equal("That is a good one.", said(s.voice.spoken()))
}

func (s *AgentSuite) TestADirectionSplitAcrossTwoDeltasIsStillTakenOut() {
	// A model writes a few characters at a time, so a direction arrives in pieces. Taken
	// out one delta at a time, half of it would reach the caller.
	s.join(true)
	s.model.reply = []string{"Well ", "[sig", "hs] fine."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "how are you")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	responded, _ := firstOf[Responded](s.reported())
	s.Equal("Well fine.", responded.Text)
}

// said is every request's text joined, which is the reply as the voice was given it.
func said(requests []tts.Request) string {
	var spoken strings.Builder
	for _, request := range requests {
		spoken.WriteString(request.Text)
	}
	return spoken.String()
}

func (s *AgentSuite) TestMemoriesBelongToTheCustomerAndTheApp() {
	s.remembers = &stubMemory{}
	s.join(true)

	s.Equal(memory.Scope{AppID: "router", UserID: "acme"}, s.remembers.scopedTo())
}

func (s *AgentSuite) TestAnAgentThatCannotRecallStillTakesTheCall() {
	s.remembers = &stubMemory{recallErr: errors.New("mem0 is down")}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 },
		"a memory store that is down is not a broken conversation")
	s.Equal("be brief", s.model.requests()[0].Instructions)
}

func (s *AgentSuite) TestAFinishedExchangeIsRemembered() {
	s.remembers = &stubMemory{}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "I moved to Austin")

	s.eventually(func() bool { return len(s.remembers.remembered()) == 1 }, "nothing was remembered")
	exchange := s.remembers.remembered()[0]
	s.Require().Len(exchange, 2, "what was asked and what was answered")
	s.Equal(llm.Message{Role: llm.User, Content: "I moved to Austin"}, exchange[0])
	s.Equal(llm.Assistant, exchange[1].Role)
}

func (s *AgentSuite) TestARequestForHelpIsHandedOverRatherThanSpoken() {
	s.delegates()
	s.join(true)
	s.model.reply = []string{"Let me check that. ", `<ask skill="think">15% of 84.20</ask>`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")

	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")
	s.Equal("think it through", s.subagent.requests()[0].Instructions)

	for _, request := range s.voice.spoken() {
		s.NotContains(request.Text, "ask skill", "the caller must never hear the request itself")
	}
	s.Equal("Let me check that.", s.voice.spoken()[0].Text)
}

func (s *AgentSuite) TestLongThinkingWorkConfirmsTheAgentIsStillListening() {
	s.delegates()
	s.duplex = DuplexOptions{
		Backchannel:      true,
		BackchannelWords: 100,
		BackchannelGap:   10 * time.Millisecond,
	}
	s.join(true)
	s.model.reply = []string{`<ask skill="think">work through the itinerary</ask>`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "please work through the itinerary")

	s.eventually(func() bool { return s.agent.delegating() }, "the thinking task never started")
	s.eventually(func() bool { return countOf[Backchannel](s.reported()) == 1 },
		"a long thinking gap sounded like a dead call")
}

func (s *AgentSuite) TestDelegatedWorkIsReported() {
	s.delegates()
	s.join(true)
	s.model.reply = []string{`One moment. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")

	s.eventually(func() bool { return countOf[Delegated](s.reported()) == 1 }, "delegation was never reported")
	delegated, _ := firstOf[Delegated](s.reported())
	s.Equal("think", delegated.Skill)
	s.Equal("15% of 84.20", delegated.Prompt)

	s.eventually(func() bool { return countOf[TaskSettled](s.reported()) == 1 }, "the task never settled")
	settled, _ := firstOf[TaskSettled](s.reported())
	s.Equal("It is 12.63.", settled.Text)
	s.Positive(settled.ElapsedMs)
}

func (s *AgentSuite) TestAnAnswerComesBackAsATurnNobodyAskedFor() {
	// The caller was told an answer was coming, so it arrives without them asking again.
	s.delegates()
	s.join(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")

	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the answer coming back should have started a turn of its own")
	s.Contains(s.model.requests()[1].Instructions, "It is 12.63.")
	s.Contains(s.model.requests()[1].Instructions, "Tell the caller")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "the answer was never spoken")
	s.Contains(s.voice.spoken()[len(s.voice.spoken())-1].ID, "turn-",
		"the answer reaches the caller as a turn of its own")
}

func (s *AgentSuite) TestAnAnswerComingBackDoesNotCutOffTheReplyBeingSpoken() {
	// The voice model is still writing when the subagent lands. Taking the floor would
	// drop the rest of the live reply, which is the cut-off the caller hears.
	s.delegates()
	s.join(true)
	s.model.reply = nil
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = nil
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	turnA := s.model.requests()[0].ID

	s.model.emitter.Send(llm.CompletionStarted{CompletionID: turnA, At: time.Now()})
	s.model.emitter.Send(llm.TextDelta{
		CompletionID: turnA, Index: 0, Text: `<ask skill="think">15% of 84.20</ask>`,
	})
	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")

	taskID := s.subagent.requests()[0].ID
	s.subagent.emitter.Send(llm.CompletionStarted{CompletionID: taskID, At: time.Now()})
	s.subagent.emitter.Send(llm.TextDelta{CompletionID: taskID, Index: 0, Text: "It is 12.63."})
	s.subagent.emitter.Send(llm.CompletionComplete{CompletionID: taskID, Text: "It is 12.63."})
	s.eventually(func() bool { return countOf[TaskSettled](s.reported()) == 1 }, "the task never settled")
	s.Equal(1, len(s.model.requests()),
		"an answer coming back must not take the floor from a reply still being written")

	s.model.emitter.Send(llm.TextDelta{
		CompletionID: turnA, Index: 1, Text: "Let me check. Won't be a moment.",
	})
	s.model.emitter.Send(llm.CompletionComplete{CompletionID: turnA, Text: "Let me check. Won't be a moment."})
	s.eventually(func() bool { return countOf[Responded](s.reported()) >= 1 }, "the live reply was cut off")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("Let me check. Won't be a moment.", responded.Text)
	history := s.agent.History()
	s.Require().GreaterOrEqual(len(history), 2)
	s.Equal("Let me check. Won't be a moment.", history[1].Content)

	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the answer coming back should have started a turn of its own after the reply finished")
	s.Contains(s.model.requests()[1].Instructions, "It is 12.63.")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "the answer was never spoken")
}

func (s *AgentSuite) TestAnAnswerComingBackWaitsForSpeechToGoOut() {
	// An utterance the voice has finished sending is still on its way out of the edge.
	// Starting the follow-up on that word talks over the tail of the reply.
	s.delegates()
	s.join(true)
	s.edge.holdSpeech(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool {
		return countOf[Responded](s.reported()) == 1 && countOf[TaskSettled](s.reported()) == 1 &&
			countOf[Spoke](s.reported()) >= 1
	}, "the first reply never finished")
	s.Require().Never(func() bool { return len(s.model.requests()) > 1 },
		presenceTick+100*time.Millisecond, 10*time.Millisecond,
		"a follow-up must not start while speech is still going out")

	s.edge.holdSpeech(false)
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the answer coming back should have started a turn once the reply was heard")
	s.Contains(s.model.requests()[1].Instructions, "It is 12.63.")
}

func (s *AgentSuite) TestAStuckPlayoutSignalDoesNotStrandAFollowUp() {
	// The edge may keep reporting buffered speech even after the last chunk that was ever
	// published. Leaving a subagent's answer behind that forever turns "let me check" into
	// the last thing the caller hears.
	s.delegates()
	s.join(true)
	s.edge.holdSpeech(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool {
		return countOf[Responded](s.reported()) == 1 && countOf[TaskSettled](s.reported()) == 1 &&
			countOf[Spoke](s.reported()) >= 1
	}, "the first reply never finished")
	s.Require().Never(func() bool { return len(s.model.requests()) > 1 },
		presenceTick+100*time.Millisecond, 10*time.Millisecond,
		"a fresh tail still gets time to drain")

	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the answer coming back should not wait forever on a stuck playout signal")
	s.Contains(s.model.requests()[1].Instructions, "It is 12.63.")
}

func (s *AgentSuite) TestFinishingWaitsForAnAnswerThatHasNotBeenSpokenYet() {
	// The subagent's answer has started a turn, but nothing has been synthesised yet.
	// Leaving on that word hangs up before the caller hears what they were promised.
	s.delegates()
	s.join(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{}
	s.subagent.reply = []string{"It is 12.63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the answer coming back should have started a turn of its own")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) >= 1 },
		"the first reply was never spoken")

	ctx, cancel := context.WithTimeout(s.ctx, 50*time.Millisecond)
	defer cancel()
	s.Error(s.agent.Finish(ctx), "the answer has not been spoken yet, so finishing waits")
}

func (s *AgentSuite) TestTheHistoryKeepsWhatWasSaidRatherThanWhatWasAskedFor() {
	// A request for help was addressed to the harness, not the caller. Remembering it
	// would have the model reading its own instructions back on the next turn.
	s.delegates()
	s.join(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask> Won't be a moment.`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	history := s.agent.History()
	s.Require().Len(history, 2)
	s.Equal("Let me check.  Won't be a moment.", history[1].Content)
}

func (s *AgentSuite) TestWithoutASubagentTheModelAnswersEverythingItself() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("be brief", s.model.requests()[0].Instructions, "nothing is offered that nobody can do")
}

func (s *AgentSuite) TestClosingAbandonsWorkNobodyWillHear() {
	s.delegates()
	s.join(true)
	s.model.reply = []string{`<ask skill="think">15% of 84.20</ask>`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "the task never started")

	s.Require().NoError(s.agent.Close())

	s.eventually(func() bool { return countOf[TaskCancelled](s.reported()) == 1 },
		"work the caller will never hear should be abandoned")
	cancelled, _ := firstOf[TaskCancelled](s.reported())
	s.Equal(harness.ReasonClosed, cancelled.Reason)
}

func (s *AgentSuite) TestARelevantNewCandidateCancelsWorkFromTheOldPremise() {
	s.delegates()
	s.join(true)
	s.model.reply = []string{`<ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"Okay."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return s.agent.delegating() }, "the task never started")

	s.says(participant, "actually, use 20 percent")

	s.eventually(func() bool { return countOf[TaskCancelled](s.reported()) == 1 },
		"work based on the old premise was not cancelled")
	cancelled, _ := firstOf[TaskCancelled](s.reported())
	s.Equal(harness.ReasonSuperseded, cancelled.Reason)
	s.Equal(1, s.subagent.interrupted())
}

func (s *AgentSuite) TestTheAgentMurmursWhileSomeoneIsStillTalking() {
	s.duplex = DuplexOptions{Backchannel: true, BackchannelWords: 4}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mutters(participant, "so I was wondering whether")

	s.eventually(func() bool { return countOf[Backchannel](s.reported()) == 1 },
		"the agent never let them know it was still there")
	murmur, _ := firstOf[Backchannel](s.reported())
	s.NotEmpty(murmur.Text)
	s.Empty(s.model.requests(), "a murmur is not a turn, so it never costs a completion")

	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the murmur never reached the call")
}

func (s *AgentSuite) TestAskingWhetherAnythingElseIsNeededIsSomethingTheAgentSaid() {
	// The model has to see the question in the history, or the "no, that was everything"
	// that comes back is answering something it has no record of asking.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.agent.checkIn(participant, "Is there anything else I can help with?")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the question was never reported as speech")
	history := s.agent.History()
	s.Require().Len(history, 1)
	s.Equal(llm.Assistant, history[0].Role)
	s.Equal("Is there anything else I can help with?", history[0].Content)
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the question never reached the call")
}

func (s *AgentSuite) TestATranscriberThatDiesIsReplacedOnTheNextAudio() {
	// A provider cuts a session that has been idle, and the socket is dead afterwards.
	// The agent keeps one transcriber per participant, so without replacing it every
	// later chunk goes to a socket nobody is listening on: the caller talks for the rest
	// of the call and is never heard, which reads as an agent that has become very slow
	// rather than one that has stopped hearing.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.eventually(func() bool { return s.opened.Load() == 1 }, "nobody started listening")

	s.ears.emitter.Send(stt.Error{
		Provider: "stub", Model: "stub-stt", Context: "read", Fatal: true,
		Err: errors.New("websocket: close 1011 (internal server error): Deadline expired"),
	})

	// The caller keeps talking, as they would on a real call: the replacement is opened
	// by the next audio to arrive rather than by the failure itself.
	s.eventually(func() bool {
		s.speak(participant)
		return s.opened.Load() == 2
	}, "the dead transcriber was kept, so nothing this participant says is heard again")
}

func (s *AgentSuite) TestSomeoneCarryingOnAfterAMurmurIsNotAnInterruption() {
	// A murmur is meant to overlap. Treating the caller carrying on as barge-in would
	// report an interruption on every single one.
	s.duplex = DuplexOptions{Backchannel: true, BackchannelWords: 4}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.mutters(participant, "so I was wondering whether")
	s.eventually(func() bool { return countOf[Backchannel](s.reported()) == 1 }, "nothing was murmured")

	s.mutters(participant, "so I was wondering whether you could")

	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 }, "audio never arrived")
	s.Zero(countOf[Interrupted](s.reported()), "there was no reply to cut short")
	s.Zero(s.voice.interrupted())
}

func (s *AgentSuite) TestATurnItDidNotCatchIsCheckedRatherThanAnswered() {
	// Confidently answering the wrong question is worse on a phone call than asking.
	s.duplex = DuplexOptions{MinConfidence: 0.7}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mumbles(participant, "book a table for far", 0.4)

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Contains(s.model.requests()[0].Instructions, "did not catch")
}

func (s *AgentSuite) TestATurnItHeardClearlyIsJustAnswered() {
	s.duplex = DuplexOptions{MinConfidence: 0.7}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mumbles(participant, "book a table for four", 0.95)

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("be brief", s.model.requests()[0].Instructions)
}

func (s *AgentSuite) TestATranscriberThatReportsNoConfidenceIsTrusted() {
	// Never having been told is not the same as having been told the caller was
	// inaudible, and treating it that way would have the agent query every turn.
	s.duplex = DuplexOptions{MinConfidence: 0.7}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mumbles(participant, "book a table for four", 0)

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("be brief", s.model.requests()[0].Instructions)
}

func (s *AgentSuite) TestWithoutDuplexNothingIsMurmured() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mutters(participant, "so I was wondering whether you could help me with a booking")

	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 }, "audio never arrived")
	s.Zero(countOf[Backchannel](s.reported()))
	s.Empty(s.voice.spoken())
}

func (s *AgentSuite) TestRelatedOverlapShortensThenAnswersTheAddition() {
	s.join(true)
	s.model.reply = nil
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	s.flow.then = []string{`{"disposition":"respond","floor":"shorten"}`}

	s.says(participant, "only the vegetarian options")

	s.eventually(func() bool { return countOf[OverlapDecided](s.reported()) == 1 },
		"the overlap was never decided")
	overlap, _ := firstOf[OverlapDecided](s.reported())
	s.Equal(string(harness.Shorten), overlap.Action)
	s.eventually(func() bool { return s.model.interrupted() == 1 }, "the long answer kept generating")
	s.Zero(s.voice.interrupted(), "speech already sent should finish rather than stop abruptly")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the addition was never answered")
}

func (s *AgentSuite) TestAcknowledgementOverlapLetsTheCurrentReplyContinue() {
	s.join(true)
	s.model.reply = nil
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	first := s.model.requests()[0].ID
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}

	s.says(participant, "okay")

	s.eventually(func() bool { return countOf[OverlapDecided](s.reported()) == 1 },
		"the overlap was never decided")
	overlap, _ := firstOf[OverlapDecided](s.reported())
	s.Equal(string(harness.Continue), overlap.Action)
	s.Zero(s.model.interrupted())
	s.Len(s.model.requests(), 1, "the acknowledgement waits behind the current reply")

	s.model.emitter.Send(llm.CompletionComplete{CompletionID: first})
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the queued turn was not answered after the current reply")
}

func (s *AgentSuite) TestBargeInStopsTheModelAndTheVoice() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, "no turn was started")

	s.says(participant, "actually, stop")

	s.eventually(func() bool { return s.voice.interrupted() == 1 && s.model.interrupted() == 1 },
		"talking over the agent should stop it mid-sentence")
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the interruption was never reported")
}

func (s *AgentSuite) TestBargeInThrowsAwaySpeechNotHeardYet() {
	// Cancelling the voice only stops what it has not synthesised yet. A reply is streamed
	// far faster than it is spoken, so most of it is still queued at the edge, and leaving
	// it there is the caller being talked over after they took the floor.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.synthesises(s.model.requests()[0].ID)
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was never published")

	s.says(participant, "actually, stop")

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "no interruption")
	s.eventually(func() bool { return len(s.edge.heard()) == 0 },
		"the rest of the abandoned reply is still on its way to the caller")
}

func (s *AgentSuite) TestShorteningLeavesSpeechAlreadyOnItsWayOut() {
	// Shortening is the softer answer to an overlap: the model stops adding to the reply,
	// but what has already been said is still worth hearing, so nothing that has been
	// published is thrown away.
	s.join(true)
	s.model.reply = nil
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	s.synthesises(s.model.requests()[0].ID)
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was never published")
	s.flow.then = []string{`{"disposition":"respond","floor":"shorten"}`}

	s.says(participant, "only the vegetarian options")

	s.eventually(func() bool { return s.model.interrupted() == 1 }, "the long answer kept generating")
	s.NotEmpty(s.edge.heard(), "shortening threw away speech already on its way to the caller")
}

func (s *AgentSuite) TestAudioFromAnAbandonedTurnIsNotPublished() {
	// A provider keeps sending for a moment after being interrupted. Dropping that audio
	// here is what makes barge-in sound immediate.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	s.says(participant, "actually, stop")
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "no interruption")

	before := len(s.edge.heard())
	s.synthesises(s.model.requests()[0].ID)

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "no interruption")
	s.Equal(before, len(s.edge.heard()), "audio from the abandoned turn stays unheard")
}

func (s *AgentSuite) TestAToolResultDoesNotCutOffTheReplyAlreadyBeingSpoken() {
	// Only a caller talking over the agent abandons a turn. The agent starting one for
	// itself, to say what a tool came back with, does not: the reply that promised to go
	// and check is still coming out of the voice, and dropping the rest of it is the
	// agent cutting itself off mid-sentence.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	// Being asked a second time is the tool turn taking the floor from the first.
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the tool result was never answered")

	// The tail of the first reply, arriving late the way a provider sends it. It is a
	// size nothing else in the test produces, so it can be picked out of what was heard.
	promised := s.model.requests()[0].ID
	s.voice.emitter.Send(tts.AudioChunk{
		SynthesisID: promised,
		Audio:       audio.PcmData{Samples: make([]int16, 200), SampleRate: 16_000, Channels: 1},
	})

	s.eventually(func() bool {
		for _, chunk := range s.edge.heard() {
			if len(chunk.Samples) == 200 {
				return true
			}
		}
		return false
	}, "the end of the sentence the agent was already speaking was thrown away")
}

func (s *AgentSuite) TestBargeInWithNothingToSayIsIgnored() {
	s.join(true)

	s.ears.emitter.Send(stt.Connected{Provider: "stub"})
	s.speak(stt.Participant{ID: "alice"})

	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 }, "audio never arrived")
	s.Zero(countOf[Interrupted](s.reported()), "there was no reply to interrupt")
}

func (s *AgentSuite) TestSayGoesStraightToTheVoice() {
	// A greeting is text the agent already has, so putting it through a model would only
	// add latency and cost.
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the greeting was never published")
	s.Empty(s.model.requests(), "a greeting does not need a model")
	s.Require().Len(s.voice.spoken(), 1)
	s.True(s.voice.spoken()[0].Final)
}

func (s *AgentSuite) TestSimpleResponseAsksTheModel() {
	s.join(true)

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "what is the time"))

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("what is the time", s.model.requests()[0].Messages[0].Content)
}

func (s *AgentSuite) TestEachParticipantIsTranscribedSeparately() {
	// A speech-to-text stream is bound to one speaker, so two people cannot share one.
	s.join(true)

	s.speak(stt.Participant{ID: "alice"})
	s.speak(stt.Participant{ID: "bob"})

	s.eventually(func() bool { return len(s.ears.transcribed()) == 2 }, "not all audio arrived")
	s.Len(s.agent.listeners, 2)
}

func (s *AgentSuite) TestOneParticipantKeepsOneSession() {
	s.join(true)

	s.speak(stt.Participant{ID: "alice"})
	s.speak(stt.Participant{ID: "alice"})

	s.eventually(func() bool { return len(s.ears.transcribed()) == 2 }, "not all audio arrived")
	s.Len(s.agent.listeners, 1, "the same speaker keeps the session they already had")
}

func (s *AgentSuite) TestFinishWaitsForTheAgentToStopTalking() {
	s.join(true)
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()

	s.Require().NoError(s.agent.Say(s.ctx, "a long sentence"))

	ctx, cancel := context.WithTimeout(s.ctx, 50*time.Millisecond)
	defer cancel()
	s.Error(s.agent.Finish(ctx), "the utterance never settled, so finishing waits")

	s.voice.emitter.Send(tts.SynthesisComplete{
		SynthesisID:     s.voice.spoken()[0].ID,
		AudioDurationMs: 900,
	})
	s.NoError(s.agent.Finish(s.ctx))
}

func (s *AgentSuite) TestFinishWaitsForTheReplyToBeHeard() {
	// A voice streams a reply far faster than it is spoken, so an utterance the provider
	// has finished sending is still on its way out of the edge. Leaving on that word
	// discards the rest, and the caller hears the agent stop short of its last words.
	s.join(true)
	s.edge.holdSpeech(true)

	s.Require().NoError(s.agent.Say(s.ctx, "the last thing the caller hears"))
	s.voice.emitter.Send(tts.SynthesisComplete{
		SynthesisID:     s.voice.spoken()[0].ID,
		AudioDurationMs: 900,
	})

	ctx, cancel := context.WithTimeout(s.ctx, 50*time.Millisecond)
	defer cancel()
	s.Error(s.agent.Finish(ctx), "the reply has not been heard yet, so finishing waits")

	s.edge.holdSpeech(false)
	s.NoError(s.agent.Finish(s.ctx))
}

func (s *AgentSuite) TestSpokenAudioIsReported() {
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "nothing was reported as spoken")
	spoke, _ := firstOf[Spoke](s.reported())
	s.Equal(float64(10), spoke.AudioDurationMs)
	s.Equal(float64(5), spoke.TimeToFirstByteMs)
}

func (s *AgentSuite) TestTheModelsLatencyIsReported() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal(float64(42), responded.TimeToFirstTokenMs,
		"time to first token is what the participant waited for")
}

func (s *AgentSuite) TestAProviderFailureIsReportedWithoutEndingTheCall() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.ears.emitter.Send(stt.Error{Provider: "stub", Err: context.DeadlineExceeded})

	s.eventually(func() bool { return countOf[Error](s.reported()) == 1 }, "the failure was never reported")

	// The conversation carries on: one bad turn is a lost reply, not a lost call.
	s.says(participant, "hello")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the agent stopped answering")
}

func (s *AgentSuite) TestClosingLeavesTheCall() {
	s.join(true)

	s.Require().NoError(s.agent.Close())

	s.True(s.edge.left)
	s.Empty(s.agent.listeners)
}

func (s *AgentSuite) TestClosingTwiceIsSafe() {
	s.join(true)

	s.NoError(s.agent.Close())
	s.NoError(s.agent.Close())
}

func (s *AgentSuite) TestClosingEndsTheEventStream() {
	s.join(true)

	s.Require().NoError(s.agent.Close())

	// The consumer's range loop ends because the channel closes, and Left is the last
	// thing it sees.
	s.eventually(func() bool { return countOf[Left](s.reported()) == 1 }, "leaving was never reported")
	select {
	case _, open := <-s.agent.Events():
		s.False(open, "the channel is closed")
	case <-time.After(settleFor):
		s.Fail("the event channel stayed open")
	}
}

func (s *AgentSuite) TestSayingSomethingBeforeJoiningFails() {
	agent, err := New(Options{
		Edge:       newLoopbackEdge(),
		CustomerID: "acme",
		LLM:        &llmrouter.Router{},
		STT:        &sttrouter.Router{},
		TTS:        &ttsrouter.Router{},
	})
	s.Require().NoError(err)

	s.ErrorContains(agent.Say(s.ctx, "hello"), "not joined")
	s.ErrorContains(agent.SimpleResponse(s.ctx, "hello"), "not joined")
}

func (s *AgentSuite) TestAnAnswerInWritingIsReportedRatherThanSpoken() {
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "hello"))

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("Hello there. How are you?", responded.Text)
	s.Positive(countOf[ResponseDelta](s.reported()), "the reply should stream as it is written")
	s.Zero(countOf[Spoke](s.reported()), "there is no voice in a conversation held in writing")
}

func (s *AgentSuite) TestAGreetingInWritingIsReportedAsSaid() {
	s.joinText()

	s.Require().NoError(s.agent.Say(s.ctx, "Good morning."))

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the greeting never arrived")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("Good morning.", responded.Text)
}

func (s *AgentSuite) TestAReaderNeverSeesTheRequestForHelpEither() {
	s.delegates()
	s.joinText()
	s.model.reply = []string{"Let me check that. ", `<ask skill="think">15% of 84.20</ask>`}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "what is 15% of 84.20"))

	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")
	s.Equal("think it through", s.subagent.requests()[0].Instructions)

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("Let me check that.", responded.Text)
	for _, event := range s.reported() {
		if delta, ok := event.(ResponseDelta); ok {
			s.NotContains(delta.Text, "ask skill", "the reader must never see the request itself")
		}
	}
}

func (s *AgentSuite) TestAConversationInWritingLooksThingsUpTheSameWay() {
	s.reads(knowledge.Document{Text: "Refunds are given within 30 days.", Source: "handbook.md"})
	s.joinText()
	s.model.reply = []string{"Let me check that."}
	s.asksFor("lookup", `{"query":"refund window"}`)

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how long do I have to ask for a refund"))

	s.eventually(func() bool { return countOf[LookedUp](s.reported()) == 1 }, "nothing was looked up")
	looked, _ := firstOf[LookedUp](s.reported())
	s.Equal("refund window", looked.Query)
	s.Equal(1, looked.Documents)
}
