package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// settleFor is how long a test waits for an expectation to become true. The flow crosses
// several goroutines, so the alternative to waiting is asserting on a race.
//
// A wait ends as soon as what it waits for happens, so this only bounds how long a broken
// flow takes to fail. It must stay at or below the agent's fallback timers (defaultPatience,
// 3s, converse.go:68, stops a tool the interruption should have stopped): a longer wait
// is satisfied by the fallback and passes with the interruption broken.
const settleFor = 3 * time.Second

// closeFor is how long closesAtEnd gives the agent to close. Nothing there waits for a
// fallback to cover for a broken flow, so it may outlast settleFor on a slow runner.
const closeFor = 10 * time.Second

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
	// terms is the last vocabulary it was retuned to.
	terms []string
	// holdClose, if set, is waited on by Close, as a provider that flushes a final transcript
	// before it lets go of its socket.
	holdClose <-chan struct{}
	// closes counts the times it was closed.
	closes atomic.Int64
}

func newStubSTT() *stubSTT { return &stubSTT{emitter: stt.NewEmitter(64)} }

func (s *stubSTT) SetKeyterms(terms []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.terms = append([]string(nil), terms...)
	return nil
}

func (s *stubSTT) keyterms() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.terms...)
}

func (s *stubSTT) Start(context.Context) error { return nil }

func (s *stubSTT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.heard = append(s.heard, pcm)
	return nil
}

func (s *stubSTT) Events() <-chan stt.Event { return s.emitter.Events() }

func (s *stubSTT) Close() error {
	s.mu.Lock()
	hold := s.holdClose
	s.mu.Unlock()
	if hold != nil {
		<-hold
	}
	s.closes.Add(1)
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
	mu    sync.Mutex
	asked []llm.ResponseParams
	// reply is streamed one delta per element for each request, unless the test writes the
	// response itself.
	reply []string
	// thinking is streamed as reasoning before the reply.
	thinking []string
	// then replaces reply from the second request onward, so a model that asked for help
	// on the first turn does not ask for it again once the answer has come back.
	then []string
	// calls are the tools the first reply asks for. Later replies ask for nothing, so a
	// model that transferred a caller does not transfer them again.
	calls []llm.ToolCall
	// keepCalling asks for them on every reply instead, which is the model that answers
	// a tool that failed by reaching for it again.
	keepCalling bool
	// callsOn is which request, counted from 1, asks for calls when it is not the first.
	callsOn int
	// senders are who each request was asked in the name of, in the order asked.
	senders []routing.Caller
	// scheduled is whether each request was asked as a turn a scheduled task fired.
	scheduled []bool

	// holdCreate, if set, is waited on after the request is recorded and before a stream
	// is returned, so a test can interrupt while Create has not come back.
	holdCreate <-chan struct{}
	// holdCompletion delays the terminal event after Close, as an upstream that takes
	// time to notice cancellation does.
	holdCompletion <-chan struct{}

	// refuses, if set, is returned instead of a response: the model a session opens onto
	// happily but that answers nothing, which is what a rejected key looks like.
	refuses error

	// scripts are the responses handed out, keyed by the id the caller correlates on, so
	// a test can write one as it goes and see which were abandoned. A second response for
	// the same id, as a reply started before its ruling and the one asked in its place,
	// takes the key.
	scripts map[string]*llmtest.Script
	order   []string
	// handed is every response handed out, those that lost their key too, so Close ends
	// each one: a response nobody ends keeps its reader, and the agent's Close, waiting.
	handed []*llmtest.Script

	// composes are the lines the model writes when the agent asks it for one (compose),
	// one per request and the last again after that. Empty writes nothing, so a test that
	// is not about those lines hears none.
	composes []string
	// composeFails, if set, is what a request for a line fails with.
	composeFails error
	// holdCompose, if set, is waited on before a line is handed back, so a test can let the
	// moment it was asked for pass first.
	holdCompose <-chan struct{}
	// composed are the requests for a line, kept apart from asked so a test counting the
	// turns the model was asked for is not thrown by them.
	composed []llm.ResponseParams
}

func newStubLLM() *stubLLM { return &stubLLM{scripts: map[string]*llmtest.Script{}} }

func (s *stubLLM) Start(context.Context) error { return nil }

func (s *stubLLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	if strings.HasPrefix(params.ID, composePrefix) {
		return s.compose(params)
	}
	s.mu.Lock()
	s.asked = append(s.asked, params)
	sender, _ := routing.CallerFrom(ctx)
	s.senders = append(s.senders, sender)
	s.scheduled = append(s.scheduled, routing.ScheduledTurn(ctx))
	hold, completion, refuses := s.holdCreate, s.holdCompletion, s.refuses
	s.mu.Unlock()

	if refuses != nil {
		return nil, refuses
	}

	if hold != nil {
		select {
		case <-hold:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	script := llmtest.New(llm.StreamOptions{
		ResponseID: params.ID,
		Provider:   s.Provider(),
		Model:      s.Model(),
	})
	if completion != nil {
		script.Stream().Observe(func(event llm.Event) {
			if _, done := event.(llm.ResponseCompleted); done {
				<-completion
			}
		})
	}

	s.mu.Lock()
	s.scripts[params.ID] = script
	s.handed = append(s.handed, script)
	s.order = append(s.order, params.ID)
	s.handed = append(s.handed, script)
	reply := append([]string(nil), s.reply...)
	thinking := append([]string(nil), s.thinking...)
	first := len(s.asked) == 1
	if !first && s.then != nil {
		reply = append([]string(nil), s.then...)
	}
	var calls []llm.ToolCall
	if len(s.asked) == max(s.callsOn, 1) || s.keepCalling {
		calls = append([]llm.ToolCall(nil), s.calls...)
	}
	s.mu.Unlock()

	// A response with nothing queued stays in flight until the test writes it, which is
	// what lets a reply be caught mid-sentence.
	if len(reply) == 0 && len(calls) == 0 {
		return script.Stream(), nil
	}

	for _, delta := range thinking {
		script.ReasoningText(delta)
	}
	for _, delta := range reply {
		script.OutputText(delta)
	}
	if len(calls) > 0 {
		script.ToolCalls(calls...)
	}
	script.Usage(llm.Usage{InputTokens: 12, OutputTokens: 8})
	script.Done()
	return script.Stream(), nil
}

func (s *stubLLM) compose(params llm.ResponseParams) (*llm.Stream, error) {
	s.mu.Lock()
	s.composed = append(s.composed, params)
	var line string
	if len(s.composes) > 0 {
		line = s.composes[min(len(s.composed), len(s.composes))-1]
	}
	fails, hold := s.composeFails, s.holdCompose
	s.mu.Unlock()

	if hold != nil {
		<-hold
	}
	if fails != nil {
		return nil, fails
	}
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: s.Provider(), Model: s.Model()})
	if line != "" {
		script.OutputText(line)
	}
	script.Done()
	return script.Stream(), nil
}

// composeRequests are the requests the agent made for a line of its own.
func (s *stubLLM) composeRequests() []llm.ResponseParams {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.ResponseParams(nil), s.composed...)
}

// writes streams a piece of a response the test is writing itself.
func (s *stubLLM) writes(responseID, delta string) {
	s.opened(responseID).OutputText(delta)
}

// finishes settles a response the test was writing itself.
func (s *stubLLM) finishes(responseID string) {
	s.opened(responseID).Done()
}

// opened is the script for responseID once Create has stored it. The request is recorded
// before the script is stored, so a test that read the ID from requests() waits here for it.
func (s *stubLLM) opened(responseID string) *llmtest.Script {
	deadline := time.Now().Add(settleFor)
	for {
		if script := s.script(responseID); script != nil {
			return script
		}
		if time.Now().After(deadline) {
			panic("the response " + responseID + " was never opened")
		}
		time.Sleep(time.Millisecond)
	}
}

func (s *stubLLM) script(responseID string) *llmtest.Script {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.scripts[responseID]
}

// abandoned is whether the response for responseID was abandoned. A response whose request
// is recorded but whose script is not stored yet (Create has not come back) is not.
func (s *stubLLM) abandoned(responseID string) bool {
	script := s.script(responseID)
	return script != nil && script.Abandoned()
}

func (s *stubLLM) Close() error {
	s.mu.Lock()
	scripts := append([]*llmtest.Script(nil), s.handed...)
	s.mu.Unlock()

	for _, script := range scripts {
		script.Done()
	}
	return nil
}

func (s *stubLLM) Provider() string               { return "stub" }
func (s *stubLLM) Model() string                  { return "stub-llm" }
func (s *stubLLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }

// sentBy is who each request was asked in the name of.
func (s *stubLLM) sentBy() []routing.Caller {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]routing.Caller(nil), s.senders...)
}

// askedScheduled is whether each request was asked as a turn a scheduled task fired.
func (s *stubLLM) askedScheduled() []bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]bool(nil), s.scheduled...)
}

func (s *stubLLM) requests() []llm.ResponseParams {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.ResponseParams(nil), s.asked...)
}

// turns is what the model was asked on behalf of the caller, leaving out the one-word
// request a swap makes to prove the model can answer at all.
func (s *stubLLM) turns() []llm.ResponseParams {
	var asked []llm.ResponseParams
	for _, request := range s.requests() {
		if request.ID != "" {
			asked = append(asked, request)
		}
	}
	return asked
}

// interrupted is how many responses the agent closed part-way through.
func (s *stubLLM) interrupted() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	var abandoned int
	for _, id := range s.order {
		if s.scripts[id].Abandoned() {
			abandoned++
		}
	}
	return abandoned
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
	active     map[string]struct{}
	// reconnecting holds interruption open while the test delivers work that would
	// normally arrive during the provider's replacement connection.
	interruptStarted chan struct{}
	interruptRelease <-chan struct{}
	unavailable      bool
	startStarted     chan struct{}
	startRelease     <-chan struct{}
	startOnce        sync.Once
	// silent stops the stub producing audio, so a test can hold a turn open.
	silent bool
	// refuses, if set, is what every synthesis fails with.
	refuses error
}

func newStubTTS(streaming bool) *stubTTS {
	return &stubTTS{emitter: tts.NewEmitter(64), streaming: streaming, active: map[string]struct{}{}}
}

func (s *stubTTS) Start(ctx context.Context) error {
	s.mu.Lock()
	started, release := s.startStarted, s.startRelease
	s.mu.Unlock()
	if release == nil {
		return nil
	}
	if started != nil {
		s.startOnce.Do(func() { close(started) })
	}
	select {
	case <-release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *stubTTS) Synthesize(request tts.Request) error {
	s.mu.Lock()
	if s.unavailable {
		s.mu.Unlock()
		return errors.New("stub: not started")
	}
	s.said = append(s.said, request)
	silent, refuses := s.silent, s.refuses
	if request.Text != "" {
		s.active[request.ID] = struct{}{}
	}
	if request.Final {
		delete(s.active, request.ID)
	}
	s.mu.Unlock()

	if refuses != nil {
		return refuses
	}
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
			Characters:        int64(utf8.RuneCountInString(request.Text)),
			AudioDurationMs:   10,
			TimeToFirstByteMs: 5,
		})
	}
	return nil
}

func (s *stubTTS) Interrupt() error {
	s.mu.Lock()
	s.interrupts++
	release := s.interruptRelease
	started := s.interruptStarted
	var active []string
	if release != nil {
		s.interruptRelease = nil
		s.interruptStarted = nil
		s.unavailable = true
		for id := range s.active {
			active = append(active, id)
			delete(s.active, id)
		}
	}
	s.mu.Unlock()
	for _, id := range active {
		s.emitter.Send(tts.SynthesisComplete{
			SynthesisID: id,
			Interrupted: true,
		})
	}
	if release != nil {
		if started != nil {
			close(started)
		}
		<-release
		s.mu.Lock()
		s.unavailable = false
		s.mu.Unlock()
	}
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

// heldCompletionTTS keeps one emitted provider completion until the test releases it, even
// when the failed session closes.
type heldCompletionTTS struct {
	*stubTTS
	events   chan tts.Event
	held     chan struct{}
	release  <-chan struct{}
	heldOnce sync.Once
}

func newHeldCompletionTTS(provider *stubTTS, held chan struct{}, release <-chan struct{}) *heldCompletionTTS {
	voice := &heldCompletionTTS{
		stubTTS: provider,
		events:  make(chan tts.Event, 64),
		held:    held,
		release: release,
	}
	go func() {
		defer close(voice.events)
		for event := range provider.Events() {
			if _, ok := event.(tts.SynthesisComplete); ok {
				voice.heldOnce.Do(func() { close(voice.held) })
				<-voice.release
			}
			voice.events <- event
		}
	}()
	return voice
}

func (s *heldCompletionTTS) Events() <-chan tts.Event { return s.events }

// stubMemory is a memory store with no network in it: it returns whatever the test has
// put in it and keeps whatever the agent hands over.
type stubMemory struct {
	mu        sync.Mutex
	knows     []memory.Memory
	scope     memory.Scope
	learnedAs memory.Scope
	learned   [][]llm.Message
	recallErr error
}

func (m *stubMemory) Recall(_ context.Context, query memory.Query) (memory.Recalled, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.scope = query.Scope
	if m.recallErr != nil {
		return memory.Recalled{}, m.recallErr
	}
	return memory.Recalled{Memories: m.knows, Billed: memory.Billed{Requests: 1}}, nil
}

func (m *stubMemory) Remember(_ context.Context, scope memory.Scope, messages []llm.Message) (memory.Billed, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.learnedAs = scope
	m.learned = append(m.learned, messages)
	return memory.Billed{Requests: 1}, nil
}

func (m *stubMemory) writtenAs() memory.Scope {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.learnedAs
}

func (m *stubMemory) Truncate(context.Context, string, string) (memory.Billed, error) {
	return memory.Billed{Requests: 1}, nil
}

func (m *stubMemory) ForgetRun(context.Context, string, string) (memory.Billed, error) {
	return memory.Billed{Requests: 1}, nil
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
			Terms:     []options.Term{options.Speed},
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

	edge *loopbackEdge
	// edgeFactory wraps the loopback in an edge with more to it, for a test of what only such
	// an edge does. Without it the agent joins the loopback itself.
	edgeFactory func(*loopbackEdge) Edge
	voice       *stubTTS
	model       *stubLLM
	flow        *stubLLM
	ears        *stubSTT
	// backup, if set, is a second transcription model, after ears, that takes over a
	// session ears refuses.
	backup *stubSTT
	// opened counts the transcription sessions the agent asked for, which is how a test
	// tells a transcriber that was replaced from one that was never reopened.
	opened atomic.Int64
	// holdOpen, if set, is waited on by each transcription session the agent opens, as a
	// provider still dialling.
	holdOpen chan struct{}
	// replacement is the voice opened after the first, for a test whose agent loses one.
	replacement *stubTTS
	// subagent is the second model, present only when a test asks for delegation.
	subagent *stubLLM
	// skills are what the model may hand over, when a test gives the agent a subagent.
	skills harness.Skills
	// box is where the agent runs code, and shelf where what the code hands back is
	// published, when a test gives it them.
	box   *codeBox
	shelf *codeShelf
	// line is what the agent may do to the call, present only when a test gives it
	// telephony.
	line *stubTelephony
	// tools are what the model may run, which mean nothing without something to run them.
	tools harness.Tools
	// runner carries out the tools this package does not own, present only when a test
	// gives the agent one.
	runner *stubToolRunner
	// toolPolicy is what a test's tools ask of the agent (Options.ToolPolicy).
	toolPolicy func(tool string) ToolPolicy
	// duplex is how the agent listens and talks at the same time, off unless a test says
	// otherwise.
	duplex DuplexOptions
	// speculates starts replies before the flow controller has ruled, off unless a test
	// says otherwise.
	speculates bool
	// unified has the reply rule on its own turn in place of the flow controller.
	unified bool
	// remembers is the memory store the agent joins with, when a test gives it one.
	remembers *stubMemory
	// incognito holds the session off the record.
	incognito bool
	// knows is what the agent may look things up in, when a test gives it a knowledge
	// base, and namespace is which one it reads.
	knows     *stubKnowledge
	namespace string
	// finds is what the agent may find out about now, when a test gives it a search
	// provider.
	finds *stubSearch
	// guards screens what the agent may be asked, when a test gives it a policy.
	guards *stubGuardrail
	// performing is what a voice that acts stage directions asks to have said about it.
	// It is set before joining, because the stub voice is built there.
	performing string
	// voiceName is the voice the agent is configured to speak in.
	voiceName string
	// records is where turns are counted, and live where they are kept, when a test gives
	// the agent a database and a Redis.
	records *store.Store
	live    *live.Client
	// metering records what memory and knowledge did, when a test reads it back. Nil records
	// nothing.
	metering *routing.Recorder
	// billing is where the model router records what each request cost, priced, when a test
	// reads those rows. Nil records nothing.
	billing *store.Store
	// agentID names the agent, so a test writing turns can find its own rows.
	agentID string
	// voiceAsked is what the voice was opened with.
	voiceAsked routing.Spec
	// logs is where the agent's log lines are written, when a test reads them.
	logs *logLines
	// callID is the call the agent serves, empty unless a test names one.
	callID string

	agent  *Agent
	events *collector
	// over is closed when the test ends, before its agent closes, so a fake tool still
	// holding a call lets it go rather than hold the agent's shutdown.
	over chan struct{}
}

func TestAgentSuite(t *testing.T) {
	suite.Run(t, new(AgentSuite))
}

func (s *AgentSuite) SetupTest() {
	s.ctx = context.Background()
	s.callID = ""
	s.edgeFactory = nil
	s.holdOpen = nil
	s.remembers = nil
	s.incognito = false
	s.knows = nil
	s.namespace = ""
	s.finds = nil
	s.guards = nil
	s.subagent = nil
	s.skills = harness.Skills{}
	s.box = nil
	s.shelf = nil
	s.line = nil
	s.tools = harness.Tools{}
	s.runner = nil
	s.toolPolicy = nil
	s.duplex = DuplexOptions{}
	s.speculates = false
	s.unified = false
	s.performing = ""
	s.voiceName = ""
	s.logs = nil
	s.backup = nil
	s.over = make(chan struct{})
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
	models := []llmrouter.Provider{s.model, s.flow}
	if s.subagent != nil {
		models = append(models, s.subagent)
	}
	var opened int
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		provider := models[min(opened, len(models)-1)]
		opened++
		return provider, nil
	})
	config := stubConfig()
	if s.billing != nil {
		for i := range config.Providers {
			config.Providers[i].Price = routing.Price{PerMillionInputTokens: 1, PerMillionOutputTokens: 2}
		}
	}
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: config, Registry: reasoning, Store: s.billing, Logger: logger,
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
	var reading *metered.Knowledge
	if s.knows != nil {
		reading = metered.NewKnowledge(s.knows, s.metering)
	}
	var running ToolRunner
	if s.runner != nil {
		running = s.runner
	}
	var box sandbox.Sandbox
	if s.box != nil {
		box = s.box
	}
	var publish sandbox.Publisher
	if s.shelf != nil {
		publish = s.shelf.publish
	}

	agent, err := New(Options{
		Text:               true,
		Sandbox:            box,
		Publish:            publish,
		Instructions:       "be brief",
		CustomerID:         "acme",
		AgentID:            s.agentID,
		SubagentTarget:     subagentTarget,
		Skills:             s.skills,
		Tools:              s.tools,
		ToolRunner:         running,
		LLM:                s.reasoner(logger),
		LLMTarget:          "en-low-latency",
		Knowledge:          reading,
		KnowledgeNamespace: s.namespace,
		Guardrail:          s.screening(),
		Logger:             logger,
	})
	s.Require().NoError(err)
	s.agent = agent

	s.Require().NoError(agent.Join(s.ctx))

	s.events = collect(agent)
	s.closesAtEnd(agent)
}

// join builds an agent over the stubs and joins the loopback call.
func (s *AgentSuite) join(streamingVoice bool) {
	s.joinWithVoice(streamingVoice, nil)
}

// joinWithVoice builds an agent with an optional provider wrapper for event-order tests.
func (s *AgentSuite) joinWithVoice(streamingVoice bool, wrapVoice func(*stubTTS) tts.TTS) {
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
	if s.logs != nil {
		logger = slog.New(slog.NewJSONHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	}

	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) {
		s.opened.Add(1)
		if s.holdOpen != nil {
			<-s.holdOpen
		}
		return s.ears, nil
	})
	transcribing := stubConfig()
	if s.backup != nil {
		transcribing.Providers = append(transcribing.Providers, routing.ProviderConfig{
			Provider: "backup", Model: "backup-model", Languages: []string{"en"}, Realtime: true,
		})
		transcription.Register("backup", func(routing.Spec) (stt.STT, error) { return s.backup, nil })
	}
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: transcribing, Registry: transcription, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	reasoner := s.reasoner(logger)

	var subagentTarget string
	if s.subagent != nil {
		subagentTarget = "en-low-latency"
	}

	speech := ttsrouter.NewRegistry()
	opened := false
	speech.Register("stub", func(spec routing.Spec) (tts.TTS, error) {
		s.voiceAsked = spec
		if opened && s.replacement != nil {
			return s.replacement, nil
		}
		opened = true
		if wrapVoice != nil {
			return wrapVoice(s.voice), nil
		}
		return s.voice, nil
	})
	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: stubConfig(), Registry: speech, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	var remembering *metered.Memory
	if s.remembers != nil {
		remembering = metered.NewMemory(s.remembers, s.metering)
	}
	var calling Telephony
	if s.line != nil {
		calling = s.line
	}
	var running ToolRunner
	if s.runner != nil {
		running = s.runner
	}
	var reading *metered.Knowledge
	if s.knows != nil {
		reading = metered.NewKnowledge(s.knows, s.metering)
	}

	var finding *searchrouter.Router
	searchTarget := ""
	if s.finds != nil {
		finding = s.searchRouter()
		searchTarget = "stub/now"
	}

	var edge Edge = s.edge
	if s.edgeFactory != nil {
		edge = s.edgeFactory(s.edge)
	}

	agent, err := New(Options{
		Edge:               edge,
		Instructions:       "be brief",
		CustomerID:         "acme",
		AgentID:            s.agentID,
		AppID:              "router",
		SessionID:          "session-1",
		CallID:             s.callID,
		Incognito:          s.incognito,
		Store:              s.records,
		Live:               s.live,
		SubagentTarget:     subagentTarget,
		Skills:             s.skills,
		Telephony:          calling,
		ToolRunner:         running,
		ToolPolicy:         s.toolPolicy,
		Tools:              s.tools,
		Duplex:             s.duplex,
		SpeculativeReplies: s.speculates,
		UnifiedReplies:     s.unified,
		LLM:                reasoner,
		LLMTarget:          "en-low-latency",
		STT:                transcriber,
		STTTarget:          "en-low-latency",
		TTS:                speaker,
		TTSTarget:          "en-low-latency",
		Voice:              s.voiceName,
		Memory:             remembering,
		Knowledge:          reading,
		KnowledgeNamespace: s.namespace,
		Search:             finding,
		SearchTarget:       searchTarget,
		Guardrail:          s.screening(),
		Logger:             logger,
	})
	s.Require().NoError(err)
	s.agent = agent

	s.Require().NoError(agent.Join(s.ctx))

	// The collector is waited on last, after the agent has closed its event channel, so no
	// goroutine from this test outlives it.
	s.events = collect(agent)
	s.closesAtEnd(agent)
}

// closesAtEnd closes the agent once the test ends. It lets go of the fake tools first, then
// gives the agent and its event stream closeFor to close, so a goroutine nothing will end
// fails the test that left it instead of holding the package until go test's 10 m timeout.
// The test's own T is kept, because the suite has moved back to the parent by cleanup.
func (s *AgentSuite) closesAtEnd(agent *Agent) {
	t, over, events := s.T(), s.over, s.events
	t.Cleanup(func() {
		select {
		case <-over:
		default:
			close(over)
		}
		closed := make(chan struct{})
		go func() {
			_ = agent.Close()
			<-events.done
			close(closed)
		}()
		select {
		case <-closed:
		case <-time.After(closeFor):
			t.Errorf("the agent did not close within %s: one of its goroutines waits on something nothing will end", closeFor)
		}
	})
}

// speak pushes a chunk of a participant's audio into the call.
func (s *AgentSuite) speak(participant stt.Participant) {
	s.edge.inbound <- InboundAudio{
		Participant: participant,
		Audio:       audio.PcmData{Samples: make([]int16, 320), SampleRate: 16_000, Channels: 1},
	}
}

// setClock sets the time the agent reads.
func (s *AgentSuite) setClock(at time.Time) {
	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	s.agent.now = func() time.Time { return at }
}

// listening reports whether the agent has a transcription session for a participant.
func (s *AgentSuite) listening(participantID string) bool {
	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	_, ok := s.agent.listeners[participantID]
	return ok
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

// heardSomeOf has some of a reply go out, so the caller's words after are talking over it.
func (s *AgentSuite) heardSomeOf(turnID string) {
	s.synthesises(turnID)
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "none of the reply went out")
}

// replyWentOut waits until the reply the agent is saying has begun to go out. Words said
// before that are the caller moving on from a reply they never heard, and words said after it
// are them talking over it; a stub voice says a reply within microseconds, so a test about one
// of the two waits for its side of that line rather than racing it.
func (s *AgentSuite) replyWentOut() {
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.voiced == s.agent.speakingTurn
	}, "the reply never went out")
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

// saysInVoice reports a settled turn the transcriber could tell the voice of, which is
// what a microphone with more than one person at it sounds like.
func (s *AgentSuite) saysInVoice(participant stt.Participant, text, voice string) {
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeFinal,
		Speaker:     voice,
		Text:        text,
		Language:    "en",
		Confidence:  1,
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

// sfuEdge is a loopback edge that says which SFU it is on.
type sfuEdge struct {
	*loopbackEdge
	sfu SFU
}

func (e sfuEdge) SFU() SFU { return e.sfu }

func (s *AgentSuite) TestJoinedSaysWhichSFUTheAgentIsOn() {
	sfu := SFU{Edge: "sfu-a1", Host: "sfu-a1.example.com"}
	s.edgeFactory = func(edge *loopbackEdge) Edge { return sfuEdge{loopbackEdge: edge, sfu: sfu} }
	s.join(true)

	s.eventually(func() bool { return countOf[Joined](s.reported()) == 1 }, "the agent never joined")
	joined, _ := firstOf[Joined](s.reported())
	s.Equal(sfu, joined.SFU)
}

func (s *AgentSuite) TestJoinedNamesNoSFUOnAnEdgeThatCannotSay() {
	s.join(true)

	s.eventually(func() bool { return countOf[Joined](s.reported()) == 1 }, "the agent never joined")
	joined, _ := firstOf[Joined](s.reported())
	s.Equal(SFU{}, joined.SFU)
}

func (s *AgentSuite) TestTheVoiceIsNotAskedForASpeed() {
	s.join(true)

	s.Nil(s.voiceAsked.TTS.Speed, "naming a speed narrows the voices that may answer")
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

func (s *AgentSuite) TestThoseAlreadyInTheCallAreReportedBeforeJoined() {
	// Joined is what a session takes to mean everyone already there has been reported, so
	// a call that looks empty at that point is empty.
	s.edgeFactory = func(edge *loopbackEdge) Edge {
		edge.attending <- Attendance{Participant: stt.Participant{ID: "session-1", UserID: "sip-+15550001111"}, Joined: true}
		return edge
	}
	s.join(true)

	s.eventually(func() bool { return countOf[Joined](s.reported()) == 1 }, "the agent never joined")
	var order []string
	for _, event := range s.reported() {
		switch event.(type) {
		case ParticipantJoined:
			order = append(order, "participant")
		case Joined:
			order = append(order, "joined")
		}
	}
	s.Equal([]string{"participant", "joined"}, order)
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

func (s *AgentSuite) TestSomebodyLeavingClosesTheirTranscriptionSession() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 }, "the audio never arrived")

	s.edge.attending <- Attendance{Participant: participant, Joined: false}

	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return len(s.agent.listeners) == 0
	}, "a participant who left still has an open transcription session")
}

func (s *AgentSuite) TestASlowTranscriberToCloseDoesNotDelayTheNextArrival() {
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 }, "the audio never arrived")
	release := make(chan struct{})
	s.ears.mu.Lock()
	s.ears.holdClose = release
	s.ears.mu.Unlock()
	s.T().Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})

	s.edge.attending <- Attendance{Participant: alice, Joined: false}
	s.edge.attending <- Attendance{Participant: stt.Participant{ID: "bob"}, Joined: true}

	s.eventually(func() bool { return countOf[ParticipantJoined](s.reported()) == 1 },
		"a departed participant's session closing held up the next arrival")
	close(release)
}

func (s *AgentSuite) TestAudioFromSomebodyWhoLeftOpensNoTranscriptionSession() {
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.edge.attending <- Attendance{Participant: alice, Joined: true}
	s.edge.attending <- Attendance{Participant: alice, Joined: false}
	s.eventually(func() bool { return countOf[ParticipantLeft](s.reported()) == 1 },
		"the departure was never reported")

	s.speak(alice)
	bob := stt.Participant{ID: "bob"}
	s.speak(bob)

	// Frames are handled in the order they come, so once bob has a session alice's frame
	// has been dealt with.
	s.eventually(func() bool { return s.listening(bob.ID) }, "bob's audio never arrived")
	s.False(s.listening(alice.ID), "a participant who left has a transcription session")
	s.EqualValues(1, s.opened.Load(), "a session was opened for a participant who left")
	s.Zero(countOf[Error](s.reported()), "audio from a participant who left was reported as a failure")
}

func (s *AgentSuite) TestSomebodyStillTalkingAfterALeaveIsTranscribedAgain() {
	s.join(true)
	left := time.Now()
	s.setClock(left)
	alice := stt.Participant{ID: "alice"}
	s.edge.attending <- Attendance{Participant: alice, Joined: true}
	s.edge.attending <- Attendance{Participant: alice, Joined: false}
	s.eventually(func() bool { return countOf[ParticipantLeft](s.reported()) == 1 },
		"the departure was never reported")

	// The agent was held up for a minute, say opening somebody else's session, before the
	// audio that trailed the leave reached it.
	trailed := left.Add(time.Minute)
	s.setClock(trailed)
	bob := stt.Participant{ID: "bob"}
	s.speak(alice)
	s.speak(bob)
	s.eventually(func() bool { return s.listening(bob.ID) }, "bob's audio never arrived")
	s.False(s.listening(alice.ID), "audio that trailed a leave opened a transcription session")

	s.setClock(trailed.Add(999 * time.Millisecond))
	s.speak(alice)
	carol := stt.Participant{ID: "carol"}
	s.speak(carol)
	s.eventually(func() bool { return s.listening(carol.ID) }, "carol's audio never arrived")
	s.False(s.listening(alice.ID), "audio that trailed a leave opened a transcription session")

	s.setClock(trailed.Add(time.Second))
	s.speak(alice)

	s.eventually(func() bool { return s.listening(alice.ID) },
		"a participant still talking after a leave is never transcribed again")
}

func (s *AgentSuite) TestSomebodyLeavingWhileTheirSessionOpensHasItClosed() {
	s.join(true)
	release := make(chan struct{})
	s.holdOpen = release
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.eventually(func() bool { return s.opened.Load() == 1 }, "no session was asked for")

	s.edge.attending <- Attendance{Participant: alice, Joined: false}
	s.eventually(func() bool { return countOf[ParticipantLeft](s.reported()) == 1 },
		"the departure was never reported")
	close(release)

	s.eventually(func() bool { return s.ears.closes.Load() == 1 },
		"the session opened for a participant who left was never closed")
	s.False(s.listening(alice.ID), "a participant who left has a transcription session")
	s.Empty(s.ears.transcribed(), "a participant who left was transcribed")
}

func (s *AgentSuite) TestSomebodyWhoComesBackIsTranscribedAgain() {
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.edge.attending <- Attendance{Participant: alice, Joined: true}
	s.edge.attending <- Attendance{Participant: alice, Joined: false}
	s.edge.attending <- Attendance{Participant: alice, Joined: true}
	s.eventually(func() bool { return countOf[ParticipantJoined](s.reported()) == 2 },
		"the return was never reported")

	s.speak(alice)

	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 },
		"a participant who came back is not transcribed")
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
	s.Require().Len(request.Input, 1)
	s.Equal("hello", request.Input[0].Content)
	s.Equal(llm.User, request.Input[0].Role)

	s.eventually(func() bool { return len(s.edge.heard()) > 0 },
		"the reply was never published to the call")
}

func (s *AgentSuite) TestASpeculativeReplyStartsBeforeTheRulingAndIsSpokenOnlyAfterIt() {
	s.speculates = true
	s.join(true)
	ruling := make(chan struct{})
	s.flow.holdCreate = ruling
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 },
		"the reply should be asked for while the flow controller is still deciding")
	s.Len(s.flow.requests(), 1)
	edge := s.edge
	s.Never(func() bool { return len(edge.heard()) > 0 }, 150*time.Millisecond, 10*time.Millisecond,
		"nothing may be spoken before the ruling says to answer")
	s.Empty(s.agent.History(), "the words are not the conversation's until they are answered")

	close(ruling)

	s.eventually(func() bool { return len(s.edge.heard()) > 0 },
		"a ruling to answer should release the reply already started")
	s.Len(s.model.requests(), 1, "the reply started early is the one spoken, not a second one")
	s.eventually(func() bool { return len(s.agent.History()) == 2 }, "the exchange was never kept")
	s.Equal("hello", s.agent.History()[0].Content)
}

func (s *AgentSuite) TestASpeculativeReplyIsDroppedWhenTheRulingDoesNotAnswer() {
	s.speculates = true
	s.join(true)
	s.flow.reply = []string{`{"disposition":"ignore","floor":"continue"}`}
	ruling := make(chan struct{})
	s.flow.holdCreate = ruling
	participant := stt.Participant{ID: "child", Name: "Child"}
	s.speak(participant)

	s.says(participant, "mom where is my backpack")
	s.eventually(func() bool { return len(s.model.requests()) == 1 },
		"the reply should be asked for while the flow controller is still deciding")

	close(ruling)

	edge := s.edge
	s.Never(func() bool { return len(edge.heard()) > 0 }, 200*time.Millisecond, 10*time.Millisecond,
		"a reply to speech the agent was not meant to answer must never be heard")
	s.Empty(s.agent.History())
	s.Zero(countOf[Responding](s.reported()))
}

func (s *AgentSuite) TestAClarifyingRulingStartsAgainRatherThanUsingTheEarlyReply() {
	// The early reply was written without the note that says the words were ambiguous, so
	// speaking it would answer a question the caller may not have asked.
	s.speculates = true
	s.join(true)
	s.flow.reply = []string{`{"disposition":"clarify","floor":"continue"}`}
	ruling := make(chan struct{})
	s.flow.holdCreate = ruling
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "do it like last time")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the early reply never started")

	close(ruling)

	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"a clarifying ruling should ask for a reply of its own")
	s.Contains(s.model.requests()[1].Instructions, "ambiguous")
	s.NotContains(s.model.requests()[0].Instructions, "ambiguous")
}

func (s *AgentSuite) TestAUnifiedReplyRulesOnItsTurnWithoutTheFlowController() {
	s.unified = true
	s.join(true)
	s.model.reply = []string{"[respond] Hello", " there."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was never spoken")
	s.Empty(s.flow.requests(), "the reply rules on its own words, so nobody else is asked")
	s.Contains(s.model.requests()[0].Instructions, "[respond]")
	s.eventually(func() bool { return len(s.agent.History()) == 2 }, "the exchange was never kept")
	s.Equal("Hello there.", s.agent.History()[1].Content, "the tag is never part of what was said")
}

func (s *AgentSuite) TestAUnifiedReplyThatIgnoresItsTurnIsNeverHeard() {
	s.unified = true
	s.join(true)
	s.model.reply = []string{"[ignore]"}
	participant := stt.Participant{ID: "child"}
	s.speak(participant)

	s.says(participant, "mom where is my backpack")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the reply was never asked for")
	edge := s.edge
	s.Never(func() bool { return len(edge.heard()) > 0 }, 200*time.Millisecond, 10*time.Millisecond,
		"words the agent was not meant to answer must never be answered")
	s.Empty(s.agent.History())
}

func (s *AgentSuite) TestWithoutSpeculationTheReplyWaitsForTheRuling() {
	s.join(true)
	ruling := make(chan struct{})
	s.flow.holdCreate = ruling
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the flow controller was never asked")

	model := s.model
	s.Never(func() bool { return len(model.requests()) > 0 }, 150*time.Millisecond, 10*time.Millisecond,
		"without speculation the reply is only asked for once the ruling is in")
	close(ruling)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the reply never started")
}

func (s *AgentSuite) TestARevisionReplacesTheWordsBeforeCadenceActs() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.ears.emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeReplacement, Text: "hel"})
	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("hello", s.model.requests()[0].Input[0].Content)
}

func (s *AgentSuite) TestStablePartialSpeechIsAnsweredWithoutAFinalEvent() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.mutters(participant, "can you help me")

	s.eventually(func() bool { return len(s.model.requests()) == 1 },
		"cadence should act without waiting for a provider final event")
	s.Equal("can you help me", s.model.requests()[0].Input[0].Content)
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

func (s *AgentSuite) TestASecondVoiceAtTheSameMicrophoneIsNotTakenForTheCaller() {
	// One track, two people in the room. The track says who joined the call, so it names
	// the caller for both of them, and only a transcriber that hears the difference can
	// say that the second turn came from somebody else.
	s.join(true)
	s.flow.reply = []string{`{"disposition":"respond","floor":"continue"}`}
	s.flow.then = []string{`{"disposition":"ignore","floor":"continue"}`}
	participant := stt.Participant{ID: "alice", Name: "Alice"}
	s.speak(participant)

	s.saysInVoice(participant, "what time is my flight", "A")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 },
		"the caller's own turn was never considered")
	// The second voice waits for the caller's answer to be asked for: said while the first
	// ruling is still coming back, it is more of the caller's turn and supersedes that ruling.
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the caller was never answered")

	s.saysInVoice(participant, "mom where is my backpack", "B")
	s.eventually(func() bool { return len(s.flow.requests()) == 2 },
		"the second voice was never considered")

	asked := s.flow.requests()
	s.NotContains(asked[0].Input[0].Content, "different voice",
		"the first voice heard on a track is the caller's")
	s.Contains(asked[1].Input[0].Content, "different voice")
	s.eventually(func() bool {
		return slices.ContainsFunc(s.reported(), func(event Event) bool {
			decided, ok := event.(Decided)
			return ok && decided.Kind == string(ActIgnore)
		})
	}, "the second voice was never ruled on")
	s.Len(s.model.requests(), 1, "only the caller was answered")
}

func (s *AgentSuite) TestATranscriberThatCannotHearVoicesLeavesTheCallerAlone() {
	// Most transcribers say nothing about who is talking. A turn with no voice on it is
	// the caller's as far as anybody can tell, and calling it somebody else's would have
	// the agent stop answering the person it is on a call with.
	s.join(true)
	s.flow.reply = []string{`{"disposition":"respond","floor":"continue"}`}
	participant := stt.Participant{ID: "alice", Name: "Alice"}
	s.speak(participant)

	s.says(participant, "what time is my flight")

	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the turn was never considered")
	s.NotContains(s.flow.requests()[0].Input[0].Content, "different voice")
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

func (s *AgentSuite) TestAStreamingVoiceReceivesADecimalAsOneTextRequest() {
	s.join(true)
	s.model.reply = []string{"That comes to 12.", "63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is the total")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	requests := s.voice.spoken()
	s.Require().Len(requests, 2)
	s.Equal("That comes to 12.63.", requests[0].Text)
	s.False(requests[0].Final, "the sentence is a text delta in the streaming utterance")
	s.Empty(requests[1].Text)
	s.True(requests[1].Final, "the streaming utterance closes after the sentence")
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

func (s *AgentSuite) TestANonStreamingVoiceReceivesADecimalAsOneTextRequest() {
	s.join(false)
	s.model.reply = []string{"That comes to 12.", "63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is the total")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	requests := s.voice.spoken()
	s.Require().Len(requests, 1)
	s.Equal("That comes to 12.63.", requests[0].Text)
	s.True(requests[0].Final, "the sentence is one complete non-streaming request")
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
	s.Require().Len(second.Input, 3)
	s.Equal("hello", second.Input[0].Content)
	s.Equal(llm.Assistant, second.Input[1].Role)
	s.Equal("and again", second.Input[2].Content)
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

func (s *AgentSuite) TestAnIDSummarisedAwayIsStillOneAToolReturned() {
	// τ³ telecom: get_customer_by_phone returned C1001, a compaction summarised the result
	// away, and refuel_data with C1001 was refused as an ID no tool had given.
	tools := []string{"get_customer_by_phone", "get_details_by_id", "refuel_data"}
	looked := []llm.Message{
		{Role: llm.User, Content: "It's five five five dash one two three dash two zero zero two."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "get_customer_by_phone"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"customer_id": "C1001", "full_name": "John Smith"}`},
		{Role: llm.Assistant, Content: "I found your account, John."},
	}
	recent := []llm.Message{
		{Role: llm.User, Content: "Can you add two gigabytes to my line?"},
	}
	emitter := NewEmitter(eventBuffer)
	defer emitter.Close()
	agent := &Agent{
		emitter: emitter,
		history: append(append([]llm.Message(nil), looked...), recent...),
	}
	agent.applyCompaction(harness.Compacted{Prefix: looked, Summary: "John Smith asked for more data."})
	refuel := llm.ToolCall{Name: "refuel_data", Arguments: `{"customer_id":"C1001","line_id":"L1002","gb_amount":2}`}

	_, _, _, unfound := unfoundID(agent.History(), refuel, tools)
	s.True(unfound, "the model's history no longer holds the result")
	_, _, _, unfound = unfoundID(agent.record(), refuel, tools)
	s.False(unfound, "the call's record still does")

	summary := agent.History()[0]
	agent.applyCompaction(harness.Compacted{Prefix: agent.History()[:2], Summary: "Still John Smith."})
	s.Equal(append(append(append(append([]llm.Message(nil), looked...), summary), recent...), agent.History()...), agent.record(),
		"what each compaction took is kept once, in the order it happened, ahead of the history")
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
	s.Positive(turn.LLMTTFTMs, "the model's own time to first token")
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

func (s *AgentSuite) TestADirectionWithNothingToSayIsNeverSentToTheVoice() {
	// Gemma answered "[short pause] <ask skill=...>", and ElevenLabs refused the bare
	// "[short pause]" as empty text, which cost the call its voice for two seconds.
	s.performing = "You may write stage directions in square brackets."
	s.delegates()
	s.join(true)
	s.model.reply = []string{`[short pause] <ask skill="think">15% of 84.20</ask>`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")

	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")
	s.NotContains(said(s.voice.spoken()), "[short pause]")
}

func (s *AgentSuite) TestANameBothSidesHaveSaidIsExpectedFromThenOn() {
	// Flux heard "Chen" once, the agent said it back, and the caller's next "Yes, Chen" came
	// back as "Yes. Ten". Once both have said it, the transcriber is told to expect it.
	s.join(true)
	s.model.reply = []string{"Two at 7:30 for Chen. ", "I'm checking with Mia now."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "a table for two at seven thirty, name Chen")

	s.eventually(func() bool { return len(s.ears.keyterms()) > 0 }, "the transcriber was never retuned")
	terms := s.ears.keyterms()
	s.Contains(terms, "Chen")
	s.NotContains(terms, "Mia", "a name only the agent said is not reinforced")
	s.NotContains(terms, "Two", "a word that only opens a sentence is not a name")
	s.NotContains(terms, "I", "nor is a contraction")
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

	s.Equal(memory.Scope{
		AppID: "acme", UserID: "acme", AgentID: s.agentID, RunID: "session-1",
		Extra: map[string]string{"app_id": "router"},
	}, s.remembers.scopedTo(), "the app id is always the customer, and a caller's own only narrows it")
}

func (s *AgentSuite) TestAnAgentWithMemoryButNoSessionIsRefused() {
	_, err := New(Options{
		Text:       true,
		CustomerID: "acme",
		AgentID:    "agent-1",
		LLM:        s.reasoner(slog.New(slog.DiscardHandler)),
		Memory:     metered.NewMemory(&stubMemory{}, nil),
	})

	s.ErrorContains(err, "run id", "a memory written without its session could never be traced back to it")
}

func (s *AgentSuite) TestAnIncognitoSessionRecallsButRemembersNothing() {
	s.remembers = &stubMemory{knows: []memory.Memory{{Text: "Prefers to be called Al"}}}
	s.incognito = true
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "I moved to Austin")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.Require().NoError(s.agent.Close())
	s.Contains(s.model.requests()[0].Instructions, "Prefers to be called Al")
	s.Empty(s.remembers.remembered(), "an incognito session is not kept, in memory or anywhere else")
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
	s.Equal(memory.Scope{
		AppID: "acme", UserID: "acme", AgentID: s.agentID, RunID: "session-1",
		Extra: map[string]string{"app_id": "router"},
	}, s.remembers.writtenAs())
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

func (s *AgentSuite) TestLongThinkingWorkTellsTheCallerWhatIsStillGoing() {
	s.delegates()
	s.duplex = DuplexOptions{updateGaps: []time.Duration{10 * time.Millisecond}}
	s.join(true)
	s.model.reply = []string{`<ask skill="think">work through the itinerary</ask>`}
	s.model.composes = []string{"Still mapping out your itinerary."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "please work through the itinerary")

	s.eventually(func() bool { return s.agent.delegating() }, "the thinking task never started")
	s.eventually(func() bool { return s.spokenText("Still mapping out your itinerary.") },
		"a long thinking gap sounded like a dead call")
	composed := s.model.composeRequests()
	s.Require().Len(composed, 1)
	update := composed[0].Input
	s.Equal(fmt.Sprintf(composeNote, updatePurpose), update[len(update)-1].Content)
	s.True(slices.ContainsFunc(update, func(m llm.Message) bool {
		return m.Content == "please work through the itinerary"
	}), "the model writes the update knowing what the caller asked for")
	s.True(slices.ContainsFunc(s.agent.History(), func(m llm.Message) bool {
		return m.Content == "Still mapping out your itinerary."
	}), "the update is something the agent said, so the model can see it said it")
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

func (s *AgentSuite) TestWhatAResponseAskedForHoldsForTheAnswerToItsDelegatedWork() {
	// The reply carrying the subagent's findings is the answer that was asked for, so a
	// limit that stopped at the first reply would miss the one that matters.
	s.delegates()
	s.join(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	limit := 64

	_, err := s.agent.RespondTo(s.ctx, "what is 15% of 84.20", nil, options.LLM{MaxOutputTokens: &limit})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the answer coming back never started a turn")
	s.Equal(64, s.model.requests()[0].MaxOutputTokens)
	s.Equal(64, s.model.requests()[1].MaxOutputTokens)
}

func (s *AgentSuite) TestATypedTurnIsAskedOfTheModelInTheNameOfWhoeverSentIt() {
	// Whose day a reply is counted on is decided where the model is asked, so the sender
	// has to reach it, the reply that finishes the turn included.
	s.delegates()
	s.join(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	user := routing.Caller{UserID: "alice", IP: "203.0.113.7"}

	_, err := s.agent.RespondTo(routing.WithCaller(s.ctx, user), "what is 15% of 84.20", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the answer coming back never started a turn")
	s.Equal([]routing.Caller{user, user}, s.model.sentBy())
}

func (s *AgentSuite) TestATurnAScheduledTaskFiredIsAskedOfTheModelAsScheduled() {
	// A scheduled turn is the customer's spend, decided where the model is asked, so the
	// mark has to reach it, the reply that finishes the turn included, and the next turn
	// somebody sends is not marked.
	s.delegates()
	s.join(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`, "Hello."}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}

	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "what is 15% of 84.20", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 2 && !s.agent.floor().Talking },
		"the answer coming back never finished the turn")

	_, err = s.agent.RespondTo(s.ctx, "hello", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the next turn was never asked")
	s.Equal([]bool{true, true, false}, s.model.askedScheduled())
}

func (s *AgentSuite) TestATurnNobodyIsNamedForIsNotAskedInTheNameOfTheLastSender() {
	s.joinText()
	user := routing.Caller{UserID: "alice", IP: "203.0.113.7"}
	_, err := s.agent.RespondTo(routing.WithCaller(s.ctx, user), "hello", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.agent.History()) == 2 }, "the first turn was never answered")

	_, err = s.agent.RespondTo(s.ctx, "and again", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the second turn was never asked")
	s.Equal([]routing.Caller{user, {}}, s.model.sentBy())
}

func (s *AgentSuite) TestASpokenTurnIsNotAskedInTheNameOfWhoeverTypedBeforeIt() {
	s.join(false)
	user := routing.Caller{UserID: "alice", IP: "203.0.113.7"}
	_, err := s.agent.RespondTo(routing.WithCaller(s.ctx, user), "hello", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.agent.History()) == 2 && !s.agent.floor().Talking },
		"the typed turn was never answered")
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the spoken turn was never asked")
	s.Equal([]routing.Caller{user, {}}, s.model.sentBy())
}

func (s *AgentSuite) TestWhatFinishesASpokenTurnIsNotAskedInTheNameOfWhoeverTypedBeforeIt() {
	// A spoken turn is counted on whoever opened the session. One answered from a reply
	// started early still begins a turn of its own, so the tool reply that finishes it is
	// not the typist's.
	s.speculates = true
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Hello."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	s.model.callsOn = 2
	user := routing.Caller{UserID: "alice", IP: "203.0.113.7"}
	_, err := s.agent.RespondTo(routing.WithCaller(s.ctx, user), "hello", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.agent.History()) == 2 && !s.agent.floor().Talking },
		"the typed turn was never answered")
	ruling := make(chan struct{})
	s.flow.holdCreate = ruling
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the reply should be asked for while the flow controller is still deciding")
	close(ruling)

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the tool result was never answered")
	s.Equal([]routing.Caller{user, {}, {}}, s.model.sentBy())
}

func (s *AgentSuite) TestASpokenTurnIsNotAskedAsScheduledWhenAScheduledTurnCameBeforeIt() {
	// A spoken turn is the caller's, not the customer's scheduled spend, so the mark of the
	// scheduled turn before it does not carry over.
	s.join(false)
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "hello", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.agent.History()) == 2 && !s.agent.floor().Talking },
		"the scheduled turn was never answered")
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the spoken turn was never asked")
	s.Equal([]bool{true, false}, s.model.askedScheduled())
}

func (s *AgentSuite) TestWhatFinishesASpokenTurnIsNotAskedAsScheduledWhenAScheduledTurnCameBeforeIt() {
	// A spoken turn answered from a reply started early begins a turn of its own, so the
	// tool reply that finishes it does not carry the mark of the scheduled turn before it.
	s.speculates = true
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Hello."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	s.model.callsOn = 2
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "hello", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.agent.History()) == 2 && !s.agent.floor().Talking },
		"the scheduled turn was never answered")
	ruling := make(chan struct{})
	s.flow.holdCreate = ruling
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the reply should be asked for while the flow controller is still deciding")
	close(ruling)

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the tool result was never answered")
	s.Equal([]bool{true, false, false}, s.model.askedScheduled())
}

func (s *AgentSuite) TestAReasoningEffortTheModelDoesNotTakeIsRefusedBeforeTheTurn() {
	s.join(true)

	_, err := s.agent.RespondTo(s.ctx, "where is my order", nil, options.LLM{ReasoningEffort: "high"})

	s.ErrorContains(err, "does not reason")
	s.Empty(s.model.requests())
	s.Zero(countOf[Responding](s.reported()))
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

	s.model.writes(turnA, `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")

	taskID := s.subagent.requests()[0].ID
	s.subagent.writes(taskID, "It is 12.63.")
	s.subagent.finishes(taskID)
	s.eventually(func() bool { return countOf[TaskSettled](s.reported()) == 1 }, "the task never settled")
	s.Equal(1, len(s.model.requests()),
		"an answer coming back must not take the floor from a reply still being written")

	s.model.writes(turnA, "Let me check. Won't be a moment.")
	s.model.finishes(turnA)
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
	model := s.model
	s.Require().Never(func() bool { return len(model.requests()) > 1 },
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
	model := s.model
	s.Require().Never(func() bool { return len(model.requests()) > 1 },
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

func (s *AgentSuite) TestWorkTheCallerWithdrewIsDroppedByTheModel() {
	s.delegates()
	s.join(true)
	s.model.reply = []string{`<ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{`<drop skill="think"/>Okay, 20 percent then.`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return s.agent.delegating() }, "the task never started")

	s.says(participant, "actually, use 20 percent")

	s.eventually(func() bool { return countOf[TaskCancelled](s.reported()) == 1 },
		"work based on the old premise was not cancelled")
	cancelled, _ := firstOf[TaskCancelled](s.reported())
	s.Equal(harness.ReasonDropped, cancelled.Reason)
	s.Equal(1, s.subagent.interrupted())
	s.model.mu.Lock()
	defer s.model.mu.Unlock()
	s.Contains(s.model.asked[len(s.model.asked)-1].Instructions, "still working on the think",
		"the model cannot drop work it was never told is running")
}

func (s *AgentSuite) TestAQuestionAboutWorkStillRunningLeavesItRunning() {
	s.delegates()
	s.join(true)
	s.model.reply = []string{`One moment. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"Still on it."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return s.agent.delegating() }, "the task never started")

	s.says(participant, "is it done yet?")

	s.eventually(func() bool { return s.spokenText("Still on it.") }, "the question was not answered")
	s.True(s.agent.delegating(), "asking how it is going is not withdrawing the request")
	s.Zero(countOf[TaskCancelled](s.reported()))
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
	s.model.composes = []string{"Anything else on the booking?"}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.agent.checkIn(participant, idlePurpose)

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the question was never reported as speech")
	history := s.agent.History()
	s.Require().Len(history, 1)
	s.Equal(llm.Assistant, history[0].Role)
	s.Equal("Anything else on the booking?", history[0].Content)
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the question never reached the call")
	composed := s.model.composeRequests()
	s.Require().Len(composed, 1)
	s.Equal(fmt.Sprintf(composeNote, idlePurpose), composed[0].Input[len(composed[0].Input)-1].Content)
	s.Empty(composed[0].Tools, "a line to say is written, not acted on")
	s.Empty(s.model.requests(), "asking for a line is not a turn")
}

func (s *AgentSuite) TestALineWrittenAfterTheCallerSpokeIsNotSaid() {
	s.join(true)
	written := make(chan struct{})
	s.model.holdCompose = written
	s.model.composes = []string{"Anything else on the booking?"}
	s.model.reply = []string{"Sure, what is it?"}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.agent.checkIn(participant, idlePurpose)
	s.eventually(func() bool { return len(s.model.composeRequests()) == 1 }, "no line was asked for")
	s.says(participant, "actually, one more thing")
	s.eventually(func() bool { return s.spokenText("what is it") }, "the caller was never answered")
	close(written)

	s.eventually(func() bool { return !s.agent.floor().Composing }, "the line was never settled")
	s.False(s.spokenText("Anything else on the booking?"), "the agent asked over a caller who had moved on")
	s.False(slices.ContainsFunc(s.agent.History(), func(m llm.Message) bool {
		return m.Content == "Anything else on the booking?"
	}), "a line nobody heard is in the history")
}

func (s *AgentSuite) TestALineThatCannotBeWrittenSaysNothingAndFailsNothing() {
	s.join(true)
	s.model.composeFails = errors.New("rate limited")
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.agent.checkIn(participant, idlePurpose)

	s.eventually(func() bool { return len(s.model.composeRequests()) == 1 }, "no line was asked for")
	s.eventually(func() bool { return !s.agent.floor().Composing }, "the failure was never settled")
	s.Empty(s.voice.spoken(), "a line that was never written was said anyway")
	s.Zero(countOf[Error](s.reported()), "a missing status line failed the turn the caller is waiting on")
	s.Empty(s.agent.History())
}

func (s *AgentSuite) TestALineCarryingASkillTagIsNotReadOut() {
	// A line is spoken as written, past the filter that acts on a reply's tags, so one
	// that asks for work would have the caller hear the request itself.
	s.join(true)
	s.model.composes = []string{`Let me look. <ask skill="think">the booking</ask>`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.agent.checkIn(participant, idlePurpose)

	s.eventually(func() bool { return len(s.model.composeRequests()) == 1 }, "no line was asked for")
	s.eventually(func() bool { return !s.agent.floor().Composing }, "the line was never settled")
	s.Empty(s.voice.spoken(), "the caller heard a skill tag read out")
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

func (s *AgentSuite) TestATranscriberTheVendorRefusesIsTakenOverByTheNextModel() {
	s.backup = newStubSTT()
	s.logs = &logLines{}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.eventually(func() bool { return s.opened.Load() == 1 }, "nobody started listening")
	finals := testutil.ToFloat64(sttFinals.WithLabelValues("backup"))

	// What ElevenLabs sends an account that has not accepted its terms (probed 2026-10-19).
	s.ears.emitter.Send(stt.Error{
		Provider: "stub", Model: "stub-model", Context: "unaccepted_terms", Fatal: true,
		Err: errors.New("You have not accepted the terms required to use this feature."),
	})
	s.eventually(func() bool {
		return len(s.logs.records("the transcriber refused the session, the next model took it over")) == 1
	}, "the agent said nothing of the next model taking over")
	s.backup.emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeFinal, Text: "book a table"})

	s.eventually(func() bool {
		return testutil.ToFloat64(sttFinals.WithLabelValues("backup")) == finals+1
	}, "the next model's words were not heard")
	line := s.logs.records("the transcriber refused the session, the next model took it over")[0]
	s.Equal("stub/stub-model", line["from"])
	s.Equal("backup", line["provider"])
	s.EqualValues(1, s.opened.Load(), "the session was dropped and reopened rather than carried on")
	for _, event := range s.events.seen() {
		if failure, ok := event.(Error); ok {
			s.NotEqual("stt", failure.Context, "a refusal the next model took over reached the caller")
		}
	}
}

func (s *AgentSuite) TestAnAgentPinnedToABatchTranscriberIsRefusedOnceWhenItJoins() {
	// On 6d4d0d4a the agent joined and every inbound frame opened a transcriber that was
	// refused: 100 frames gave 100 "stt" errors (rev1-probes/agent_head_probe.go.txt).
	agent, built := s.buildTranscribing("stub/batch", nil)

	err := agent.Join(s.ctx)

	s.Require().ErrorIs(err, routing.ErrRequest, "the caller's to change")
	s.EqualError(err, "agent: start stt: routing: stub/batch is a batch model for recordings, not live calls; "+
		"for a live call name stub/stub-model")
	s.Zero(built.Load(), "nothing was built, so no vendor was dialled")
	s.False(s.edge.joined, "the call was never joined, so no frame is heard and refused")
}

func (s *AgentSuite) TestAnAgentWhoseLiveTranscriberFailsToStartTriesAgainOnEachFrame() {
	// On d1654d42 each frame opened a transcriber, and each failure was one "stt" error.
	agent, built := s.buildTranscribing("stub/stub-model", errors.New("stub: connect failed"))
	s.Require().NoError(agent.Join(s.ctx))
	events := collect(agent)
	s.T().Cleanup(func() { <-events.done })
	s.T().Cleanup(func() { _ = agent.Close() })

	for range 3 {
		s.speak(stt.Participant{ID: "alice"})
	}

	failed := func() int {
		count := 0
		for _, event := range events.seen() {
			if failure, ok := event.(Error); ok && failure.Context == "stt" {
				count++
			}
		}
		return count
	}
	s.eventually(func() bool { return failed() == 3 }, "a failed start was not tried again on the next frame")
	s.EqualValues(3, built.Load())
}

// buildTranscribing builds an agent whose transcriber is pinned to target, over a config
// that also declares the batch model stub/batch. A start error fails every session opened.
func (s *AgentSuite) buildTranscribing(target string, start error) (*Agent, *atomic.Int64) {
	s.edge = newLoopbackEdge()
	s.model = newStubLLM()
	s.flow = newStubLLM()
	s.voice = newStubTTS(true)
	logger := slog.New(slog.DiscardHandler)

	config := stubConfig()
	config.Providers = append(config.Providers, routing.ProviderConfig{Provider: "stub", Model: "batch", Languages: []string{"en"}})
	var built atomic.Int64
	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) {
		built.Add(1)
		return &unstartableSTT{stubSTT: newStubSTT(), err: start}, nil
	})
	transcriber, err := sttrouter.New(sttrouter.Options{Config: config, Registry: transcription, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)
	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return s.voice, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: speech, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	agent, err := New(Options{
		Edge: s.edge, Instructions: "be brief", CustomerID: "acme",
		LLM: s.reasoner(logger), LLMTarget: "en-low-latency",
		STT: transcriber, STTTarget: target,
		TTS: speaker, TTSTarget: "en-low-latency", Logger: logger,
	})
	s.Require().NoError(err)
	return agent, &built
}

// unstartableSTT is a transcriber whose Start fails with err, when it is set.
type unstartableSTT struct {
	*stubSTT
	err error
}

func (u *unstartableSTT) Start(context.Context) error { return u.err }

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

func (s *AgentSuite) TestAReplyWaitingOnHeadersDoesNotBlockTheFloor() {
	s.join(true)
	s.model.reply = nil
	hold := make(chan struct{})
	s.model.holdCreate = hold
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	s.flow.then = []string{`{"disposition":"wait","floor":"continue"}`}

	s.mutters(participant, "okay")

	s.eventually(func() bool { return len(s.flow.requests()) >= 2 },
		"the overlap must be asked about while the reply is still waiting for headers")
	close(hold)
}

func (s *AgentSuite) TestBargeInCancelsAReplyThatIsStillOpening() {
	s.join(true)
	s.model.reply = nil
	hold := make(chan struct{})
	s.model.holdCreate = hold
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}

	s.mutters(participant, "wait stop")

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the hung create must not keep the floor")
	s.Empty(s.voice.spoken(), "nothing was spoken from a create that never returned")
}

func (s *AgentSuite) TestRelatedOverlapShortensThenAnswersTheAddition() {
	s.join(true)
	s.model.reply = nil
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	s.heardSomeOf(s.model.requests()[0].ID)
	s.flow.then = []string{`{"disposition":"respond","floor":"shorten"}`}
	shortened := testutil.ToFloat64(settledOverAgent.WithLabelValues("4+", "true", "shorten"))

	s.says(participant, "only the vegetarian options")

	s.eventually(func() bool { return countOf[OverlapDecided](s.reported()) == 1 },
		"the overlap was never decided")
	overlap, _ := firstOf[OverlapDecided](s.reported())
	s.Equal(string(harness.Shorten), overlap.Action)
	s.eventually(func() bool { return s.model.interrupted() == 1 }, "the long answer kept generating")
	s.Zero(s.voice.interrupted(), "speech already sent should finish rather than stop abruptly")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the addition was never answered")
	s.Equal(1.0, testutil.ToFloat64(settledOverAgent.WithLabelValues("4+", "true", "shorten"))-shortened)
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

	s.model.finishes(first)
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the queued turn was not answered after the current reply")
}

func (s *AgentSuite) TestTalkingOverTheAgentStopsItBeforeTheWordsSettle() {
	// The caller talks over a long reply and changes direction. The agent stops on the
	// in-progress words rather than waiting for them to settle first.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.model.reply = nil
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, "no turn was started")

	s.mutters(participant, "actually wait")

	s.eventually(func() bool { return s.voice.interrupted() == 1 && s.model.interrupted() == 1 },
		"talking over the agent should stop it before the words settle")
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the interruption was never reported")

	s.says(participant, "actually wait, make it six")
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the settled words were never answered")
	s.Equal(1, countOf[Interrupted](s.reported()),
		"the settled words follow the stop, they do not interrupt again")
}

func (s *AgentSuite) TestAnInterruptionKeepsReturnedWorkUntilTheVoiceReconnects() {
	s.delegates()
	s.join(true)
	s.model.reply = nil
	s.model.then = []string{"The result is twelve point six three."}
	s.subagent.reply = nil
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool {
		requests := s.model.requests()
		return len(requests) > 0 && s.model.script(requests[0].ID) != nil
	}, "the model response stream never opened")
	turnID := s.model.requests()[0].ID

	s.model.writes(turnID, "One moment, checking.")
	s.eventually(func() bool { return s.spokenText("One moment, checking.") }, "the reply never reached the voice")
	s.model.writes(turnID, `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool {
		requests := s.subagent.requests()
		return len(requests) > 0 && s.subagent.script(requests[0].ID) != nil
	}, "the subagent response stream never opened")
	s.Require().NotEmpty(s.subagent.requests())
	taskID := s.subagent.requests()[0].ID
	s.subagent.writes(taskID, "It is 12.63.")
	s.subagent.finishes(taskID)
	s.eventually(func() bool { return countOf[TaskSettled](s.reported()) == 1 }, "the task never settled")
	s.True(s.agent.harness.Pending(), "the result is still owed while the first reply is open")

	started := make(chan struct{})
	release := make(chan struct{})
	s.voice.mu.Lock()
	s.voice.interruptStarted = started
	s.voice.interruptRelease = release
	s.voice.mu.Unlock()
	var releaseOnce sync.Once
	openVoice := func() { releaseOnce.Do(func() { close(release) }) }
	defer openVoice()

	s.says(participant, "actually wait")
	s.eventually(func() bool {
		select {
		case <-started:
			return true
		default:
			return false
		}
	}, "the voice interruption never reached the reconnect gate")
	s.eventually(func() bool {
		if countOf[Error](s.reported()) > 0 {
			return true
		}
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.utterances == 0
	}, "the interrupted utterance did not settle")
	s.eventually(func() bool { return !s.agent.converse.Listening() },
		"the settled caller turn still held the floor")
	// Exercise the same admission path used when presence notices the returned result.
	s.NoError(s.agent.follow())
	s.never(func() bool {
		return countOf[Error](s.reported()) > 0
	}, "a reply tried to use the voice before reconnect completed")

	openVoice()
	s.eventually(func() bool {
		for _, request := range s.model.requests() {
			if !strings.Contains(request.Instructions, "It is 12.63.") {
				continue
			}
			if slices.ContainsFunc(request.Input, func(message llm.Message) bool {
				return message.Content == "actually wait"
			}) {
				return true
			}
		}
		return false
	}, "the replacement model request did not receive the delegated result")
	s.eventually(func() bool { return s.spokenText("twelve point six three") }, "the returned result was not spoken")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the replacement reply produced no audio")
	s.eventually(func() bool { return !s.agent.Busy() }, "the replacement reply never settled")
	s.Zero(countOf[Error](s.reported()), "the interruption left the agent with a voice error")
}

func (s *AgentSuite) TestASettledCallerCorrectionWaitsForTheReplacementVoice() {
	s.join(true)
	s.model.reply = []string{"The old answer is still being spoken."}
	s.model.then = []string{"The corrected answer is ready."}
	s.voice.silent = true
	s.replacement = newStubTTS(true)
	participant := stt.Participant{ID: "alice"}

	started := make(chan struct{})
	release := make(chan struct{})
	s.replacement.mu.Lock()
	s.replacement.startStarted = started
	s.replacement.startRelease = release
	s.replacement.mu.Unlock()
	var releaseOnce sync.Once
	openReplacement := func() { releaseOnce.Do(func() { close(release) }) }
	defer openReplacement()

	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool {
		requests := s.model.requests()
		return len(requests) > 0 && s.model.script(requests[0].ID) != nil
	}, "the first model response stream never opened")
	s.eventually(func() bool {
		return countOf[Responded](s.reported()) == 1 && len(s.voice.spoken()) > 0 && s.agent.Busy()
	}, "the first reply did not leave speech in flight")

	s.voice.mu.Lock()
	s.voice.unavailable = true
	s.voice.mu.Unlock()
	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Fatal: true, Err: errors.New("socket closed"),
	})
	s.eventually(func() bool {
		select {
		case <-started:
			return true
		default:
			return false
		}
	}, "the replacement voice never reached its start gate")
	s.eventually(func() bool { return countOf[Error](s.reported()) == 1 },
		"the old voice failure was not reported")
	errorsBeforeCorrection := countOf[Error](s.reported())

	callerText := "Actually, make that a short poem."
	s.says(participant, callerText)
	s.eventually(func() bool {
		for _, event := range s.reported() {
			if decided, ok := event.(Decided); ok &&
				(decided.Kind == string(ActAnswer) || decided.Kind == string(ActQueue)) && decided.Text == callerText {
				return true
			}
		}
		return false
	}, "the flow controller never ruled on the settled correction")
	// The correction takes the floor during recovery, then remains queued until speech is possible.
	s.eventually(func() bool {
		return countOf[Interrupted](s.reported()) > 0 ||
			countOf[Responding](s.reported()) > 1 || countOf[Error](s.reported()) > 1
	}, "the caller correction was neither interrupted nor admitted")
	s.never(func() bool {
		return countOf[Responding](s.reported()) > 1 || countOf[Error](s.reported()) > errorsBeforeCorrection
	}, "the caller correction was admitted before the replacement voice started")
	s.Require().Equal(1, countOf[Interrupted](s.reported()), "the correction must take the floor during recovery")
	s.Equal(1, countOf[Responding](s.reported()), "the caller answer must wait for reconnection")
	s.Equal(errorsBeforeCorrection, countOf[Error](s.reported()), "the caller answer must not use the lost voice")

	openReplacement()
	var answerTurnID string
	s.eventually(func() bool {
		for _, request := range s.model.requests() {
			if slices.ContainsFunc(request.Input, func(message llm.Message) bool {
				return message.Content == callerText
			}) {
				answerTurnID = request.ID
				return true
			}
		}
		return false
	}, "the replacement model request did not receive the queued correction")
	s.eventually(func() bool {
		return slices.ContainsFunc(s.replacement.spoken(), func(request tts.Request) bool {
			return strings.Contains(request.Text, "corrected answer is ready")
		})
	},
		"the corrected answer was not spoken by the replacement voice")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the corrected answer produced no audio")
	s.eventually(func() bool {
		for _, event := range s.reported() {
			if spoke, ok := event.(Spoke); ok && spoke.TurnID == answerTurnID {
				return true
			}
		}
		return false
	},
		"the corrected answer did not report completed speech")
	s.eventually(func() bool { return !s.agent.Busy() }, "the corrected answer never settled")
	s.Equal(2, countOf[Heard](s.reported()), "the queued caller correction was not duplicated")
}

func (s *AgentSuite) TestAnInterruptionStillCountsAsBusyBeforeTheFirstSentence() {
	s.join(true)
	s.model.reply = nil
	s.model.then = []string{"Understood."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool {
		requests := s.model.requests()
		return len(requests) > 0 && s.model.script(requests[0].ID) != nil
	}, "the model response stream never opened")
	s.Empty(s.voice.spoken(), "the reply has not written its first sentence")

	started := make(chan struct{})
	release := make(chan struct{})
	s.voice.mu.Lock()
	s.voice.interruptStarted = started
	s.voice.interruptRelease = release
	s.voice.mu.Unlock()
	var releaseOnce sync.Once
	openVoice := func() { releaseOnce.Do(func() { close(release) }) }
	defer openVoice()

	s.says(participant, "actually wait")
	s.eventually(func() bool {
		select {
		case <-started:
			return true
		default:
			return false
		}
	}, "the voice interruption never reached the reconnect gate")
	s.True(s.agent.Busy(), "an interruption with no open utterance still has cleanup to finish")

	openVoice()
	s.eventually(func() bool { return s.spokenText("Understood.") }, "the caller's reply was not spoken")
	s.eventually(func() bool { return !s.agent.Busy() }, "the caller's reply never settled")
	s.Zero(countOf[Error](s.reported()), "the interruption left the agent with an error")
}

func (s *AgentSuite) TestAnAcknowledgementInProgressDoesNotStopTheAgent() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.model.reply = nil
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	s.flow.then = []string{`{"disposition":"wait","floor":"continue"}`}

	s.mutters(participant, "okay")

	s.eventually(func() bool { return len(s.flow.requests()) >= 2 }, "the overlap was never considered")
	s.never(func() bool { return s.model.interrupted() > 0 || s.voice.interrupted() > 0 },
		"an acknowledgement is not a reason to stop")
	s.Len(s.model.requests(), 1, "the acknowledgement does not earn its own reply while the agent talks")
}

func (s *AgentSuite) TestTheControllerIsToldWhatTheAgentIsSaying() {
	// A ruling on partial words overlapping the reply has to see the reply, or it cannot
	// tell a correction from the caller's line echoing the agent back.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.model.reply = nil
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first reply never started")
	turnA := s.model.requests()[0].ID
	s.model.writes(turnA, "The menu has three courses")
	s.eventually(func() bool { return countOf[ResponseDelta](s.reported()) >= 1 }, "the reply never streamed")
	s.heardSomeOf(turnA)
	s.flow.then = []string{`{"disposition":"wait","floor":"continue"}`}

	s.mutters(participant, "actually wait")

	s.eventually(func() bool { return len(s.flow.requests()) >= 2 }, "the overlap was never considered")
	overlap := s.flow.requests()[len(s.flow.requests())-1]
	s.Contains(overlap.Input[0].Content, "The menu has three courses",
		"the controller judges the overlap against what the agent has said so far")
}

func (s *AgentSuite) TestBargeInStopsTheModelAndTheVoice() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	// A reply that is still being written is the only one there is anything to stop.
	s.model.reply = nil
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

func (s *AgentSuite) TestAQuietPhraseIsRaisedBeforeTheCallHearsIt() {
	// An expressive voice can read a phrase 20 dB below the sentence before it. The caller
	// hears it raised towards the rest, so the agent does not seem to trail off or finish.
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	turnID := s.model.requests()[0].ID

	sine := func(ms int, amplitude float64) []int16 {
		samples := make([]int16, 16*ms)
		for i := range samples {
			samples[i] = int16(amplitude * 32767 * math.Sin(2*math.Pi*220*float64(i)/16_000))
		}
		return samples
	}
	quiet := append(sine(1000, 0.1), sine(1500, 0.01)...)
	for start := 0; start < len(quiet); start += 320 {
		s.voice.emitter.Send(tts.AudioChunk{
			SynthesisID: turnID,
			Audio:       audio.PcmData{Samples: quiet[start:min(start+320, len(quiet))], SampleRate: 16_000, Channels: 1},
		})
	}

	s.eventually(func() bool { return len(s.edge.heard()) == len(quiet)/320 }, "the reply was never published")
	var last []int16
	for _, chunk := range s.edge.heard()[len(quiet)/320-25:] {
		last = append(last, chunk.Samples...)
	}
	var sum float64
	for _, sample := range last {
		sum += float64(sample) * float64(sample)
	}
	rms := math.Sqrt(sum/float64(len(last))) / 32767
	s.Greater(rms, 0.01/math.Sqrt2*5, "the end of the quiet phrase reached the call at least 14 dB louder")
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

func (s *AgentSuite) TestBargeInDropsQueuedSpeechBeforeTheVoiceInterruptReturns() {
	s.join(true)
	s.edge.holdSpeech(true)
	s.model.reply = nil
	s.model.then = []string{"I heard you. I will stop."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool {
		requests := s.model.requests()
		return len(requests) == 1 && s.model.script(requests[0].ID) != nil
	}, "the first reply never started")
	turnID := s.model.requests()[0].ID
	s.model.writes(turnID, "The old reply is still in flight.")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the old reply never reached the edge")

	started := make(chan struct{})
	release := make(chan struct{})
	s.voice.mu.Lock()
	s.voice.interruptStarted = started
	s.voice.interruptRelease = release
	s.voice.mu.Unlock()
	var releaseOnce sync.Once
	openVoice := func() { releaseOnce.Do(func() { close(release) }) }
	defer openVoice()

	s.says(participant, "actually, stop")
	select {
	case <-started:
	case <-time.After(settleFor):
		s.FailNow("the voice interruption never reached its blocking gate")
	}

	s.False(s.edge.SpeechPending(), "queued speech must be dropped before a blocking voice interruption returns")
	s.Empty(s.edge.heard(), "audio queued before the interruption must be removed")
	model := s.model
	s.Require().Never(func() bool { return len(model.requests()) > 1 },
		100*time.Millisecond, 5*time.Millisecond, "the replacement reply must wait for the voice interruption")

	s.synthesises(turnID)
	edge := s.edge
	s.Require().Never(func() bool { return len(edge.heard()) > 0 },
		100*time.Millisecond, 5*time.Millisecond, "late audio from the abandoned reply must stay unheard")

	openVoice()
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the interrupted reply was not reported")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the caller's reply did not resume after interruption")
	secondTurnID := s.model.requests()[1].ID
	s.eventually(func() bool {
		var text strings.Builder
		for _, request := range s.voice.spoken() {
			if request.ID == secondTurnID {
				text.WriteString(request.Text)
			}
		}
		return strings.Contains(text.String(), "I heard you.") && strings.Contains(text.String(), "I will stop.")
	}, "the queued caller reply was not spoken")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the queued caller reply produced no audio")
	s.eventually(func() bool { return countOf[Responding](s.reported()) >= 2 }, "the replacement reply never reported starting")
	s.Equal(1, countOf[Interrupted](s.reported()), "the old reply is interrupted once")
	s.Equal(2, countOf[Responding](s.reported()), "the replacement reply starts once after interruption")
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

func (s *AgentSuite) TestASecondToolDoesNotStartACompetingReply() {
	// Two tools in one reply each used to call respondAfterTool. The second stole
	// speakingTurn, so the first result's generate was dropped unspoken.
	s.ownsTools("both orders found")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"Both came back."}
	s.model.calls = []llm.ToolCall{
		{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"1"}`},
		{ID: "call-2", Name: "lookup_order", Arguments: `{"order":"2"}`},
	}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where are my orders")

	s.eventually(func() bool { return len(s.runner.asked()) == 2 }, "both tools should run")
	s.eventually(func() bool { return s.spokenText("Both came back") },
		"the caller was left in silence after the second tool stole the floor")
	s.Len(s.model.requests(), 2, "a second tool must not start a competing generate")
}

func (s *AgentSuite) TestAPromiseToCheckThatCalledNothingIsKept() {
	// Gemma told Voicebench callers "one moment, checking..." and stopped there, so the
	// caller waited on a lookup that was never made.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Order twelve, one moment, checking..."}
	s.model.then = []string{}
	s.model.callsOn = 2
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is order twelve")

	s.eventually(func() bool { return len(s.runner.asked()) == 1 }, "the promised lookup was never made")
	kept := s.model.requests()[1]
	s.True(strings.HasPrefix(kept.ID, toolPrefix))
	s.Contains(kept.Instructions, keepPromise)
	s.Empty(kept.ToolChoice, "a hold phrase may come before what the call needs is known")
}

func (s *AgentSuite) TestAClaimThatSomethingIsDoneWithNoCallMustCallATool() {
	// Chased up with a note alone, Gemma told a Voicebench caller "You're all set!" and then
	// "Is there anything else?", and the booking was never made.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Order twelve, you're all set."}
	s.model.then = []string{}
	s.model.callsOn = 2
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is order twelve")

	s.eventually(func() bool { return len(s.runner.asked()) == 1 }, "the claimed action was never made")
	s.Equal("required", s.model.requests()[1].ToolChoice)
}

func (s *AgentSuite) TestAForcedChaseThatWritesNothingIsAskedAgainUnforced() {
	// Constrained to a call, Gemma on vLLM wrote only whitespace and the caller was left
	// with a booking that was never made.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Order twelve, you're all set."}
	s.model.then = []string{" "}
	s.model.callsOn = 3
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is order twelve")

	s.eventually(func() bool { return len(s.runner.asked()) == 1 }, "the empty forced chase was not asked again")
	s.Equal("required", s.model.requests()[1].ToolChoice)
	s.Empty(s.model.requests()[2].ToolChoice)
}

func (s *AgentSuite) TestAGoAheadAnsweredWithAQuestionIsChasedUp() {
	// After "Yes, that's right. Please go ahead." Gemma asked Voicebench callers for a phone
	// number instead of booking, and the call ended with nothing booked.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Shall I look up order twelve?"}
	s.model.then = []string{"Could I get a phone number first?"}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is order twelve")
	s.eventually(func() bool { return s.spokenText("Shall I look up") }, "the first reply was not spoken")

	s.says(participant, "yes, please go ahead")

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the go-ahead was left unacted on")
	s.Contains(s.model.requests()[2].Instructions, actOnGoAhead)
}

func (s *AgentSuite) TestAPromiseHandedToTheSubagentIsNotChasedUp() {
	// Gemma met "please go ahead" with "<ask skill=...>", and asking it again for a call
	// superseded the work it had just handed over.
	s.delegates()
	s.ownsTools("order 12 ships tomorrow")
	held := make(chan struct{})
	defer close(held)
	s.join(false)
	s.subagent.holdCreate = held
	s.model.reply = []string{`One moment, checking. <ask skill="think">where order twelve is</ask>`}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "yes, please go ahead")

	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")
	s.never(func() bool { return len(s.model.requests()) > 1 }, "the handed-over work was chased up")
}

func (s *AgentSuite) TestARepeatedCallIsAnsweredFromTheFirstInsteadOfRunAgain() {
	// After a result, Gemma often asked for the same call again: a second dispatch, a second
	// booking.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{}
	s.model.keepCalling = true
	s.asksFor("lookup_order", `{"order": "12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is order twelve")

	s.eventually(func() bool {
		return slices.ContainsFunc(s.history(), func(m llm.Message) bool {
			return m.Role == llm.ToolResult && strings.Contains(m.Content, "order 12 ships tomorrow") &&
				strings.HasPrefix(m.Content, alreadyCalled)
		})
	}, "the repeated call was not answered from the first")
	s.Len(s.runner.asked(), 1, "the same call ran twice")
}

func (s *AgentSuite) TestABookingMadeBeforeTheCallerSpokeAgainIsNotMadeTwice() {
	booking := llm.ToolCall{ID: "c1", Name: "create_reservation", Arguments: `{"name":"Alvarez","time":"7:30"}`}
	history := []llm.Message{
		{Role: llm.User, Content: "book it for Alvarez at 7:30"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{booking}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"name":"Alvarez","time":"7:30"}`},
		{Role: llm.User, Content: "yes, please go ahead"},
	}
	again := llm.ToolCall{ID: "c2", Name: "create_reservation", Arguments: `{"time": "7:30", "name": "Alvarez"}`}
	earlier, repeated := answeredSinceCaller(history, again)
	s.True(repeated)
	s.Contains(earlier, "Alvarez")

	_, repeated = answeredSinceCaller(history, llm.ToolCall{ID: "c3", Name: "check_availability", Arguments: booking.Arguments})
	s.False(repeated, "a lookup is free to run again once the caller has spoken")

	history[2].Content = `{"error":"no table at 7:30"}`
	_, repeated = answeredSinceCaller(history, again)
	s.False(repeated, "a booking that failed may be tried again")
}

func (s *AgentSuite) TestAVerificationRefusedOnceIsNotSentTheSameValuesAgain() {
	verify := llm.ToolCall{ID: "c1", Name: "verify_account", Arguments: `{"pin":"1044"}`}
	history := []llm.Message{
		{Role: llm.User, Content: "PIN four four seven one"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{verify}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"ok":false}`},
		{Role: llm.User, Content: "I said four four seven one"},
	}
	refused, again := refusedBefore(history, llm.ToolCall{ID: "c2", Name: "verify_account", Arguments: `{"pin": "1044"}`})
	s.True(again)
	s.Equal(`{"ok":false}`, refused)
	s.True(failedResult(alreadyRefused + refused + tryOtherValues))

	_, again = refusedBefore(history, llm.ToolCall{ID: "c3", Name: "verify_account", Arguments: `{"pin":"4471"}`})
	s.False(again, "new values are worth trying")

	history[2].Content = didNotWork + "timeout"
	_, again = refusedBefore(history, llm.ToolCall{ID: "c4", Name: "verify_account", Arguments: `{"pin":"1044"}`})
	s.False(again, "a call that could not be made may be tried again")
}

func (s *AgentSuite) TestASentenceAboutTheCallerIsNotSaidToThem() {
	s.True(narrates("The caller said yes after I offered two times."))
	s.True(narrates("The caller is listening, not reading: tell them the part that answers them."))
	s.True(narrates("[respond] The user wants a table."))
	s.False(narrates("The callback number I have is 512 555 0142."))
	s.False(narrates("Your table is booked."))
}

func (s *AgentSuite) TestAWithheldClaimIsCutFromWhatTheCallerHeard() {
	for said, heard := range map[string]string{
		"One moment, checking... I've checked the outage map and there are no outages.": "One moment, checking...",
		"You're all set for Saturday.":                     "",
		"Let me look. Your table is ready, you're booked.": "Let me look.",
		"I can help with that. What time would you like?":  "I can help with that. What time would you like?",
	} {
		s.Equal(heard, unsaid(said), said)
	}
}

func (s *AgentSuite) TestALookupBacksALookButNotAChange() {
	looked := []llm.Message{
		{Role: llm.User, Content: "move my appointment to Tuesday"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "lookup_appointment"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"appointments":[{"id":"a1"}]}`},
	}
	s.True(backsClaim(looked, "I've checked, you have one on Thursday.", nil))
	s.False(backsClaim(looked, "Your appointment has been rescheduled to Tuesday.", nil))

	moved := append(looked,
		llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "reschedule_appointment"}}},
		llm.Message{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"ok":true}`})
	s.True(backsClaim(moved, "Your appointment has been rescheduled to Tuesday.", nil))
}

func (s *AgentSuite) TestARefusalIsChasedToTheToolThatLogsIt() {
	s.True(refuses("I understand you're in pain, but I'm not able to send oxycodone refills over the phone."))
	s.False(refuses("I can help with that. What's your member ID?"))

	tools := harness.Tools{Tools: []harness.Tool{{Name: "verify_identity"}, {Name: "log_escalation"}}}
	s.Equal("log_escalation", loggingTool(tools))
	s.Empty(loggingTool(harness.Tools{Tools: []harness.Tool{{Name: "create_reservation"}}}))
	s.Equal("create_transfer_summary", loggingTool(harness.Tools{Tools: []harness.Tool{{Name: "apply_credit"}, {Name: "create_transfer_summary"}}}))
	s.True(refuses("I still can't change your plan under a threat, but I can look into a credit."))
	s.True(refuses("I can't credit a bill under a threat, but I can look into it."))
	s.False(refuses("Sorry, I can't hear you very well."))
	s.True(refuses("On the credit, I did try, but the account isn't eligible for a bill credit, so that one isn't possible."))

	logged := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "log_escalation"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"ok":true}`},
		{Role: llm.User, Content: "just send it"},
	}
	s.True(calledInCall(logged, "log_escalation"))
	s.False(calledInCall(logged[2:], "log_escalation"))
}

func (s *AgentSuite) TestAClaimThatNamesWhatAToolMakesTakesThatTool() {
	tools := []string{"verify_account", "walk_reboot", "create_ticket", "dispatch_tech"}
	rebooted := []llm.Message{
		{Role: llm.User, Content: "please keep going"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "walk_reboot"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"ok":true}`},
	}
	s.False(backsClaim(rebooted, "Since the Internet was still down, I opened a ticket for you.", tools))
	s.True(backsClaim(rebooted, "You're all set, the gateway is back.", tools))

	ticketed := append(rebooted,
		llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "create_ticket"}}},
		llm.Message{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"id":"T-1842"}`})
	s.True(backsClaim(ticketed, "I opened a ticket for you.", tools))
	s.False(backsClaim(ticketed, "Your technician is dispatched for the morning window.", tools))
}

func (s *AgentSuite) TestWhatFollowsAPromisedStepIsCutFromWhatTheCallerHeard() {
	said := "Let me verify your account information. I have your PIN as 4471. Is that correct?"
	s.Equal("Let me verify your account information.", heardOf(said, "[short pause] I have your PIN as 4471."))
	s.Equal("Let me look.", heardOf("Let me look. Your table is ready, you're booked.", ""))
	s.True(promisesStep("Let me verify your account information."))
	s.True(promisesStep("One moment."))
	s.False(promisesStep("What time would you like?"))
	s.True(directionTags.MatchString("[checks outage map] I don't see any outages."))
}

func (s *AgentSuite) TestOnlyAQuestionTheCallerOwesAnAnswerToShortensTheWait() {
	for said, asked := range map[string]bool{
		"What time would you like?":                         true,
		"Is there anything else you need help with?":        false,
		"Is there anything else I can help you with today?": false,
		"Anything else?":                                    false,
		"Your table is booked.":                             false,
		"I can help with that. To get started, I'll need your date of birth and your member ID.": true,
		"I'll need your name. Your table is booked.":                                             false,
	} {
		s.Equal(asked, owesAnAnswer(said), said)
	}
}

func (s *AgentSuite) TestWhatAReplyOwesACallFor() {
	cases := []struct {
		said   string
		backed bool
		owes   bool
	}{
		{"Order twelve, one moment, checking...", false, true},
		{"One moment, checking to see if we can finalize that, Patel, you're booked: 8 at 8:15.", false, true},
		{"Patel, you're booked for eight at 8:15.", false, true},
		{"I've checked the outage map and there are no outages.", false, true},
		{"Alvarez, you're booked: 4 at 7:30.", true, false},
		{"Checking orders is what I do. Which order is it?", false, false},
		{"Thank you, Maya. I will reschedule your appointment to Tuesday morning.", true, true},
		{"I'll book that for you. What name should it be under?", false, false},
		{"You're verified, Maya. Let me pull up your appointments now.", true, true},
		{"I have Maya Chen, member ABC123456. Let me verify that now.", false, false},
		{"Is there anything else I can help you with?", false, false},
	}
	for _, c := range cases {
		s.Equal(c.owes, owesACall(c.said, c.backed), c.said)
	}
}

func (s *AgentSuite) TestOnlyAToolThatAnsweredBacksAClaim() {
	// LFM2.5 told a Voicebench caller "I've got you down for eight at 8:15" after
	// create_reservation came back "no table at 8:15".
	caller := llm.Message{Role: llm.User, Content: "book it"}
	for _, c := range []struct {
		result string
		backed bool
	}{
		{`{"reservation_id":"r1"}`, true},
		{`{"error":"no table at 8:15"}`, false},
		{didNotWork + "no table at 8:15. Tell the caller, in your own words.", false},
		{`{"ok":false}`, false},
		{`{"error":""}`, true},
	} {
		history := []llm.Message{caller, {Role: llm.ToolResult, Content: c.result}}
		s.Equal(c.backed, toolRanSinceCaller(history), c.result)
	}
	s.True(owesACall("I've got you down for eight at 8:15.", false))
}

func (s *AgentSuite) TestAReplyThatPromisesNothingIsNotChasedUp() {
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Checking orders is what I do. Which order is it?"}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "can you check an order")

	s.eventually(func() bool { return s.spokenText("Which order is it") }, "the reply was not spoken")
	model := s.model
	s.Never(func() bool { return len(model.requests()) > 1 }, 200*time.Millisecond, 10*time.Millisecond,
		"a question to the caller owes no tool")
}

func (s *AgentSuite) TestTheReplyToAToolResultContinuesTheTurnThatAskedForIt() {
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return countOf[Responding](s.reported()) == 2 },
		"the tool result was never answered")
	var asked, followed Responding
	for _, event := range s.reported() {
		if responding, ok := event.(Responding); ok {
			if asked.TurnID == "" {
				asked = responding
			} else {
				followed = responding
			}
		}
	}
	s.Empty(asked.Continues, "a question somebody asked continues nothing")
	s.Equal(asked.TurnID, followed.Continues)
}

func (s *AgentSuite) TestALostVoiceIsReplacedForTheRestOfTheCall() {
	// A voice can go part way through a call, as one out of quota does. Kept, it leaves the
	// caller with silence for as long as the call lasts.
	s.join(false)
	s.replacement = newStubTTS(false)
	lost := s.agent.voice()
	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Err: errors.New("quota exceeded"), Context: "read", Fatal: true,
	})
	s.eventually(func() bool { return s.agent.voice() != lost },
		"the lost voice was never replaced")

	_, err := s.agent.RespondTo(s.ctx, "where is my order", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.replacement.spoken()) > 0 },
		"the reply was not spoken in the voice that replaced the lost one")
}

func (s *AgentSuite) TestAnAgentWithNoVoiceOpensItsVoiceWithNone() {
	s.join(true)

	s.Empty(s.voiceAsked.Voice, "the provider's default, which is what no voice has always meant")
}

func (s *AgentSuite) TestTheVoiceThatReplacesALostOneSpeaksInTheAgentsVoice() {
	s.voiceName = "library-voice"
	s.join(false)
	s.Equal("library-voice", s.voiceAsked.Voice)
	s.replacement = newStubTTS(false)
	lost := s.agent.voice()
	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Err: errors.New("quota exceeded"), Context: "read", Fatal: true,
	})
	s.eventually(func() bool { return s.agent.voice() != lost }, "the lost voice was never replaced")

	s.Equal("library-voice", s.voiceAsked.Voice, "the replacement is not opened in its provider's default")
}

func (s *AgentSuite) TestReturnedWorkWaitsForTheReplacementVoice() {
	s.delegates()
	s.join(true)
	s.model.reply = nil
	s.model.then = []string{"The result is twelve point six three."}
	s.subagent.reply = nil
	s.voice.silent = true
	s.replacement = newStubTTS(true)

	started := make(chan struct{})
	release := make(chan struct{})
	s.replacement.mu.Lock()
	s.replacement.startStarted = started
	s.replacement.startRelease = release
	s.replacement.mu.Unlock()
	var releaseOnce sync.Once
	openReplacement := func() { releaseOnce.Do(func() { close(release) }) }
	defer openReplacement()

	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool {
		requests := s.model.requests()
		return len(requests) > 0 && s.model.script(requests[0].ID) != nil
	}, "the first model response never opened")
	turnID := s.model.requests()[0].ID
	s.model.writes(turnID, "I will check that for you.")
	s.eventually(func() bool { return s.spokenText("I will check that for you") },
		"the first reply never reached the old voice")
	s.model.writes(turnID, `<ask skill="think">15% of 84.20</ask>`)
	s.model.finishes(turnID)
	s.eventually(func() bool {
		requests := s.subagent.requests()
		return len(requests) > 0 && s.subagent.script(requests[0].ID) != nil
	}, "the delegated work never started")
	taskID := s.subagent.requests()[0].ID
	s.eventually(func() bool {
		return countOf[Responded](s.reported()) == 1 && !s.agent.converse.Listening()
	}, "the first reply did not settle the caller's floor")
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.utterances > 0
	}, "the silent old voice did not keep its utterance open")
	old := s.agent.voice()

	s.voice.mu.Lock()
	s.voice.unavailable = true
	s.voice.mu.Unlock()
	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Fatal: true, Err: errors.New("socket closed"),
	})
	s.eventually(func() bool {
		select {
		case <-started:
			return true
		default:
			return false
		}
	}, "the replacement voice never reached its start gate")
	s.eventually(func() bool { return countOf[Error](s.reported()) == 1 },
		"the old voice failure was not reported")
	errorsBeforeFollow := countOf[Error](s.reported())

	s.subagent.writes(taskID, "It is 12.63.")
	s.subagent.finishes(taskID)
	s.eventually(func() bool { return countOf[TaskSettled](s.reported()) == 1 },
		"the delegated result never returned")
	s.NoError(s.agent.follow())
	s.never(func() bool { return countOf[Error](s.reported()) > errorsBeforeFollow },
		"the returned work was sent to the unavailable voice")
	s.True(s.agent.harness.Pending(), "the returned result remains queued during reconnect")
	s.True(s.agent.Busy(), "the returned result remains owed while the replacement starts")

	openReplacement()
	s.eventually(func() bool {
		voice := s.agent.voice()
		return voice != nil && voice != old
	}, "the replacement voice never became active")
	var resultTurnID string
	s.eventually(func() bool {
		for _, request := range s.model.requests() {
			if strings.Contains(request.Instructions, "It is 12.63.") &&
				slices.ContainsFunc(request.Input, func(message llm.Message) bool {
					return message.Content == "what is 15% of 84.20"
				}) {
				resultTurnID = request.ID
				return true
			}
		}
		return false
	}, "the replacement model request did not receive the delegated result")
	s.eventually(func() bool {
		return slices.ContainsFunc(s.replacement.spoken(), func(request tts.Request) bool {
			return strings.Contains(request.Text, "twelve point six three")
		})
	}, "the returned result was not spoken by the replacement voice")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 },
		"the replacement reply produced no audio")
	s.eventually(func() bool {
		for _, event := range s.reported() {
			if spoke, ok := event.(Spoke); ok && spoke.TurnID == resultTurnID {
				return true
			}
		}
		return false
	}, "the replacement turn did not report completed speech")
	s.eventually(func() bool { return !s.agent.Busy() }, "the replacement reply never settled")
	s.Equal(errorsBeforeFollow, countOf[Error](s.reported()),
		"the replacement reply left an additional voice error")
}

func (s *AgentSuite) TestACompletionFromALostVoiceDoesNotSettleReplacementSpeech() {
	held := make(chan struct{})
	releaseOld := make(chan struct{})
	s.joinWithVoice(true, func(provider *stubTTS) tts.TTS {
		return newHeldCompletionTTS(provider, held, releaseOld)
	})
	s.voice.silent = true
	s.model.reply = []string{"The old voice is speaking."}
	s.model.then = []string{"The replacement voice is speaking."}
	s.replacement = newStubTTS(true)
	s.replacement.silent = true

	started := make(chan struct{})
	releaseStart := make(chan struct{})
	s.replacement.mu.Lock()
	s.replacement.startStarted = started
	s.replacement.startRelease = releaseStart
	s.replacement.mu.Unlock()
	var startOnce sync.Once
	openReplacement := func() { startOnce.Do(func() { close(releaseStart) }) }
	var oldOnce sync.Once
	openOldCompletion := func() { oldOnce.Do(func() { close(releaseOld) }) }
	defer func() {
		openReplacement()
		openOldCompletion()
	}()

	oldTurnID, err := s.agent.RespondTo(s.ctx, "first request", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool {
		return len(s.voice.spoken()) > 0 && countOf[Responded](s.reported()) == 1 && s.agent.Busy()
	}, "the old voice did not leave an utterance in flight")
	oldSynthesisID := s.voice.spoken()[0].ID
	old := s.agent.voice()

	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Fatal: true, Err: errors.New("quota exceeded"),
	})
	s.eventually(func() bool {
		select {
		case <-started:
			return true
		default:
			return false
		}
	}, "the replacement voice never reached its start gate")
	s.voice.emitter.Send(tts.SynthesisComplete{
		SynthesisID: oldSynthesisID, Characters: 24, Interrupted: true,
	})
	s.eventually(func() bool {
		select {
		case <-held:
			return true
		default:
			return false
		}
	}, "the old completion was not held in the provider event stream")

	openReplacement()
	s.eventually(func() bool {
		voice := s.agent.voice()
		return voice != nil && voice != old
	}, "the replacement voice never became active")
	newTurnID, err := s.agent.RespondTo(s.ctx, "second request", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool {
		return len(s.replacement.spoken()) > 0 && countOf[Responded](s.reported()) == 2 && s.agent.Busy()
	}, "the replacement reply did not stay in flight")
	newSynthesisID := s.replacement.spoken()[0].ID
	s.replacement.emitter.Send(tts.AudioChunk{
		SynthesisID: newSynthesisID,
		Audio:       audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1},
	})
	s.eventually(func() bool { return len(s.edge.heard()) > 0 },
		"the replacement consumer did not publish its audio")

	openOldCompletion()
	s.eventually(func() bool {
		for _, event := range s.reported() {
			if turn, ok := event.(Turn); ok && turn.TurnID == oldTurnID {
				return true
			}
		}
		return false
	}, "the old session's completion was not consumed")
	s.True(s.agent.Busy(), "a late completion from the old session cleared replacement speech")

	s.replacement.emitter.Send(tts.SynthesisComplete{
		SynthesisID: newSynthesisID, Characters: 31, AudioDurationMs: 10, TimeToFirstByteMs: 5,
	})
	s.eventually(func() bool { return len(s.edge.heard()) > 0 },
		"the replacement voice did not publish its audio")
	s.eventually(func() bool {
		for _, event := range s.reported() {
			if spoke, ok := event.(Spoke); ok && spoke.TurnID == newTurnID {
				return true
			}
		}
		return false
	}, "the replacement turn did not report its completed speech")
	s.eventually(func() bool { return !s.agent.Busy() }, "the replacement completion did not settle")
}

func (s *AgentSuite) TestAnAgentThatLostItsVoiceMidReplyIsNoLongerTalking() {
	// What the lost voice was saying never finishes. Still counted as talking, the agent
	// would hear everything the caller says next as talking over it.
	s.join(false)
	s.voice.silent = true
	s.replacement = newStubTTS(false)
	_, err := s.agent.RespondTo(s.ctx, "where is my order", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 && s.agent.floor().Talking },
		"the reply never reached the voice")

	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Err: errors.New("quota exceeded"), Context: "read", Fatal: true,
	})

	s.eventually(func() bool { return !s.agent.floor().Talking },
		"the agent was still talking in a voice it had lost")
}

func (s *AgentSuite) TestTheGreetingIsSaidAsTheGreeting() {
	s.join(true)

	s.Require().NoError(s.agent.Greet(s.ctx, "Northwind, how can I help?", false))

	s.eventually(func() bool {
		for _, event := range s.reported() {
			if said, ok := event.(Said); ok && said.Purpose == PurposeGreeting {
				return true
			}
		}
		return false
	}, "the greeting was never reported as the greeting")
}

func (s *AgentSuite) TestALineALostVoiceWasSayingIsReportedCutOff() {
	// A greeting said in a voice out of quota was never heard, and a transcript kept it as
	// still being said for the rest of the call.
	s.join(false)
	s.voice.silent = true
	s.replacement = newStubTTS(false)
	s.Require().NoError(s.agent.Say(s.ctx, "Hi, thanks for getting in touch."))
	var turn string
	s.eventually(func() bool {
		for _, event := range s.reported() {
			if said, ok := event.(Said); ok {
				turn = said.TurnID
				return true
			}
		}
		return false
	}, "the greeting was never reported")

	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Err: errors.New("quota exceeded"), Context: "read", Fatal: true,
	})

	s.eventually(func() bool {
		for _, event := range s.reported() {
			if cut, ok := event.(Interrupted); ok && cut.TurnID == turn {
				return true
			}
		}
		return false
	}, "the line the lost voice was saying was never reported cut off")
}

func (s *AgentSuite) TestALineTheVoiceWouldNotTakeIsReportedCutOff() {
	// Reported before its sound, so a watcher has it in time: a line never said is then
	// closed, not left as still coming.
	s.join(false)
	s.voice.refuses = errors.New("text too long")

	s.Require().Error(s.agent.Say(s.ctx, "Hi, thanks for getting in touch."))

	s.eventually(func() bool {
		said, saidOK := firstOf[Said](s.reported())
		cut, cutOK := firstOf[Interrupted](s.reported())
		return saidOK && cutOK && cut.TurnID == said.TurnID
	}, "the line was left as still being said")
	s.False(s.agent.floor().Talking)
}

func (s *AgentSuite) TestAReplyStillBeingWrittenGoesOnWhenItsVoiceIsLost() {
	// The rest of it is said in the voice that replaces the lost one, so it is not over.
	s.join(false)
	s.voice.silent = true
	s.replacement = newStubTTS(false)
	s.model.reply = nil
	lost := s.agent.voice()
	_, err := s.agent.RespondTo(s.ctx, "where is my order", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	turn := s.model.requests()[0].ID
	s.model.writes(turn, "Let me look that up. ")
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the reply never reached the voice")

	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Err: errors.New("quota exceeded"), Context: "read", Fatal: true,
	})
	s.eventually(func() bool { return s.agent.voice() != lost }, "the lost voice was never replaced")

	for _, event := range s.reported() {
		if cut, ok := event.(Interrupted); ok {
			s.NotEqual(turn, cut.TurnID, "a reply still being written was reported cut off")
		}
	}
	s.False(s.agent.abandonedTurn(turn), "the rest of the reply would be dropped")
	s.model.writes(turn, "It shipped on Monday.")
	s.model.finishes(turn)
}

func (s *AgentSuite) TestWhatIsTypedIntoACallIsHeardAsTheCallerSaid() {
	// The transcript, the review and the logs take a caller's line from Heard. Without it
	// a line typed into a call is answered and then nowhere to be read.
	s.join(false)

	_, err := s.agent.RespondTo(s.ctx, "my order number is 12", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool {
		for _, event := range s.reported() {
			if heard, ok := event.(Heard); ok && heard.Text == "my order number is 12" {
				return true
			}
		}
		return false
	}, "the typed line was never reported as what the caller said")
}

func (s *AgentSuite) TestWhatIsTypedIntoATextSessionIsNotAlsoHeard() {
	// A text session records its caller's lines from Responding, and taking Heard as well
	// would record each of them twice.
	s.joinText()

	_, err := s.agent.RespondTo(s.ctx, "my order number is 12", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 },
		"the typed line was never answered")
	s.Zero(countOf[Heard](s.reported()))
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

func (s *AgentSuite) TestWhatIsSaidAsItIsIsReportedAsSaidNotAsAnAnswer() {
	// A transcript needs the greeting the caller heard, and nothing that treats an answer
	// as the model's reply to somebody should count it as one.
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool {
		for _, event := range s.events.seen() {
			if said, ok := event.(Said); ok && said.Text == "welcome" && said.TurnID != "" {
				return true
			}
		}
		return false
	}, "the line was never reported as said")
	for _, event := range s.events.seen() {
		_, answered := event.(Responded)
		s.False(answered, "a scripted line is not an answer")
	}
}

func (s *AgentSuite) TestASaidLineIsReportedBeforeItsVoiceFinishes() {
	// The transcript stores a turn's text when its voice finishes, so the text has to have
	// arrived by then or a greeting never reaches the transcript or the responses.
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the greeting was never spoken")
	reported := s.reported()
	said, _ := firstOf[Said](reported)
	spoke, _ := firstOf[Spoke](reported)
	s.Equal(spoke.TurnID, said.TurnID, "the line opens a turn of its own")
	s.Equal("welcome", said.Text)
	s.Less(slices.IndexFunc(reported, func(event Event) bool { _, ok := event.(Said); return ok }),
		slices.IndexFunc(reported, func(event Event) bool { _, ok := event.(Spoke); return ok }),
		"the text arrives before the voice finishes")
}

func (s *AgentSuite) TestTheModelIsToldTheLineTheCallOpenedWith() {
	// The caller's first words answer the greeting's question, which the model has to know.
	// It is told rather than handed a conversation opening on the assistant, which chat
	// templates that need a user turn first refuse.
	s.join(true)
	participant := stt.Participant{ID: "alice"}

	s.Require().NoError(s.agent.Say(s.ctx, "Where does it hurt?"))
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the question was never spoken")
	s.speak(participant)
	s.says(participant, "my lower back")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the answer was never asked about")
	request := s.model.requests()[0]
	s.Contains(request.Instructions, `You opened the call by saying: "Where does it hurt?"`)
	s.Require().Len(request.Input, 1)
	s.Equal(llm.User, request.Input[0].Role)
	s.Equal("my lower back", request.Input[0].Content)
}

func (s *AgentSuite) TestALineSaidPartWayThroughIsInTheConversation() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")

	s.Require().NoError(s.agent.Say(s.ctx, "Is there anything else?"))
	s.says(participant, "no thanks")

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the second turn was never asked")
	input := s.model.requests()[1].Input
	s.Require().Len(input, 4)
	s.Equal(llm.Message{Role: llm.Assistant, Content: "Is there anything else?"}, input[2])
	s.Equal("no thanks", input[3].Content)
	s.NotContains(s.model.requests()[1].Instructions, "You opened the call by saying")
}

func (s *AgentSuite) TestALineTheVoiceCouldNotSayIsNotToldToTheModel() {
	s.join(true)
	s.voice.mu.Lock()
	s.voice.unavailable = true
	s.voice.mu.Unlock()

	s.Require().Error(s.agent.Say(s.ctx, "Where does it hurt?"))

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the transcript was never told the line went unheard")
	said, _ := firstOf[Said](s.reported())
	interrupted, _ := firstOf[Interrupted](s.reported())
	s.Equal(said.TurnID, interrupted.TurnID)
	s.agent.mu.Lock()
	instructions := s.agent.instructions()
	s.agent.mu.Unlock()
	s.NotContains(instructions, "Where does it hurt?")
	s.Empty(s.agent.History())
}

func (s *AgentSuite) TestAnExactGreetingIsSaidWordForWord() {
	s.join(true)

	s.Require().NoError(s.agent.Greet(s.ctx, "Northwind, how can I help?", false))

	s.eventually(func() bool { return s.spokenText("Northwind, how can I help?") }, "the greeting was never said")
	s.Empty(s.model.composeRequests(), "an exact greeting does not need a model")
}

func (s *AgentSuite) TestAVariedGreetingIsTheModelsOwnWording() {
	s.join(true)
	s.model.composes = []string{"Hi, Northwind here, what can I do for you?"}

	s.Require().NoError(s.agent.Greet(s.ctx, "Northwind, how can I help?", true))

	s.eventually(func() bool { return s.spokenText("Hi, Northwind here, what can I do for you?") },
		"the model's variation was never said")
	s.False(s.spokenText("Northwind, how can I help?"), "the configured words were said as well")
	composed := s.model.composeRequests()
	s.Require().Len(composed, 1)
	s.Contains(composed[0].Input[len(composed[0].Input)-1].Content, "Northwind, how can I help?",
		"the model writes its variation of the configured greeting")
}

func (s *AgentSuite) TestAVariedGreetingTheModelCannotWriteIsSaidAsWritten() {
	s.join(true)
	s.model.composeFails = errors.New("the model is down")

	s.Require().NoError(s.agent.Greet(s.ctx, "Northwind, how can I help?", true))

	s.eventually(func() bool { return s.spokenText("Northwind, how can I help?") },
		"a caller was left in silence because the greeting could not be reworded")
}

func (s *AgentSuite) TestSimpleResponseAsksTheModel() {
	s.join(true)

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "what is the time"))

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Equal("what is the time", s.model.requests()[0].Input[0].Content)
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
	s.Positive(responded.TimeToFirstTokenMs,
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

func (s *AgentSuite) TestACallerIsToldWhenTheReplyIsLost() {
	s.join(false)
	s.model.refuses = errors.New("400 Bad Request")
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is our forecast")

	s.eventually(func() bool { return s.spokenText("went wrong") },
		"the caller was left in silence by a reply that failed")
	s.eventually(func() bool { return countOf[Error](s.reported()) == 1 }, "the failure was never reported")
	history := s.history()
	s.Require().NotEmpty(history)
	s.Equal(lostReply, history[len(history)-1].Content, "the model is shown what the caller heard")
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

// TestAnAnswerInWritingKeepsItsBrackets guards the reader against the voice's stripper.
//
// Square brackets are a stage direction only to a voice. To a reader they are a markdown
// link or an array literal, and a documentation agent writes both in almost every answer:
// stripped, every citation it gives becomes a bare URL in parentheses and every code
// sample loses its destructuring.
func (s *AgentSuite) TestAnAnswerInWritingKeepsItsBrackets() {
	s.joinText()
	s.model.reply = []string{
		"See [the channel guide](https://getstream.io/chat/docs/) ",
		"and hold them with useState([]).",
	}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I list channels"))

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal(
		"See [the channel guide](https://getstream.io/chat/docs/) and hold them with useState([]).",
		responded.Text,
	)
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

// TestThinkingInWritingIsReportedApartFromTheReply keeps a reader's view of the model's
// reasoning separate from the answer: it is reported as it streams and never joins the
// reply that is stored or remembered.
func (s *AgentSuite) TestThinkingInWritingIsReportedApartFromTheReply() {
	s.joinText()
	s.model.thinking = []string{"The user greets; ", "greet back."}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "hello"))

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("Hello there. How are you?", responded.Text)
	var thought strings.Builder
	for _, event := range s.reported() {
		if delta, ok := event.(ReasoningDelta); ok {
			thought.WriteString(delta.Text)
		}
	}
	s.Equal("The user greets; greet back.", thought.String())
}

func TestCountedDigitsArePausedBeforeTheyAreSpoken(t *testing.T) {
	cases := map[string]string{
		"Verified, last four 9821 on file.": "Verified, last four digits, 9821 on file.",
		"The last 4 9821 match.":            "The last 4 digits, 9821 match.",
		"Last four digits, please.":         "Last four digits, please.",
		"Party of four 7:30 on the patio.":  "Party of four 7:30 on the patio.",
	}
	for in, want := range cases {
		if got := pauseCounted(in); got != want {
			t.Errorf("%q = %q, want %q", in, got, want)
		}
	}
}

func TestADescribedStepIsTheToolNotYetCalled(t *testing.T) {
	tools := harness.Tools{Tools: []harness.Tool{{Name: "walk_reboot"}, {Name: "check_outage"}, {Name: "create_ticket"}}}
	checked := []llm.Message{{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{Name: "check_outage"}}}}
	cases := []struct {
		said string
		want string
	}{
		{"No outages. Let's try walking through a reboot of your gateway.", "walk_reboot"},
		{"Would you like me to walk you through a reboot?", ""},
		{"Let me check the outage map for you.", ""},
		{"Rebooting usually helps.", ""},
		{"I'll create a ticket for that error.", "create_ticket"},
		{"Since the reboot didn't work, I'll need to walk you through a formal reboot process.", "walk_reboot"},
		{"To walk you through a reboot, I'll need your account PIN.", ""},
	}
	for _, c := range cases {
		if got := describedStep(c.said, tools, checked); got != c.want {
			t.Errorf("describedStep(%q) = %q, want %q", c.said, got, c.want)
		}
	}
}

func (s *AgentSuite) TestAFailureThatNamesAnotherToolsStepNamesThatTool() {
	tools := []harness.Tool{{Name: "dispatch_tech"}, {Name: "walk_reboot"}, {Name: "check_outage"}}
	called := llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "1", Name: "dispatch_tech"}}}
	failed := llm.Message{Role: llm.ToolResult, ToolCallID: "1",
		Content: `{"error": "reboot the gateway before dispatching a truck"}`}
	s.Equal("walk_reboot", stepNamed([]llm.Message{called, failed}, tools))

	worked := llm.Message{Role: llm.ToolResult, ToolCallID: "1", Content: `{"ok": true, "note": "reboot done"}`}
	s.Empty(stepNamed([]llm.Message{called, worked}, tools), "a result that did not fail names no step")
	s.Empty(stepNamed([]llm.Message{called, failed, {Role: llm.User, Content: "ok"}}, tools),
		"the caller spoke since")
}

func (s *AgentSuite) TestACallThatLeavesARequiredArgumentEmptyIsNotRun() {
	tool := harness.Tool{Name: "verify_account", Parameters: map[string]any{
		"required": []any{"pin", "last4", "address"},
	}}
	s.Equal([]string{"last4", "address"}, blankRequired(tool, `{"pin": "4471", "last4": "", "address": " "}`))
	s.Empty(blankRequired(tool, `{"pin": "4471", "last4": "9821", "address": "14 Cedar Lane"}`))
	s.Empty(blankRequired(harness.Tool{Name: "check_outage"}, `{}`), "a tool that requires nothing")
}

func (s *AgentSuite) TestASucceededCallIsAnsweredFromItEvenAfterTheCallerSpoke() {
	booked := llm.ToolCall{ID: "1", Name: "create_reservation", Arguments: `{"time": "7:30"}`}
	history := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{booked}},
		{Role: llm.ToolResult, ToolCallID: "1", Content: `{"time": "7:30"}`},
		{Role: llm.User, Content: "Thanks."},
	}
	_, repeated := answeredSinceCaller(history, llm.ToolCall{Name: "create_reservation", Arguments: `{"time":"7:30"}`})
	s.True(repeated)

	failed := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "1", Name: "lookup_appointment", Arguments: `{}`}}},
		{Role: llm.ToolResult, ToolCallID: "1", Content: `{"error": "identity not verified"}`},
		{Role: llm.User, Content: "My member ID is ABC123456."},
	}
	_, repeated = answeredSinceCaller(failed, llm.ToolCall{Name: "lookup_appointment", Arguments: `{}`})
	s.False(repeated, "a call that failed may be tried again")
}

func (s *AgentSuite) TestACallTheRouterHeldBackRunsOnceThePersonHasActed() {
	echo := llm.ToolCall{ID: "1", Name: "crm__echo", Arguments: `{"text":"a joke"}`}
	answered := func(result string) bool {
		history := []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{echo}},
			{Role: llm.ToolResult, ToolCallID: "1", Content: result},
			{Role: llm.User, Content: "I pressed it, try again."},
		}
		_, repeated := answeredSinceCaller(history, llm.ToolCall{ID: "2", Name: echo.Name, Arguments: echo.Arguments})
		return repeated
	}

	// What internal/session answers a connector call it did not send with.
	for _, status := range []string{"authorization_required", "scope_required", "credential_rejected",
		"declined", "expired", "failed", "stopped", "unavailable"} {
		s.False(answered(`{"status":"`+status+`","message":"Tell them to press it."}`), status)
	}
	s.True(answered(`{"status":"sent","message":"Nash heard the joke."}`), "a status that says it was done answers the call")

	// A third-party write that was done and reports its new state under a generic status.
	for _, result := range []string{
		`{"status":"stopped","recording_id":"r1"}`,
		`{"status":"declined","event_id":"e1"}`,
		`{"status":"expired","id":"cs_1"}`,
		`{"status":"failed","job_id":"j1","rows_written":120}`,
		`{"status":"unavailable","message":"x","agent":"a1"}`,
	} {
		s.True(answered(result), result)
	}
}

func (s *AgentSuite) TestAClaimIsNotBackedByACallTheRouterHeldBack() {
	held := `{"status":"scope_required","message":"Tell them to press it."}`
	done := `{"status":"sent","message":"Nash heard the joke."}`
	history := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "Send it."},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "1", Name: "crm__send_message"}}},
			{Role: llm.ToolResult, ToolCallID: "1", Content: result},
		}
	}
	s.False(calledInCall(history(held), "crm__send_message"))
	s.True(calledInCall(history(done), "crm__send_message"))
	s.False(changedSinceCaller(history(held), anyTool))
	s.True(changedSinceCaller(history(done), anyTool))
	s.False(toolRanSinceCaller(history(held)))
	s.True(toolRanSinceCaller(history(done)))
	s.False(succeeded(history(held), "crm__send_message", false))
	s.True(succeeded(history(done), "crm__send_message", false))
}

func (s *AgentSuite) TestANameSpelledOutIsCalledWithTheNameAlone() {
	s.JSONEq(`{"name": "Patel", "party_size": 4}`, unspelled(`{"name": "Patel P A T E L", "party_size": 4}`))
	s.JSONEq(`{"name": "Ana Lopez"}`, unspelled(`{"name": "Ana Lopez L-O-P-E-Z"}`))
	s.JSONEq(`{"pin": "4471", "address": "14 Cedar Lane"}`, unspelled(`{"pin": "four four seven one", "address": "one four Cedar Lane"}`))
	s.JSONEq(`{"address": "14 Cedar Lane", "reason": "down since 6, error 840"}`, unspelled(`{"address": "fourteen Cedar Lane", "reason": "down since 6, error 840"}`))
	s.JSONEq(`{"address": "42 Elm Street"}`, unspelled(`{"address": "forty-two Elm Street"}`))
	s.JSONEq(`{"notes": "one high chair"}`, unspelled(`{"notes": "one high chair"}`), "a lone number word")
	s.Equal(`{"address": "14 Cedar Lane A B"}`, unspelled(`{"address": "14 Cedar Lane A B"}`), "letters that spell nothing before them")
}

func (s *AgentSuite) TestACallThatOnlyFillsInABlankIsTheSameCall() {
	s.True(fillsIn(`{"name": "Alvarez", "phone": ""}`, `{"name": "Alvarez", "phone": "512-550-1422"}`))
	s.False(fillsIn(`{"party_size": 4, "phone": ""}`, `{"party_size": 6, "phone": ""}`))
	s.True(fillsIn(`{"name": "Alvarez"}`, `{"name": "Alvarez", "phone": "512-550-1422"}`), "a text argument the first call left out")
	s.False(fillsIn(`{"party_size": 6}`, `{"party_size": 6, "patio": true}`), "a choice the first call never made")
	s.True(fillsIn(`{"party_size": 2}`, `{"party_size": 2, "high_chair": false}`), "a choice left at its default")
}

func (s *AgentSuite) TestAToolIsNotRunBeforeTheStepItsDescriptionNeeds() {
	tools := []harness.Tool{
		{Name: "verify_account", Description: "Required before tickets."},
		{Name: "walk_reboot", Description: "Must run before dispatch_tech."},
		{Name: "create_ticket", Description: "Open a support ticket. Identity required."},
		{Name: "dispatch_tech", Description: "Dispatch a technician. Identity required."},
		{Name: "check_outage", Description: "Check the outage map."},
	}
	result := func(name, content string) []llm.Message {
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: name, Name: name}}},
			{Role: llm.ToolResult, ToolCallID: name, Content: content},
		}
	}
	s.Equal("verify_account", unmetPrerequisite(tools[2], tools, nil))
	s.Equal("verify_account", unmetPrerequisite(tools[2], tools, result("verify_account", `{"ok": false}`)))
	s.Empty(unmetPrerequisite(tools[2], tools, result("verify_account", `{"ok": true}`)))
	verified := result("verify_account", `{"ok": true}`)
	s.Equal("walk_reboot", unmetPrerequisite(tools[3], tools, verified))
	s.Empty(unmetPrerequisite(tools[3], tools, append(verified, result("walk_reboot", `{"ok": false, "error_code": "840"}`)...)),
		"a reboot that failed is the one a dispatch needs")
	s.Empty(unmetPrerequisite(tools[4], tools, nil))
	s.Equal("verify_account", stepNamed(result("create_ticket", fmt.Sprintf(runFirst, "create_ticket", "verify_account", "verify_account")), tools))
}

func (s *AgentSuite) TestAGoAheadToWorkAlreadyDoneHasNothingToDo() {
	booked := []llm.Message{
		{Role: llm.User, Content: "That's all. Thanks."},
		{Role: llm.Assistant, Content: "Alvarez, you're booked: 4 at 7:30."},
		{Role: llm.User, Content: "Yes, that's right. Please go ahead."},
	}
	s.True(doneBeforeGoAhead(booked))
	booked[1].Content = "Shall I book it?"
	s.False(doneBeforeGoAhead(booked))
}

func (s *AgentSuite) TestAHoldOrADoneClaimOwesACall() {
	s.True(owesACall("One moment, checking. Let me verify your PIN 4471. Is that correct?", false),
		"a hold that ends in a question")
	s.False(owesACall("Checking orders is what I do. Which order is it?", false))
	s.True(owesACall("Your appointment has been successfully rescheduled to Tuesday morning.", false))
	s.False(owesACall("Your appointment has been successfully rescheduled to Tuesday morning.", true),
		"a claim a tool backs")
}

func TestALineComposedUnaskedSaysWhatItIsFor(t *testing.T) {
	require.Equal(t, PurposeCheckIn, composedFor(idlePurpose))
	require.Equal(t, PurposeUpdate, composedFor(updatePurpose))
	require.Empty(t, composedFor(greetPurpose), "a varied greeting is said as the greeting, not composed as a check-in")
}
