package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/openailive"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/openairealtime"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// heldPlayback models a track whose queued speech cannot play until it is dropped.
type heldPlayback struct {
	*loopbackEdge
	released chan struct{}
	once     sync.Once
}

func (e *heldPlayback) PublishAudio(pcm audio.PcmData) error {
	if err := e.loopbackEdge.PublishAudio(pcm); err != nil {
		return err
	}
	<-e.released
	return nil
}

func (e *heldPlayback) DropSpeech() {
	e.loopbackEdge.DropSpeech()
	e.once.Do(func() { close(e.released) })
}

func (e *heldPlayback) Leave() error {
	e.DropSpeech()
	return e.loopbackEdge.Leave()
}

func (s *AgentSuite) TestNativeInterruptionStopsBlockedPlaybackAndKeepsTheNextReply() {
	edge := &heldPlayback{loopbackEdge: newLoopbackEdge(), released: make(chan struct{})}
	agent, err := New(Options{CustomerID: "acme", Edge: edge, STS: &stsrouter.Router{}, STSTarget: "openai/gpt-realtime-2"})
	s.Require().NoError(err)
	agent.ctx, agent.cancel = context.WithCancel(s.ctx)
	s.T().Cleanup(func() { agent.Close() })
	seen := collect(agent)
	source := make(chan sts.Event, 8)
	ordered := make(chan sts.Event, 8)
	agent.pipe = newPipeline(agent.ctx, true)
	agent.pipe.running.Add(2)
	go agent.receiveSTS(agent.pipe, source, ordered)
	go agent.consumeSTS(agent.pipe, nil, ordered)
	s.T().Cleanup(func() { close(source) })

	pcm := audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24000, Channels: 1}
	source <- sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()}
	source <- sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: pcm}
	s.Eventually(func() bool {
		edge.mu.Lock()
		defer edge.mu.Unlock()
		return len(edge.published) == 1
	}, time.Second, time.Millisecond)
	source <- sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: pcm}
	source <- sts.ResponseComplete{ResponseID: "r1", Generation: 1, Interrupted: true}
	source <- sts.ResponseStarted{ResponseID: "r2", Generation: 2, At: time.Now()}
	source <- sts.AudioChunk{ResponseID: "r2", Generation: 2, Audio: pcm}
	source <- sts.ResponseComplete{ResponseID: "r2", Generation: 2}

	s.Eventually(func() bool { return countOf[Spoke](seen.seen()) == 1 }, time.Second, time.Millisecond)
	s.Equal(1, countOf[Interrupted](seen.seen()))
	spoke, _ := firstOf[Spoke](seen.seen())
	s.Equal("r2", spoke.TurnID)
	edge.mu.Lock()
	defer edge.mu.Unlock()
	s.Equal([]audio.PcmData{pcm}, edge.published)
}

// nativePeer exercises the real OpenAI adapter over a local WebSocket.
type nativePeer struct {
	conn   *websocket.Conn
	frames chan map[string]json.RawMessage
	setup  json.RawMessage
	// next is each connection the agent opens after the first, as it replaces a lost one.
	next chan *websocket.Conn
}

func (s *AgentSuite) joinNativeDelegating() *nativePeer { return s.joinNativeWith(nil) }

// joinNativeWith joins a native agent over the real OpenAI adapter, with configure changing
// its options first when given.
func (s *AgentSuite) joinNativeWith(configure func(*Options)) *nativePeer {
	handshake := func(conn *websocket.Conn, frames chan map[string]json.RawMessage) bool {
		if conn.WriteJSON(map[string]any{"type": "session.created", "session": map[string]any{}}) != nil {
			return false
		}
		var setup map[string]json.RawMessage
		if conn.ReadJSON(&setup) != nil {
			return false
		}
		frames <- setup
		return conn.WriteJSON(map[string]any{"type": "session.updated", "session": map[string]any{}}) == nil
	}
	adapter := func(url string, spec routing.Spec) (sts.STS, error) {
		return openairealtime.New(openairealtime.Options{
			Vendor: openairealtime.OpenAI, URL: url, APIKey: "test",
			Model: spec.Model, Tools: spec.Tools, Instructions: spec.STS.Instructions, InputTranscript: true,
		})
	}
	return s.joinNativeOver(handshake, adapter, "openai/gpt-realtime-2", configure)
}

// joinLiveDelegating joins a native agent over the real GPT-Live adapter, whose replies
// settle after replyGap of quiet.
func (s *AgentSuite) joinLiveDelegating(replyGap time.Duration) *nativePeer {
	handshake := func(conn *websocket.Conn, frames chan map[string]json.RawMessage) bool {
		var start map[string]json.RawMessage
		if conn.ReadJSON(&start) != nil {
			return false
		}
		frames <- start
		return conn.WriteJSON(map[string]any{"type": "session.started", "session": map[string]any{"id": "test-live", "model": "gpt-live-1"}}) == nil
	}
	adapter := func(url string, spec routing.Spec) (sts.STS, error) {
		return openailive.New(openailive.Options{
			APIKey: "test-key", URL: url, Model: spec.Model, Tools: spec.Tools,
			Instructions: spec.STS.Instructions, ReplyGap: replyGap, HandshakeTimeout: 5 * time.Second,
		})
	}
	return s.joinNativeOver(handshake, adapter, "openai/gpt-live-1", nil)
}

// joinNativeOver joins a native agent, delegating to a stub subagent, over the adapter that
// connects to a local server greeting each connection with handshake.
func (s *AgentSuite) joinNativeOver(
	handshake func(*websocket.Conn, chan map[string]json.RawMessage) bool,
	adapter func(url string, spec routing.Spec) (sts.STS, error),
	target string,
	configure func(*Options),
) *nativePeer {
	connections := make(chan *websocket.Conn, 4)
	frames := make(chan map[string]json.RawMessage, 32)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		if !handshake(conn, frames) {
			return
		}
		connections <- conn
		for {
			var frame map[string]json.RawMessage
			if conn.ReadJSON(&frame) != nil {
				return
			}
			frames <- frame
		}
	}))
	s.T().Cleanup(server.Close)
	s.subagent = newStubLLM()
	registry := llmrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return s.subagent, nil })
	reasoner, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech := stsrouter.NewRegistry()
	speech.Register("openai", func(spec routing.Spec) (sts.STS, error) {
		return adapter("ws"+strings.TrimPrefix(server.URL, "http"), spec)
	})
	speech.Register("gemini", func(spec routing.Spec) (sts.STS, error) {
		return adapter("ws"+strings.TrimPrefix(server.URL, "http"), spec)
	})
	router, err := stsrouter.New(stsrouter.Options{Config: config[routing.STS], Registry: speech})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	s.edge = newLoopbackEdge()
	options := Options{
		CustomerID: "acme", Edge: s.edge, STS: router, STSTarget: target,
		LLM: reasoner, SubagentTarget: "stub/stub-model",
		Skills: harness.Skills{Skills: []harness.Skill{{Name: "think", Description: "reason carefully", Instructions: "Solve it", Deadline: time.Minute}}},
	}
	if configure != nil {
		configure(&options)
	}
	s.agent, err = New(options)
	s.Require().NoError(err)
	s.events = collect(s.agent)
	s.T().Cleanup(func() { s.agent.Close() })
	s.Require().NoError(s.agent.Join(s.ctx))
	return &nativePeer{conn: <-connections, frames: frames, next: connections}
}

func (s *AgentSuite) nativeFrame(peer *nativePeer, kind string) map[string]json.RawMessage {
	for {
		select {
		case frame := <-peer.frames:
			if string(frame["type"]) == strconv.Quote(kind) {
				return frame
			}
		case <-time.After(settleFor):
			s.FailNow("missing native frame", kind)
			return nil
		}
	}
}

func (s *AgentSuite) nativeSend(peer *nativePeer, frame string) {
	s.Require().NoError(peer.conn.WriteMessage(websocket.TextMessage, []byte(frame)))
}

func (s *AgentSuite) TestNativeDelegationKeepsTalkingAndDeliversTheResultWhenIdle() {
	peer := s.joinNativeDelegating()
	setup := s.nativeFrame(peer, "session.update")
	s.Contains(string(setup["session"]), "delegate_skill")
	s.Contains(string(setup["session"]), "cancel_skill")
	s.Nil(s.agent.LLM(), "STS must not open a text conversation model")
	turnID, err := s.agent.RespondTo(s.ctx, "My booking is for six people", nil, options.LLM{})
	s.Require().NoError(err)
	s.Empty(turnID, "a native agent names its own turns")
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	ack := s.nativeFrame(peer, "conversation.item.create")
	s.Contains(string(ack["item"]), "task_id")
	s.Contains(string(ack["item"]), "running")
	s.nativeFrame(peer, "response.create")
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)
	request := s.subagent.requests()[0]
	s.Equal("My booking is for six people", request.Input[0].Content)
	s.True(s.agent.delegating())

	s.nativeSend(peer, `{"type":"response.created","response":{"id":"chat"}}`)
	s.nativeSend(peer, `{"type":"response.output_audio.delta","response_id":"chat","delta":"AQACAA=="}`)
	s.Eventually(func() bool { return len(s.edge.heard()) == 1 }, time.Second, time.Millisecond)
	s.True(s.agent.delegating(), "speech must flow while the subagent is still working")
	s.Eventually(func() bool {
		response, ok := firstOf[Responded](s.reported())
		return ok && response.PendingWork
	}, time.Second, time.Millisecond)
	s.nativeSend(peer, `{"type":"conversation.item.input_audio_transcription.completed","transcript":"My booking is for six people"}`)
	s.subagent.writes(request.ID, "The total is 42 dollars.")
	s.subagent.finishes(request.ID)
	s.Eventually(func() bool { return s.agent.harness.Pending() }, time.Second, time.Millisecond)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"chat","status":"completed"}}`)
	delivery := s.nativeFrame(peer, "response.create")
	s.Contains(string(delivery["response"]), "The total is 42 dollars.")
	s.False(s.agent.harness.Pending())
	s.Eventually(func() bool { return countOf[TaskSettled](s.reported()) == 1 }, time.Second, time.Millisecond)
	s.Equal(1, countOf[Delegated](s.reported()), "a duplicated tool call must not restart the task")
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInATurnAScheduledTaskFiredIsAskedAsScheduled() {
	// A scheduled turn is the customer's spend, and so is the work the model hands over
	// for it, until the caller is next heard.
	peer := s.joinNativeDelegating()
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.nativeSend(peer, `{"type":"conversation.item.input_audio_transcription.completed","transcript":"And for eight people?"}`)
	s.Eventually(func() bool { return countOf[Heard](s.reported()) == 1 }, time.Second, time.Millisecond)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r2","call_id":"call2","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Price eight seats\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 2 }, time.Second, time.Millisecond)

	s.Equal([]bool{true, false}, s.subagent.askedScheduled(), "the work the caller's own turn hands over is theirs")
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInATypedTurnIsNotAskedAsScheduled() {
	peer := s.joinNativeDelegating()
	_, err := s.agent.RespondTo(s.ctx, "What is my booking total?", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{false}, s.subagent.askedScheduled())
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInATurnTheCallerSpokeIntoBeforeItsTranscriptIsNotAskedAsScheduled() {
	peer := s.joinNativeDelegating()
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)

	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r2","call_id":"call2","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Price eight seats\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{false}, s.subagent.askedScheduled(), "the caller's voice came before the transcript")
}

// heardSpeechStart waits until the agent has taken in the caller's speech starting.
func (s *AgentSuite) heardSpeechStart() {
	s.Eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.nativeListening
	}, time.Second, time.Millisecond)
}

// heldStart holds each tool call as it starts, between its arrival and its run, until let.
type heldStart struct {
	started chan struct{}
	release chan struct{}
	let     func()
}

func newHeldStart() *heldStart {
	held := &heldStart{started: make(chan struct{}, 4), release: make(chan struct{})}
	held.let = sync.OnceFunc(func() { close(held.release) })
	return held
}

func (h *heldStart) hold(ToolStarted) {
	h.started <- struct{}{}
	<-h.release
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInAScheduledTurnIsAskedAsScheduledWhenTheCallerCutsInBeforeItRuns() {
	// The barge-in arrives after the model asked for the work and before the work runs: the
	// work is still the scheduled turn's.
	held := newHeldStart()
	s.T().Cleanup(held.let)
	peer := s.joinNativeWith(func(options *Options) { options.OnToolStarted = held.hold })
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	select {
	case <-held.started:
	case <-time.After(settleFor):
		s.FailNow("the tool call never started")
	}
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	held.let()
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{true}, s.subagent.askedScheduled(), "the caller cut in after the model asked")
	s.Eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return len(s.agent.toolScheduled) == 0
	}, time.Second, time.Millisecond, "a finished call's mark is kept")
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInAScheduledReplyIsAskedAsScheduledWhenTheCallerCutInBeforeTheCall() {
	// The model can finish asking for a tool in the reply the caller cut into: that call is
	// the reply's, and the reply is the scheduled turn's.
	peer := s.joinNativeDelegating()
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"cancelled"}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r2","call_id":"call2","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Price eight seats\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 2 }, time.Second, time.Millisecond)

	s.Equal([]bool{true, false}, s.subagent.askedScheduled(), "the reply that answers the caller is theirs")
}

// stoppedSpeaking sends the caller's stop and waits until the agent has taken it in.
func (s *AgentSuite) stoppedSpeaking(peer *nativePeer) {
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_stopped"}`)
	s.Eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return !s.agent.nativeListening
	}, time.Second, time.Millisecond)
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInTheReplyToTheCallersStopIsNotAskedAsScheduledWhenAScheduledTurnLandsFirst() {
	// The server opens a reply for the caller's speech when they stop. A scheduled turn typed
	// in before that reply begins is answered after it, so the reply is the caller's.
	peer := s.joinNativeDelegating()
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.stoppedSpeaking(peer)
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r0"}}`)
	s.nativeSend(peer, `{"type":"conversation.item.input_audio_transcription.completed","transcript":"And for eight people?"}`)
	s.Eventually(func() bool { return countOf[Heard](s.reported()) == 1 }, time.Second, time.Millisecond)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r0","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Price eight seats\"}"}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{false}, s.subagent.askedScheduled(), "the caller's reply is theirs")
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInAScheduledTurnAfterTheCallerWasAnsweredIsAskedAsScheduled() {
	// Only the first reply after the caller's stop is theirs: a scheduled turn typed in once
	// it began keeps its reply's mark across a barge-in.
	peer := s.joinNativeDelegating()
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.stoppedSpeaking(peer)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r0"}}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r0","status":"completed"}}`)
	s.Eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, time.Second, time.Millisecond)
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{true}, s.subagent.askedScheduled(), "the caller's answer came before the scheduled turn")
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInAScheduledTurnOnAReplacedSessionIsAskedAsScheduled() {
	// A session lost after the caller stopped never answers them, and its replacement heard
	// no stop: its first reply to a scheduled turn is the task's.
	peer := s.joinNativeDelegating()
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.stoppedSpeaking(peer)
	old := s.agent.speech()
	_ = peer.conn.Close()
	var replaced *websocket.Conn
	select {
	case replaced = <-peer.next:
	case <-time.After(settleFor):
		s.FailNow("the lost session was never replaced")
	}
	s.Eventually(func() bool { return !s.agent.switching.Load() && s.agent.speech() != old }, settleFor, time.Millisecond)
	peer.conn = replaced
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{true}, s.subagent.askedScheduled(), "the old session's stop is not the new one's")
}

func (s *AgentSuite) TestWorkGPTLiveHandsOverInAScheduledTurnIsAskedAsScheduledWhenItsCallComesAfterTheCaller() {
	// GPT-Live's backend asks for the work on its own time: here after the reply that handed
	// it over ended and the caller spoke again, in the reply answering them.
	peer := s.joinLiveDelegating(50 * time.Millisecond)
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"session.delegation.created","delegation":{"id":"d1","type":"responses","target":"responses"}}`)
	s.Eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, settleFor, time.Millisecond, "the reply the work was handed over in never ended")
	s.nativeSend(peer, `{"type":"session.input_transcript.delta","delta":"And for eight people?"}`)
	s.heardSpeechStart()
	s.nativeSend(peer, `{"type":"session.output_audio.delta","delta":"6APQBw=="}`)
	s.Eventually(func() bool { return countOf[Responding](s.reported()) == 2 }, settleFor, time.Millisecond, "the caller was never answered")
	s.nativeSend(peer, `{"type":"response.event","delegation_id":"d1","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, settleFor, time.Millisecond)

	s.Equal([]bool{true}, s.subagent.askedScheduled(), "the work is the scheduled turn's")
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInTheReplyToTheCallersStopIsNotAskedAsScheduledWhenItsCallComesBeforeTheirWords() {
	// The transcript of what the caller said runs alongside the reply, so a call of the reply
	// answering them can arrive while the scheduled turn typed in before it is the last thing.
	peer := s.joinNativeDelegating()
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.stoppedSpeaking(peer)
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r0"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r0","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Price eight seats\"}"}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{false}, s.subagent.askedScheduled(), "the caller's reply is theirs")
}

func (s *AgentSuite) TestWorkGPTLiveHandsOverInTheCallersReplyIsNotAskedAsScheduledWhenItsCallComesInAScheduledTurn() {
	// The backend asks for the work after a scheduled turn was typed in, but the work was
	// handed over in the reply answering the caller.
	peer := s.joinLiveDelegating(50 * time.Millisecond)
	s.nativeSend(peer, `{"type":"session.input_transcript.delta","delta":"Price eight seats"}`)
	s.heardSpeechStart()
	s.nativeSend(peer, `{"type":"session.output_audio.delta","delta":"6APQBw=="}`)
	s.nativeSend(peer, `{"type":"session.delegation.created","delegation":{"id":"d0","type":"responses","target":"responses"}}`)
	s.Eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, settleFor, time.Millisecond, "the caller's reply never ended")
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeSend(peer, `{"type":"response.event","delegation_id":"d0","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Price eight seats\"}"}}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, settleFor, time.Millisecond)

	s.Equal([]bool{false}, s.subagent.askedScheduled(), "the work is the caller's")
}

// geminiAudio is a server message with a tenth of a second of the model's speech.
func geminiAudio() string {
	return `{"serverContent":{"modelTurn":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"` +
		base64.StdEncoding.EncodeToString(make([]byte, 4800)) + `"}}]}}}`
}

func (s *AgentSuite) TestWorkGeminiHandsOverInAScheduledTurnIsAskedAsScheduledAfterTheCallerWasHeardDuringAReply() {
	// On Gemini a reply still open takes the caller's stop and none follows for it, so the
	// next reply, a scheduled turn's, is the task's.
	handshake := func(conn *websocket.Conn, frames chan map[string]json.RawMessage) bool {
		var setup map[string]json.RawMessage
		if conn.ReadJSON(&setup) != nil {
			return false
		}
		frames <- setup
		return conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)) == nil
	}
	adapter := func(url string, spec routing.Spec) (sts.STS, error) {
		return gemini.New(gemini.Options{APIKey: "test", URL: url, Model: spec.Model, Tools: spec.Tools, Instructions: spec.STS.Instructions})
	}
	peer := s.joinNativeOver(handshake, adapter, "gemini/gemini-3.8-live", nil)
	send := func(frame string) { s.Require().NoError(peer.conn.WriteMessage(websocket.TextMessage, []byte(frame))) }
	send(`{"serverContent":{"inputTranscription":{"text":"What time is my table?"}}}`)
	s.heardSpeechStart()
	send(geminiAudio())
	s.Eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, settleFor, time.Millisecond)
	send(`{"serverContent":{"inputTranscription":{"text":"mhm"}}}`)
	s.heardSpeechStart()
	send(geminiAudio())
	send(`{"serverContent":{"turnComplete":true}}`)
	s.Eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, settleFor, time.Millisecond, "the caller's reply never ended")
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	send(geminiAudio())
	s.Eventually(func() bool { return countOf[Responding](s.reported()) == 2 }, settleFor, time.Millisecond, "the scheduled turn was never answered")
	send(`{"serverContent":{"inputTranscription":{"text":"wait"}}}`)
	s.heardSpeechStart()
	send(`{"toolCall":{"functionCalls":[{"id":"call1","name":"delegate_skill","args":{"skill":"think","prompt":"Work out the booking total"}}]}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, settleFor, time.Millisecond)

	s.Equal([]bool{true}, s.subagent.askedScheduled(), "the work is the scheduled turn's")
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInAReplyWithNoIDIsNotAskedAsScheduled() {
	// A call the adapter can tie to no reply is not mistaken for the scheduled turn's.
	peer := s.joinNativeDelegating()
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{false}, s.subagent.askedScheduled())
}

func (s *AgentSuite) TestWorkANativeModelHandsOverInAReplyWithNoIDIsAskedAsScheduledDuringAScheduledTurn() {
	// A call tied to no reply is not taken for the reply that answered the caller, whose ID
	// is not known either: during a scheduled turn it is the task's.
	peer := s.joinNativeDelegating()
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{true}, s.subagent.askedScheduled())
}

func (s *AgentSuite) TestAToolsPictureInAScheduledNativeTurnIsLookedAtAsScheduledWhenTheCallerCutsInWhileItRuns() {
	running := make(chan struct{}, 1)
	release := make(chan struct{})
	let := sync.OnceFunc(func() { close(release) })
	s.T().Cleanup(let)
	peer := s.joinNativeWith(func(options *Options) {
		options.Skills.Skills = append(options.Skills.Skills, harness.Skill{Name: "vision", Description: "look at pictures", Instructions: "describe them", Deadline: time.Minute})
		options.Tools = harness.Tools{Tools: []harness.Tool{{Name: "photograph_room", Description: "photograph the room"}}}
		options.ToolRunner = &stubToolRunner{run: func(context.Context, llm.ToolCall) ([]llm.ContentPart, error) {
			running <- struct{}{}
			<-release
			return []llm.ContentPart{{Text: "the room"}, {Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8, 0xff}}}}, nil
		}}
	})
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "what is in the room", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"photograph_room","arguments":"{}"}`)
	select {
	case <-running:
	case <-time.After(settleFor):
		s.FailNow("the tool never ran")
	}
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	let()
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{true}, s.subagent.askedScheduled(), "looking at the tool's picture is the scheduled turn's")
}

func (s *AgentSuite) TestAToolsPictureAskedInTheCallersOwnNativeReplyIsNotLookedAtAsScheduled() {
	// The scheduled turn is over when the caller speaks: a picture tool the model asks for in
	// the reply that answers the caller is the caller's spend, not the scheduled turn's.
	running := make(chan struct{}, 1)
	release := make(chan struct{})
	let := sync.OnceFunc(func() { close(release) })
	s.T().Cleanup(let)
	peer := s.joinNativeWith(func(options *Options) {
		options.Skills.Skills = append(options.Skills.Skills, harness.Skill{Name: "vision", Description: "look at pictures", Instructions: "describe them", Deadline: time.Minute})
		options.Tools = harness.Tools{Tools: []harness.Tool{{Name: "photograph_room", Description: "photograph the room"}}}
		options.ToolRunner = &stubToolRunner{run: func(context.Context, llm.ToolCall) ([]llm.ContentPart, error) {
			running <- struct{}{}
			<-release
			return []llm.ContentPart{{Text: "the room"}, {Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8, 0xff}}}}, nil
		}}
	})
	_, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind me of my booking", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r2","call_id":"call2","name":"photograph_room","arguments":"{}"}`)
	select {
	case <-running:
	case <-time.After(settleFor):
		s.FailNow("the tool never ran")
	}
	let()
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)

	s.Equal([]bool{false}, s.subagent.askedScheduled(), "the caller asked for this picture")
}

func (s *AgentSuite) TestNativeDelegationCancelsWorkWithoutEndingTheCall() {
	peer := s.joinNativeDelegating()
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	s.nativeFrame(peer, "conversation.item.create")
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r2","call_id":"call2","name":"cancel_skill","arguments":"{\"skill\":\"think\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.Eventually(func() bool { return countOf[TaskCancelled](s.reported()) == 1 }, time.Second, time.Millisecond)
	cancelled, _ := firstOf[TaskCancelled](s.reported())
	s.Equal(harness.ReasonDropped, cancelled.Reason)
	s.False(s.agent.delegating())
	s.False(s.agent.harness.Pending())
	s.Zero(countOf[Left](s.reported()))
}

func (s *AgentSuite) TestNativeDelegationSupersedesOldWorkAndStopsOnClose() {
	peer := s.joinNativeDelegating()
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Price six seats\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	s.nativeFrame(peer, "conversation.item.create")
	s.Eventually(func() bool { return len(s.subagent.requests()) == 1 }, time.Second, time.Millisecond)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r2","call_id":"call2","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Actually price eight seats instead\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.Eventually(func() bool { return len(s.subagent.requests()) == 2 && countOf[TaskCancelled](s.reported()) == 1 }, time.Second, time.Millisecond)
	cancelled, _ := firstOf[TaskCancelled](s.reported())
	s.Equal(harness.ReasonSuperseded, cancelled.Reason)
	s.True(s.agent.delegating())
	s.Require().NoError(s.agent.Close())
	s.Eventually(func() bool { return countOf[TaskCancelled](s.reported()) == 2 }, time.Second, time.Millisecond)
	s.False(s.agent.delegating())
	s.Zero(countOf[TaskSettled](s.reported()), "cancelled work must not be delivered as an answer")
}

func (s *AgentSuite) TestNativeToolCancellationBeforeExecution() {
	s.joinNativeDelegating()
	requested := harness.ToolRequested{TurnID: "r1", Call: llm.ToolCall{ID: "cancel-before-run", Name: delegateSkill, Arguments: `{"skill":"think","prompt":"Work out the booking total"}`}}
	ctx, cancel := s.agent.prepareTool(requested)
	source := make(chan sts.Event, 1)
	source <- sts.ToolCancel{CallIDs: []string{requested.Call.ID}}
	close(source)
	stopped := newPipeline(s.ctx, true)
	stopped.running.Add(1)
	s.agent.receiveSTS(stopped, source, make(chan sts.Event, 1))
	s.agent.executeTool(ctx, cancel, requested)
	s.Eventually(func() bool { return countOf[ToolRan](s.reported()) == 1 }, time.Second, time.Millisecond)
	result, _ := firstOf[ToolRan](s.reported())
	s.ErrorIs(result.Err, context.Canceled)
	s.False(s.agent.delegating())
	s.Zero(countOf[Delegated](s.reported()))
}

type liveRetryMode struct {
	rejectFrom     int
	withholdAt     int
	dropEvery      bool
	failFirstWrite bool
}

type failedAudioWriteSTS struct{ sts.STS }

func (failedAudioWriteSTS) ProcessAudio(sts.PcmData, sts.Participant) error {
	return errors.New("test audio write failure")
}

type liveRetryFixture struct {
	peers    chan *nativePeer
	attempts atomic.Int32
}

func (s *AgentSuite) joinNative(registry *stsrouter.Registry, opts Options) {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	router, err := stsrouter.New(stsrouter.Options{Config: config[routing.STS], Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	s.edge = newLoopbackEdge()
	opts.Edge = s.edge
	opts.STS = router
	agent, err := New(opts)
	s.Require().NoError(err)
	s.agent = agent
	s.events = collect(agent)
	s.T().Cleanup(func() { _ = agent.Close() })
	s.Require().NoError(agent.Join(s.ctx))
}

func (s *AgentSuite) newLiveRetryFixture(mode liveRetryMode, tools harness.Tools, runner ToolRunner) *liveRetryFixture {
	fixture := &liveRetryFixture{peers: make(chan *nativePeer, 8)}
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		var start struct {
			Session json.RawMessage `json:"session"`
		}
		if conn.ReadJSON(&start) != nil {
			return
		}
		attempt := int(fixture.attempts.Add(1))
		if mode.withholdAt == attempt {
			_, _, _ = conn.ReadMessage()
			return
		}
		if mode.rejectFrom > 0 && attempt >= mode.rejectFrom {
			_ = conn.WriteJSON(map[string]any{"type": "error", "error": map[string]any{"code": "rejected", "message": "setup rejected"}})
			return
		}
		if conn.WriteJSON(map[string]any{"type": "session.started", "session": map[string]any{"id": "test-live", "model": "gpt-live-1"}}) != nil {
			return
		}
		peer := &nativePeer{conn: conn, frames: make(chan map[string]json.RawMessage, 64), setup: start.Session}
		select {
		case fixture.peers <- peer:
		default:
		}
		if mode.dropEvery {
			return
		}
		for {
			var frame map[string]json.RawMessage
			if conn.ReadJSON(&frame) != nil {
				return
			}
			peer.frames <- frame
		}
	}))
	s.T().Cleanup(server.Close)
	registry := stsrouter.NewRegistry()
	registry.Register("openai", func(spec routing.Spec) (sts.STS, error) {
		session, err := openailive.New(openailive.Options{
			APIKey: "test-key", URL: "ws" + strings.TrimPrefix(server.URL, "http"),
			Model: spec.Model, Instructions: spec.STS.Instructions,
			ReplyGap: time.Minute, HandshakeTimeout: 5 * time.Second,
		})
		if err != nil {
			return nil, err
		}
		if mode.failFirstWrite && fixture.attempts.Load() == 0 {
			return failedAudioWriteSTS{STS: session}, nil
		}
		return session, nil
	})
	s.joinNative(registry, Options{
		CustomerID: "acme", STSTarget: "openai/gpt-live-1",
		Tools: tools, ToolRunner: runner,
	})
	return fixture
}

func (s *AgentSuite) nextNativePeer(peers <-chan *nativePeer) *nativePeer {
	select {
	case peer := <-peers:
		return peer
	case <-time.After(settleFor):
		s.FailNow("the live session did not connect")
		return nil
	}
}

func (s *AgentSuite) TestLosingTheLiveSocketRecoversTheConversationAndAudio() {
	fixture := s.newLiveRetryFixture(liveRetryMode{}, harness.Tools{}, nil)
	oldPeer := s.nextNativePeer(fixture.peers)
	oldSession := s.agent.speech()
	_, err := s.agent.RespondTo(s.ctx, "The delivery is for 12 Harbor Road", nil, options.LLM{})
	s.Require().NoError(err)
	s.nativeSend(oldPeer, `{"type":"session.output_audio.delta","delta":"6APQBw=="}`)
	s.nativeSend(oldPeer, `{"type":"session.output_transcript.delta","delta":"Your order is on its way."}`)
	s.Require().Eventually(func() bool {
		return len(s.edge.heard()) == 1 && countOf[ResponseDelta](s.events.seen()) == 1
	}, settleFor, time.Millisecond)
	oldReply, _ := firstOf[Responding](s.events.seen())
	s.Equal([]int16{1000, 2000}, s.edge.heard()[0].Samples)
	s.nativeSend(oldPeer, `{"type":"session.input_transcript.delta","delta":"unfinished caller words"}`)
	s.Eventually(func() bool {
		for _, event := range s.events.seen() {
			if heard, ok := event.(Hearing); ok && strings.Contains(heard.Text, "unfinished caller words") {
				return true
			}
		}
		return false
	}, settleFor, time.Millisecond)
	_ = oldPeer.conn.Close()

	newPeer := s.nextNativePeer(fixture.peers)
	s.Equal(int32(2), fixture.attempts.Load())
	s.Eventually(func() bool {
		return !s.agent.switching.Load() && s.agent.speech() != oldSession
	}, settleFor, time.Millisecond)
	var setup struct {
		Instructions string `json:"instructions"`
	}
	s.Require().NoError(json.Unmarshal(newPeer.setup, &setup))
	s.Contains(setup.Instructions, "Caller: The delivery is for 12 Harbor Road")
	s.Contains(setup.Instructions, "You: Your order is on its way.")
	s.Contains(setup.Instructions, "Caller: unfinished caller words")
	s.Empty(s.edge.heard(), "recovery drops queued audio from the old reply")

	pcm := audio.PcmData{Samples: []int16{5, 6, 7}, SampleRate: sts.InputSampleRate, Channels: 1}
	s.edge.inbound <- InboundAudio{Audio: pcm, Participant: stt.Participant{ID: "caller"}}
	frame := s.nativeFrame(newPeer, "session.input_audio.append")
	encoded := strings.Trim(string(frame["audio"]), `"`)
	data, err := base64.StdEncoding.DecodeString(encoded)
	s.Require().NoError(err)
	s.Equal(audio.Resample(pcm, openailive.SampleRate, 1).Bytes(), data)

	before := len(s.events.seen())
	s.nativeSend(newPeer, `{"type":"session.output_audio.delta","delta":"uAugDw=="}`)
	s.Require().Eventually(func() bool {
		return len(s.edge.heard()) == 1 && countOf[Responding](s.events.seen()[before:]) == 1
	}, settleFor, time.Millisecond)
	newReply, ok := firstOf[Responding](s.events.seen()[before:])
	s.Require().True(ok)
	s.NotEqual(oldReply.TurnID, newReply.TurnID)
	s.Equal([]int16{3000, 4000}, s.edge.heard()[0].Samples)
	s.Zero(countOf[Left](s.events.seen()))
	s.Zero(countOf[Error](s.events.seen()), "a recovered socket loss is not an agent error")
	s.Equal(int32(2), fixture.attempts.Load())
	_ = newPeer.conn.Close()
	thirdPeer := s.nextNativePeer(fixture.peers)
	s.Require().NoError(json.Unmarshal(thirdPeer.setup, &setup))
	s.Equal(1, strings.Count(setup.Instructions, "Caller: unfinished caller words"))
	s.Equal(int32(3), fixture.attempts.Load())
}

func (s *AgentSuite) TestFailedNativeReplacementSetupsExhaustTheCallBudget() {
	fixture := s.newLiveRetryFixture(liveRetryMode{rejectFrom: 2}, harness.Tools{}, nil)
	_ = s.nextNativePeer(fixture.peers).conn.Close()
	select {
	case <-s.events.done:
	case <-time.After(settleFor):
		s.FailNow("the agent did not leave after exhausting native retries")
	}
	events := s.events.seen()
	s.Equal(1, countOf[Left](events))
	s.Equal(1, countOf[Error](events))
	s.Equal(int32(4), fixture.attempts.Load())
}

func (s *AgentSuite) TestRepeatedNativeConnectAndDropUsesTheSameCallBudget() {
	fixture := s.newLiveRetryFixture(liveRetryMode{dropEvery: true}, harness.Tools{}, nil)
	select {
	case <-s.events.done:
	case <-time.After(settleFor):
		s.FailNow("the agent did not leave after repeated native drops")
	}
	s.Equal(int32(4), fixture.attempts.Load())
	s.Len(fixture.peers, 4)
	s.Equal(1, countOf[Left](s.events.seen()))
}

func (s *AgentSuite) TestCleanNativeCloseDoesNotRetry() {
	fixture := s.newLiveRetryFixture(liveRetryMode{}, harness.Tools{}, nil)
	peer := s.nextNativePeer(fixture.peers)
	s.nativeSend(peer, `{"type":"session.closed","reason":"finished"}`)
	select {
	case <-s.events.done:
	case <-time.After(settleFor):
		s.FailNow("the agent did not leave on a clean close")
	}
	s.Equal(int32(1), fixture.attempts.Load())
	s.Equal(1, countOf[Left](s.events.seen()))
}

func (s *AgentSuite) TestNonfatalNativeAPIErrorKeepsTheSession() {
	fixture := s.newLiveRetryFixture(liveRetryMode{}, harness.Tools{}, nil)
	peer := s.nextNativePeer(fixture.peers)
	s.nativeSend(peer, `{"type":"error","error":{"code":"transient","message":"request rejected"}}`)
	s.Eventually(func() bool { return countOf[Error](s.events.seen()) == 1 }, settleFor, time.Millisecond)
	s.nativeSend(peer, `{"type":"session.output_audio.delta","delta":"6APQBw=="}`)
	s.Eventually(func() bool { return len(s.edge.heard()) == 1 }, settleFor, time.Millisecond)
	s.Equal(1, int(fixture.attempts.Load()))
	s.Zero(countOf[Left](s.events.seen()))
}

func (s *AgentSuite) TestANativeWriteFailureRecoversWithoutAReadError() {
	fixture := s.newLiveRetryFixture(liveRetryMode{failFirstWrite: true}, harness.Tools{}, nil)
	_ = s.nextNativePeer(fixture.peers)
	pcm := audio.PcmData{Samples: []int16{5, 6, 7}, SampleRate: sts.InputSampleRate, Channels: 1}
	s.edge.inbound <- InboundAudio{Audio: pcm, Participant: stt.Participant{ID: "caller"}}

	peer := s.nextNativePeer(fixture.peers)
	s.Equal(int32(2), fixture.attempts.Load())
	s.Require().Eventually(func() bool { return !s.agent.switching.Load() }, settleFor, time.Millisecond)
	s.edge.inbound <- InboundAudio{Audio: pcm, Participant: stt.Participant{ID: "caller"}}
	frame := s.nativeFrame(peer, "session.input_audio.append")
	encoded := strings.Trim(string(frame["audio"]), `"`)
	data, err := base64.StdEncoding.DecodeString(encoded)
	s.Require().NoError(err)
	s.Equal(audio.Resample(pcm, openailive.SampleRate, 1).Bytes(), data)
	s.Equal(int32(2), fixture.attempts.Load())
	s.Zero(countOf[Error](s.events.seen()))
	s.Zero(countOf[Left](s.events.seen()))
	_ = peer.conn.Close()
}

type heldNativeTool struct {
	started chan llm.ToolCall
	release chan struct{}
	runs    atomic.Int32
}

func (r *heldNativeTool) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	r.runs.Add(1)
	r.started <- call
	select {
	case <-r.release:
		return llm.TextParts("packed and ready"), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *AgentSuite) TestNativeRetryWaitsForAndReplaysACompletedTool() {
	tools := harness.Tools{Tools: []harness.Tool{{
		Name: "lookup_order", Description: "find the order",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"order": map[string]any{"type": "string"}}},
	}}}
	runner := &heldNativeTool{started: make(chan llm.ToolCall, 1), release: make(chan struct{})}
	fixture := s.newLiveRetryFixture(liveRetryMode{}, tools, runner)
	oldPeer := s.nextNativePeer(fixture.peers)
	s.nativeSend(oldPeer, `{"type":"response.event","delegation_id":"reply","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"old-call","name":"lookup_order","arguments":"{\"order\":\"A42\"}"}}}`)
	select {
	case call := <-runner.started:
		s.Equal("old-call", call.ID)
	case <-time.After(settleFor):
		s.FailNow("the registered tool did not start")
	}
	_ = oldPeer.conn.Close()
	s.Eventually(func() bool { return s.agent.switching.Load() }, settleFor, time.Millisecond)
	select {
	case peer := <-fixture.peers:
		s.Fail("replacement setup completed before the old tool finished", peer)
	case <-time.After(200 * time.Millisecond):
	}
	s.Equal(int32(1), fixture.attempts.Load(), "no replacement setup began while the old tool was running")
	close(runner.release)
	newPeer := s.nextNativePeer(fixture.peers)
	s.Equal(int32(2), fixture.attempts.Load())
	var setup struct {
		Instructions string `json:"instructions"`
	}
	s.Require().NoError(json.Unmarshal(newPeer.setup, &setup))
	s.Contains(setup.Instructions, `Completed tool lookup_order({"order":"A42"}): packed and ready`)
	s.NotContains(setup.Instructions, "old-call")
	s.Equal(int32(1), runner.runs.Load())
	s.Eventually(func() bool { return countOf[ToolRan](s.events.seen()) == 1 }, settleFor, time.Millisecond)
	result, ok := firstOf[ToolRan](s.events.seen())
	s.Require().True(ok)
	s.Equal("packed and ready", result.Result)
	s.Zero(countOf[Error](s.events.seen()))
	s.Zero(countOf[Left](s.events.seen()))
}

func (s *AgentSuite) TestCloseCancelsAWithheldNativeReplacementHandshake() {
	fixture := s.newLiveRetryFixture(liveRetryMode{withholdAt: 2}, harness.Tools{}, nil)
	_ = s.nextNativePeer(fixture.peers).conn.Close()
	s.Require().Eventually(func() bool { return fixture.attempts.Load() == 2 }, settleFor, time.Millisecond)
	done := make(chan error, 1)
	go func() { done <- s.agent.Close() }()
	select {
	case err := <-done:
		s.NoError(err)
	// Under nativeRetrySetupTimeout (5s, native.go:22): past it the setup times out on its
	// own and Close returns even when it did not cancel the handshake.
	case <-time.After(time.Second):
		s.FailNow("Close did not cancel the withheld native handshake")
	}
	select {
	case <-s.events.done:
	case <-time.After(settleFor):
		s.FailNow("the event stream did not close after Close")
	}
	s.Equal(int32(2), fixture.attempts.Load())
}

func (s *AgentSuite) TestGeminiResumesInsideItsAdapterWithoutLeavingTheCall() {
	peers := make(chan *nativePeer, 3)
	var attempts atomic.Int32
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		attempts.Add(1)
		var start map[string]json.RawMessage
		if conn.ReadJSON(&start) != nil || conn.WriteJSON(map[string]any{"setupComplete": map[string]any{}}) != nil {
			return
		}
		peer := &nativePeer{conn: conn, setup: start["setup"], frames: make(chan map[string]json.RawMessage, 32)}
		peers <- peer
		for {
			var frame map[string]json.RawMessage
			if conn.ReadJSON(&frame) != nil {
				return
			}
			peer.frames <- frame
		}
	}))
	s.T().Cleanup(server.Close)
	registry := stsrouter.NewRegistry()
	registry.Register("gemini", func(spec routing.Spec) (sts.STS, error) {
		return gemini.New(gemini.Options{
			APIKey: "test-key", URL: "ws" + strings.TrimPrefix(server.URL, "http"),
			Model: spec.Model, Instructions: spec.STS.Instructions,
			InputTranscript: true, OutputTranscript: true, HandshakeTimeout: time.Second,
		})
	})
	s.joinNative(registry, Options{
		CustomerID: "acme",
		STSTarget:  "gemini/" + gemini.DefaultModel,
	})
	first := s.nextNativePeer(peers)
	if len(first.setup) == 0 {
		s.FailNow("Gemini setup was missing")
	}
	s.nativeSend(first, `{"sessionResumptionUpdate":{"newHandle":"resume-handle","resumable":true}}`)
	s.Require().NoError(first.conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseGoingAway, "connection dropped"),
		time.Now().Add(time.Second),
	))
	second := s.nextNativePeer(peers)
	var setup struct {
		SessionResumption struct {
			Handle string `json:"handle"`
		} `json:"sessionResumption"`
	}
	s.Require().NoError(json.Unmarshal(second.setup, &setup))
	s.Equal("resume-handle", setup.SessionResumption.Handle)

	s.nativeSend(second, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQACAA=="}}]}}}`)
	s.Require().Eventually(func() bool { return len(s.edge.heard()) == 1 }, settleFor, time.Millisecond)
	s.Equal([]int16{1, 2}, s.edge.heard()[0].Samples)
	pcm := audio.PcmData{Samples: []int16{11, 12, 13}, SampleRate: sts.InputSampleRate, Channels: 1}
	s.edge.inbound <- InboundAudio{Audio: pcm, Participant: stt.Participant{ID: "caller"}}
	var frame map[string]json.RawMessage
	select {
	case frame = <-second.frames:
	case <-time.After(settleFor):
		s.FailNow("the resumed Gemini session did not receive caller audio")
	}
	var realtime struct {
		Audio struct {
			Data string `json:"data"`
		} `json:"audio"`
	}
	s.Require().NoError(json.Unmarshal(frame["realtimeInput"], &realtime))
	data, err := base64.StdEncoding.DecodeString(realtime.Audio.Data)
	s.Require().NoError(err)
	s.Equal(pcm.Bytes(), data)
	s.Equal(int32(2), attempts.Load(), "the adapter resumed instead of opening a replacement agent session")
	s.Zero(countOf[Left](s.events.seen()))
	s.Zero(countOf[Error](s.events.seen()))
}

// replayNative plays a speech-to-speech provider's events through the agent and returns
// what it reported once every reply has been spoken.
func (s *AgentSuite) replayNative(events ...sts.Event) []Event {
	agent, err := New(Options{CustomerID: "acme", Edge: newLoopbackEdge(), STS: &stsrouter.Router{}, STSTarget: "openai/gpt-realtime-2"})
	s.Require().NoError(err)
	agent.ctx, agent.cancel = context.WithCancel(s.ctx)
	s.T().Cleanup(func() { agent.Close() })
	seen := collect(agent)
	ordered := make(chan sts.Event, len(events))
	agent.pipe = newPipeline(agent.ctx, true)
	agent.pipe.running.Add(1)
	go agent.consumeSTS(agent.pipe, nil, ordered)
	s.T().Cleanup(func() { close(ordered) })

	replies := 0
	for _, event := range events {
		if _, ok := event.(sts.ResponseComplete); ok {
			replies++
		}
		ordered <- event
	}
	s.eventually(func() bool { return countOf[Spoke](seen.seen()) == replies }, "the replies were never spoken")
	return seen.seen()
}

// turnFor is the measured turn of a reply, if it was measured.
func turnFor(events []Event, reply string) (Turn, bool) {
	for _, event := range events {
		if turn, ok := event.(Turn); ok && turn.TurnID == reply {
			return turn, true
		}
	}
	return Turn{}, false
}

var nativePCM = audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24000, Channels: 1}

// The order openairealtime reports a caller's turn in (TestTheReplyIsTimedFromWhenTheCallerStopped).
func (s *AgentSuite) TestAnOpenAIRealtimeReplyIsTimedFromTheStopBeforeIt() {
	stopped := time.Now().Add(-50 * time.Millisecond)
	events := s.replayNative(
		sts.SpeechStarted{At: stopped.Add(-time.Second)},
		sts.SpeechStopped{At: stopped},
		sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()},
		sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: nativePCM},
		sts.ResponseComplete{ResponseID: "r1", Generation: 1},
	)
	turn, ok := turnFor(events, "r1")
	s.Require().True(ok, "the caller was waiting on this reply")
	s.GreaterOrEqual(turn.RoundtripMs, 50.0)
}

// The order gemini and openailive report a caller's turn in: the transcript settles before
// the stop, and the stop before the reply (TestAReplyWhoseWordsComeFirstStillFollowsTheCallerStopping
// in both packages).
func (s *AgentSuite) TestAGeminiOrOpenAILiveReplyIsTimedFromTheStopBeforeIt() {
	stopped := time.Now().Add(-50 * time.Millisecond)
	events := s.replayNative(
		sts.SpeechStarted{At: stopped.Add(-time.Second)},
		sts.InputTranscript{Mode: stt.ModeDelta, Text: "hello"},
		sts.InputTranscript{Mode: stt.ModeFinal, Text: "hello"},
		sts.SpeechStopped{At: stopped},
		sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()},
		sts.OutputTranscript{ResponseID: "r1", Mode: stt.ModeDelta, Text: "Hi."},
		sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: nativePCM},
		sts.ResponseComplete{ResponseID: "r1", Generation: 1},
	)
	turn, ok := turnFor(events, "r1")
	s.Require().True(ok, "the caller was waiting on this reply")
	s.GreaterOrEqual(turn.RoundtripMs, 50.0)
}

// A Prompt's reply can be playing when the caller stops. That stop is not what the next
// reply answers, so the next reply is not timed from it.
func (s *AgentSuite) TestAStopHeardDuringANativeReplyIsNotTheNextReplysWait() {
	events := s.replayNative(
		sts.ResponseStarted{ResponseID: "prompted", Generation: 1, At: time.Now()},
		sts.AudioChunk{ResponseID: "prompted", Generation: 1, Audio: nativePCM},
		sts.SpeechStarted{At: time.Now()},
		sts.SpeechStopped{At: time.Now()},
		sts.ResponseComplete{ResponseID: "prompted", Generation: 1},
		sts.ResponseStarted{ResponseID: "next", Generation: 2, At: time.Now()},
		sts.AudioChunk{ResponseID: "next", Generation: 2, Audio: nativePCM},
		sts.ResponseComplete{ResponseID: "next", Generation: 2},
	)
	_, measured := turnFor(events, "next")
	s.False(measured, "nobody was waiting on the next reply")
}
