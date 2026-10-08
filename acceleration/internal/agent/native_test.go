package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/openairealtime"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
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
	go agent.consumeSTS(agent.pipe, ordered)
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
}

func (s *AgentSuite) joinNativeDelegating() *nativePeer {
	connections := make(chan *websocket.Conn, 1)
	frames := make(chan map[string]json.RawMessage, 32)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		if conn.WriteJSON(map[string]any{"type": "session.created", "session": map[string]any{}}) != nil {
			return
		}
		var setup map[string]json.RawMessage
		if conn.ReadJSON(&setup) != nil {
			return
		}
		frames <- setup
		if conn.WriteJSON(map[string]any{"type": "session.updated", "session": map[string]any{}}) != nil {
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
		return openairealtime.New(openairealtime.Options{
			Vendor: openairealtime.OpenAI, URL: "ws" + strings.TrimPrefix(server.URL, "http"), APIKey: "test",
			Model: spec.Model, Tools: spec.Tools, Instructions: spec.STS.Instructions, InputTranscript: true,
		})
	})
	router, err := stsrouter.New(stsrouter.Options{Config: config[routing.STS], Registry: speech})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	s.edge = newLoopbackEdge()
	s.agent, err = New(Options{
		CustomerID: "acme", Edge: s.edge, STS: router, STSTarget: "openai/gpt-realtime-2",
		LLM: reasoner, SubagentTarget: "stub/stub-model",
		Skills: harness.Skills{Skills: []harness.Skill{{Name: "think", Description: "reason carefully", Instructions: "Solve it", Deadline: time.Minute}}},
	})
	s.Require().NoError(err)
	s.events = collect(s.agent)
	s.T().Cleanup(func() { s.agent.Close() })
	s.Require().NoError(s.agent.Join(s.ctx))
	return &nativePeer{conn: <-connections, frames: frames}
}

func (s *AgentSuite) nativeFrame(peer *nativePeer, kind string) map[string]json.RawMessage {
	for {
		select {
		case frame := <-peer.frames:
			if string(frame["type"]) == strconv.Quote(kind) {
				return frame
			}
		case <-time.After(3 * time.Second):
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
	turnID, err := s.agent.RespondTo(s.ctx, "My booking is for six people", nil)
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
