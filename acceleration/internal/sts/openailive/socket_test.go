package openailive

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

// OpenAILiveSocketSuite exercises the provider against a Live API that is answering, which
// is where session.start, the audio encoding and the backend round trips live.
type OpenAILiveSocketSuite struct {
	suite.Suite
}

func TestOpenAILiveSocketSuite(t *testing.T) {
	suite.Run(t, new(OpenAILiveSocketSuite))
}

// fakeLive is a Live API that accepts one session. It answers session.start itself, or
// refuses it when told to, and keeps what it was sent and how it was authenticated.
type fakeLive struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	starts chan map[string]any
	auth   chan string
	refuse string
	done   chan struct{}
}

func newFakeLive(refuse string) *fakeLive {
	fake := &fakeLive{
		conns:  make(chan *websocket.Conn, 1),
		starts: make(chan map[string]any, 1),
		auth:   make(chan string, 1),
		refuse: refuse,
		done:   make(chan struct{}),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.auth <- r.Header.Get("Authorization")
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}

		_, raw, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var start map[string]any
		if err := json.Unmarshal(raw, &start); err != nil {
			return
		}
		fake.starts <- start

		reply := `{"type":"session.started","session":{"id":"sess_123","model":"gpt-live-1"}}`
		if fake.refuse != "" {
			reply = fake.refuse
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(reply)); err != nil {
			return
		}
		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakeLive) endpoint() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeLive) close() {
	close(f.done)
	f.server.Close()
}

// connect returns a started provider and the server side of its connection.
func (s *OpenAILiveSocketSuite) connect(fake *fakeLive, settings Options) (*STS, *websocket.Conn) {
	settings.URL = fake.endpoint()
	settings.APIKey = "test-key"
	settings.HandshakeTimeout = 5 * time.Second
	provider, err := New(settings)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	select {
	case conn := <-fake.conns:
		return provider, conn
	case <-time.After(5 * time.Second):
		s.FailNow("the provider never connected")
		return nil, nil
	}
}

// nextFrame reads the next message the provider sent upstream.
func (s *OpenAILiveSocketSuite) nextFrame(conn *websocket.Conn) map[string]any {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)

	var frame map[string]any
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

func (s *OpenAILiveSocketSuite) TestSessionStartNamesTheModelTheAudioAndTheBackend() {
	fake := newFakeLive("")
	defer fake.close()
	provider, _ := s.connect(fake, Options{
		Voice:        "quartz",
		Instructions: "Be brief.",
		Tools:        []llm.Tool{{Name: "get_weather", Description: "Weather.", Parameters: map[string]any{"type": "object"}}},
		Backend:      "gpt-5.6-luna",
	})
	defer func() { _ = provider.Close() }()

	s.Equal("Bearer test-key", <-fake.auth)
	start := <-fake.starts
	s.Equal("session.start", start["type"])
	session := start["session"].(map[string]any)
	s.Equal("gpt-live-1", session["model"], "the model goes in the frame, not on the URL")
	s.Equal("Be brief.", session["instructions"])

	audioConfig := session["audio"].(map[string]any)
	s.Equal(map[string]any{"type": "audio/pcm", "rate": float64(24000)}, audioConfig["format"])
	s.Equal(map[string]any{"voice": "quartz"}, audioConfig["output"])

	delegated := session["delegation"].(map[string]any)
	s.Equal("responses", delegated["type"], "tools only reach the model through a Responses backend")
	responses := delegated["responses"].(map[string]any)
	s.Equal("gpt-5.6-luna", responses["model"])
	s.Equal("Be brief.", responses["instructions"], "the backend is the one that chooses the tools")
	s.Equal("auto", responses["tool_choice"])
	s.Equal(false, responses["parallel_tool_calls"])
	tools := responses["tools"].([]any)
	s.Require().Len(tools, 1)
	s.Equal(map[string]any{
		"type":        "function",
		"name":        "get_weather",
		"description": "Weather.",
		"parameters":  map[string]any{"type": "object"},
	}, tools[0], "the Responses shape is flat")
}

func (s *OpenAILiveSocketSuite) TestASessionAskingForNothingExtraSendsNothingExtra() {
	fake := newFakeLive("")
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	session := (<-fake.starts)["session"].(map[string]any)
	s.NotContains(session, "instructions")
	s.NotContains(session["audio"].(map[string]any), "output", "no voice, so the model's own")
	responses := session["delegation"].(map[string]any)["responses"].(map[string]any)
	s.Equal(map[string]any{"model": DefaultBackend}, responses, "a backend with no tools is told nothing about them")
}

func (s *OpenAILiveSocketSuite) TestARefusedSessionFailsStart() {
	fake := newFakeLive(`{"type":"error","error":{"type":"invalid_request_error","code":"unknown_parameter","message":"Unknown parameter: 'session.turn_detection'."}}`)
	defer fake.close()
	provider, err := New(Options{APIKey: "test-key", URL: fake.endpoint(), HandshakeTimeout: 5 * time.Second})
	s.Require().NoError(err)

	err = provider.Start(context.Background())
	s.ErrorContains(err, "session rejected")
	s.ErrorContains(err, "unknown_parameter")
}

func (s *OpenAILiveSocketSuite) TestTheCallersAudioGoesUpAtTheSessionRate() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	pcm := sts.PcmData{Samples: make([]int16, 1600), SampleRate: sts.InputSampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, sts.Participant{ID: "alice"}))

	frame := s.nextFrame(conn)
	s.Equal("session.input_audio.append", frame["type"])
	decoded, err := base64.StdEncoding.DecodeString(frame["audio"].(string))
	s.Require().NoError(err)
	s.Len(decoded, len(audio.Resample(pcm, SampleRate, 1).Bytes()), "100ms at 16 kHz goes up as 100ms at 24 kHz")
}

func (s *OpenAILiveSocketSuite) TestAToolAnswerContinuesTheBackend() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.Answer("call_1", `{"forecast":"sunny"}`, nil))
	result := s.nextFrame(conn)
	s.Equal("response.item.create", result["type"])
	s.Equal(map[string]any{"type": "function_call_output", "call_id": "call_1", "output": `{"forecast":"sunny"}`}, result["item"])
	s.Equal(map[string]any{"type": "response.create"}, s.nextFrame(conn), "a result alone does not carry the backend on")
}

func (s *OpenAILiveSocketSuite) TestATypedTurnGoesToTheBackendAndAPromptToTheLiveModel() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.SendText("My order number is A0042.", sts.Participant{ID: "alice"}))
	typed := s.nextFrame(conn)
	s.Equal("response.item.create", typed["type"])
	s.Equal(map[string]any{
		"type":    "message",
		"role":    "user",
		"content": []any{map[string]any{"type": "input_text", "text": "My order number is A0042."}},
	}, typed["item"])
	s.Equal("response.create", s.nextFrame(conn)["type"])

	s.Require().NoError(provider.Prompt("Greet the caller."))
	prompt := s.spokenTo(conn)
	s.Equal("session.instructions.append", prompt["type"])
	s.Contains(prompt, "delegation_id", "required even when it names no delegation")
	s.Nil(prompt["delegation_id"])
	s.True(strings.HasPrefix(prompt["content"].(string), "Greet the caller."))

	// An instruction only steers the speech a turn would have produced. The commentary is
	// what starts the model talking when no turn is under way.
	commentary := s.spokenTo(conn)
	s.Equal("session.commentary.append", commentary["type"])
	s.Nil(commentary["delegation_id"])
	s.Equal("Greet the caller.", commentary["content"])
}

// A Live session injects an append at a point on a clock that only input audio advances, so
// nothing appended is spoken until silence carries the session there.
func (s *OpenAILiveSocketSuite) TestSilenceCarriesTheSessionClockUntilTheModelSpeaks() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.Prompt("Greet the caller."))
	s.Equal("session.instructions.append", s.spokenTo(conn)["type"])
	s.Equal("session.commentary.append", s.spokenTo(conn)["type"])

	silence := s.nextFrame(conn)
	s.Require().Equal("session.input_audio.append", silence["type"], "silence should follow a prompt")
	decoded, err := base64.StdEncoding.DecodeString(silence["audio"].(string))
	s.Require().NoError(err)
	s.Len(decoded, SampleRate/50*2, "20ms of silence at the session rate")
	for _, sample := range decoded {
		s.Zero(sample, "the filler has to be silent")
	}

	// Once the model is talking, its own audio is the clock and the filler stands down.
	var reply serverEvent
	s.Require().NoError(json.Unmarshal([]byte(spoke(240)), &reply))
	provider.handleMessage(reply)
	s.Eventually(func() bool {
		provider.mu.Lock()
		defer provider.mu.Unlock()
		return !provider.carrying
	}, time.Second, 10*time.Millisecond, "the filler should stop when the model speaks")
}

// spokenTo is the next frame that is not the silence filling the gap, which may arrive at any
// point once a prompt has started the clock.
func (s *OpenAILiveSocketSuite) spokenTo(conn *websocket.Conn) map[string]any {
	for {
		frame := s.nextFrame(conn)
		if frame["type"] != "session.input_audio.append" {
			return frame
		}
	}
}

func (s *OpenAILiveSocketSuite) TestNewToolsReplaceTheWholeDelegation() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{Instructions: "Be brief."})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.SetTools([]llm.Tool{{Name: "get_time"}}))
	update := s.nextFrame(conn)
	s.Equal("session.update", update["type"])
	session := update["session"].(map[string]any)
	s.Equal([]string{"delegation"}, keys(session), "only the delegation can change after startup")
	responses := session["delegation"].(map[string]any)["responses"].(map[string]any)
	s.Equal(DefaultBackend, responses["model"], "the delegation is replaced whole, so the model goes again")
	s.Equal("Be brief.", responses["instructions"])
	s.Equal("get_time", responses["tools"].([]any)[0].(map[string]any)["name"])
}

func (s *OpenAILiveSocketSuite) TestCloseAsksTheServerToFinishTheSession() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	s.Require().NoError(provider.Close())
	s.Equal(map[string]any{"type": "session.close"}, s.nextFrame(conn))
}

func keys(object map[string]any) []string {
	var found []string
	for key := range object {
		found = append(found, key)
	}
	return found
}
