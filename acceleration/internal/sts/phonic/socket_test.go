package phonic

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
)

// PhonicSocketSuite exercises the provider against a Phonic socket that is answering, which
// is where the config, the audio framing and the mid-session frames live.
type PhonicSocketSuite struct {
	suite.Suite
}

func TestPhonicSocketSuite(t *testing.T) {
	suite.Run(t, new(PhonicSocketSuite))
}

// fakePhonic accepts one conversation. It answers the config with ready, or with the refusal
// it was given, and keeps the config and how it was authenticated.
type fakePhonic struct {
	server  *httptest.Server
	conns   chan *websocket.Conn
	configs chan map[string]any
	auth    chan string
	refuse  string
	done    chan struct{}
}

func newFakePhonic(refuse string) *fakePhonic {
	fake := &fakePhonic{
		conns:   make(chan *websocket.Conn, 1),
		configs: make(chan map[string]any, 1),
		auth:    make(chan string, 1),
		refuse:  refuse,
		done:    make(chan struct{}),
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
		var configured map[string]any
		if err := json.Unmarshal(raw, &configured); err != nil {
			return
		}
		fake.configs <- configured

		if fake.refuse != "" {
			_ = conn.WriteMessage(websocket.TextMessage, []byte(fake.refuse))
			_ = conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(4204, "Invalid parameters"))
			return
		}
		for _, frame := range []string{
			`{"type":"ready_to_start_conversation"}`,
			`{"type":"conversation_created","conversation_id":"conv_123"}`,
		} {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(frame)); err != nil {
				return
			}
		}
		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakePhonic) endpoint() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakePhonic) close() {
	close(f.done)
	f.server.Close()
}

func (s *PhonicSocketSuite) provider(fake *fakePhonic, settings Options) *STS {
	settings.URL = fake.endpoint()
	settings.APIKey = "test-key"
	settings.HandshakeTimeout = 5 * time.Second
	provider, err := New(settings)
	s.Require().NoError(err)
	return provider
}

// connect returns a started provider and the server side of its connection.
func (s *PhonicSocketSuite) connect(fake *fakePhonic, settings Options) (*STS, *websocket.Conn) {
	provider := s.provider(fake, settings)
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

// nextFrame reads the next frame the provider sent upstream.
func (s *PhonicSocketSuite) nextFrame(conn *websocket.Conn) map[string]any {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	var frame map[string]any
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

// collect reads events until the channel closes or the wait runs out.
func (s *PhonicSocketSuite) collect(provider *STS, wait time.Duration) []sts.Event {
	var events []sts.Event
	deadline := time.After(wait)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
		case <-deadline:
			return events
		}
	}
}

func (s *PhonicSocketSuite) TestTheConfigNamesTheModelTheAudioTheToolsAndTheLimits() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, _ := s.connect(fake, Options{
		Voice:        "grant",
		Instructions: "Be brief.",
		Languages:    []string{"fr", "en"},
		Tools:        []llm.Tool{{Name: "get_weather", Description: "Weather.", Parameters: map[string]any{"type": "object"}}, {Name: "hang_up", Description: "End the call."}},
	})
	defer provider.Close()

	s.Equal("Bearer test-key", <-fake.auth)
	configured := <-fake.configs
	s.Equal("config", configured["type"])
	s.Equal("phonic-1", configured["model"])
	s.Equal("Be brief.", configured["system_prompt"])
	s.Equal("grant", configured["voice_id"])
	s.Equal("pcm_16000", configured["input_format"], "the router's audio goes up as it is")
	s.Equal("pcm_24000", configured["output_format"])
	s.Equal(true, configured["stream_ahead_of_real_time"])
	s.Equal(float64(300), configured["websocket_timeout_sec"])
	s.Equal(float64(3600), configured["no_input_end_conversation_sec"], "a quiet caller is not hung up on after three minutes")
	s.Equal("fr", configured["default_language"])
	s.Equal([]any{"en"}, configured["additional_languages"])

	tools := configured["tools"].([]any)
	s.Require().Len(tools, 2)
	s.Equal(map[string]any{
		"type":                        "custom_websocket",
		"tool_call_output_timeout_ms": float64(180000),
		"tool_schema": map[string]any{"type": "function", "function": map[string]any{
			"name": "get_weather", "description": "Weather.", "parameters": map[string]any{"type": "object"}, "strict": true,
		}},
	}, tools[0])
	parameterless := tools[1].(map[string]any)["tool_schema"].(map[string]any)["function"].(map[string]any)
	s.Equal(map[string]any{"type": "object", "properties": map[string]any{}}, parameterless["parameters"], "Phonic requires a parameters object")
}

func (s *PhonicSocketSuite) TestARefusedConfigFailsStartWithWhatPhonicSaid() {
	fake := newFakePhonic(`{"type":"error","error":{"message":"Invalid parameters","code":"user_message_validation_failed"},"param_errors":{"no_such_field":"Unrecognized key"}}`)
	defer fake.close()
	provider := s.provider(fake, Options{})

	err := provider.Start(context.Background())
	s.ErrorContains(err, "user_message_validation_failed")
	s.ErrorContains(err, "no_such_field")
}

func (s *PhonicSocketSuite) TestTheCallersAudioGoesUpInPiecesPhonicAccepts() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.ProcessAudio(testaudio.Silence(100), sts.Participant{ID: "caller"}))
	for range 5 {
		frame := s.nextFrame(conn)
		s.Equal("audio_chunk", frame["type"])
		raw, err := base64.StdEncoding.DecodeString(frame["audio"].(string))
		s.Require().NoError(err)
		s.Len(raw, 640, "20 ms at 16 kHz, under Phonic's 40 ms ceiling")
	}
}

func (s *PhonicSocketSuite) TestATypedTurnAsksForAReplyAndItsEchoIsNotTheCallerSpeaking() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.SendText("What is the weather in Paris?", sts.Participant{ID: "caller"}))
	s.Equal(map[string]any{"type": "generate_reply", "user_message": "What is the weather in Paris?"}, s.nextFrame(conn))

	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"input_text","language":"en","text":"What is the weather in Paris?"}`)))
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"input_text","language":"en","text":"hello"}`)))

	var heard []string
	for _, event := range s.collect(provider, 300*time.Millisecond) {
		if transcript, ok := event.(sts.InputTranscript); ok {
			heard = append(heard, transcript.Text)
		}
	}
	s.Equal([]string{"hello"}, heard)
}

func (s *PhonicSocketSuite) TestPromptAsksForAReplyGuidedByASystemMessage() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Prompt("Greet the caller."))
	s.Equal(map[string]any{"type": "generate_reply", "system_message": "Greet the caller."}, s.nextFrame(conn))
}

func (s *PhonicSocketSuite) TestAToolsAnswerOrItsFailureGoesBackAgainstItsCall() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Answer("tool_call_1", `{"forecast":"sunny"}`, nil))
	s.Equal(map[string]any{"type": "tool_call_output", "tool_call_id": "tool_call_1", "output": `{"forecast":"sunny"}`}, s.nextFrame(conn))

	s.Require().NoError(provider.Answer("tool_call_2", "", context.DeadlineExceeded))
	s.Equal(map[string]any{"type": "tool_call_output", "tool_call_id": "tool_call_2", "output": "Error: context deadline exceeded"}, s.nextFrame(conn))
}

func (s *PhonicSocketSuite) TestInstructionsAndToolsChangeMidSession() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.SetInstructions("End with goodbye."))
	s.Equal(map[string]any{"type": "update_system_prompt", "system_prompt": "End with goodbye."}, s.nextFrame(conn))

	s.Require().NoError(provider.SetTools([]llm.Tool{{Name: "get_weather", Description: "Weather.", Parameters: map[string]any{"type": "object"}}}))
	updated := s.nextFrame(conn)
	s.Equal("update_tools_subset", updated["type"])
	s.Len(updated["tools"], 1)

	s.Require().NoError(provider.SetTools(nil))
	s.Equal(map[string]any{"type": "update_tools_subset", "tools": []any{}}, s.nextFrame(conn), "no tools is an empty list, not a missing one")
}

func (s *PhonicSocketSuite) TestClosingReportsTheSessionsSecondsAsFinal() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, _ := s.connect(fake, Options{})

	s.Require().NoError(provider.Close())
	var final []sts.SessionUsage
	for _, event := range s.collect(provider, time.Second) {
		if usage, ok := event.(sts.SessionUsage); ok && usage.Final {
			final = append(final, usage)
		}
	}
	s.Len(final, 1, "the session's seconds are what Phonic bills, and the router waits for the last count")
}

func (s *PhonicSocketSuite) TestTheModelHangingUpIsACleanEnd() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"assistant_ended_conversation"}`)))
	events := s.collect(provider, 2*time.Second)

	var disconnected []sts.Disconnected
	var failed, final bool
	for _, event := range events {
		switch typed := event.(type) {
		case sts.Disconnected:
			disconnected = append(disconnected, typed)
		case sts.Error:
			failed = true
		case sts.SessionUsage:
			final = final || typed.Final
		}
	}
	s.Require().Len(disconnected, 1)
	s.True(disconnected[0].Clean)
	s.Equal(reasonEndedConversation, disconnected[0].Reason)
	s.False(failed, "the socket closing after the model hung up is not a failure")
	s.True(final)
}

func (s *PhonicSocketSuite) TestPhonicClosingWithItsOwnCodeIsAFailure() {
	fake := newFakePhonic("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	s.Require().NoError(conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(4505, "max conversation duration reached")))
	var fatal []sts.Error
	for _, event := range s.collect(provider, 2*time.Second) {
		if failure, ok := event.(sts.Error); ok && failure.Fatal {
			fatal = append(fatal, failure)
		}
	}
	s.Require().Len(fatal, 1)
	s.ErrorContains(fatal[0], "max conversation duration reached")
}
