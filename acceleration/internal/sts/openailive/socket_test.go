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
	provider, conn := s.connect(fake, Options{
		Voice:        "quartz",
		Instructions: "Be brief.",
		Tools:        []llm.Tool{{Name: "get_weather", Description: "Weather.", Parameters: map[string]any{"type": "object"}}},
		Backend:      "gpt-5.6-luna",
	})
	defer s.hangUp(provider, conn)

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

// sendMessage is Slack's MCP slack_send_message input schema, as on the text path (AI-969):
// two required properties and four optional ones.
var sendMessage = llm.Tool{
	Name:        "slack__slack_send_message",
	Description: "Sends a message to a Slack channel or user.",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"channel_id":       map[string]any{"type": "string"},
			"message":          map[string]any{"type": "string"},
			"thread_ts":        map[string]any{"type": "string"},
			"draft_id":         map[string]any{"type": "string"},
			"reply_broadcast":  map[string]any{"type": "boolean"},
			"unfurl_app_links": map[string]any{"type": "boolean"},
		},
		"required": []any{"channel_id", "message"},
	},
}

// backendTools are the tools a session.start or session.update frame gave the backend.
func backendTools(frame map[string]any) []any {
	session := frame["session"].(map[string]any)
	return session["delegation"].(map[string]any)["responses"].(map[string]any)["tools"].([]any)
}

// TestABackendToolsOptionalArgumentsStayOptional is AI-993: the backend is a Responses model,
// which turns a tool with strict left out into strict mode, so the model filled thread_ts with
// "" and reply_broadcast with false.
func (s *OpenAILiveSocketSuite) TestABackendToolsOptionalArgumentsStayOptional() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{Tools: []llm.Tool{sendMessage}})
	defer s.hangUp(provider, conn)

	started := backendTools(<-fake.starts)[0].(map[string]any)
	s.Equal(false, started["strict"], "an omitted strict is strict mode on the Responses API")
	s.Equal([]any{"channel_id", "message"}, started["parameters"].(map[string]any)["required"])

	s.Require().NoError(provider.SetTools([]llm.Tool{sendMessage}))
	updated := backendTools(s.nextFrame(conn))[0].(map[string]any)
	s.Equal(false, updated["strict"], "tools set mid-session follow the same rule")
}

// TestABackendToolWithEveryPropertyRequiredIsSentAsBefore leaves strict out, as on base:
// OpenAI's strict normalization takes nothing from a schema with no optional property.
func (s *OpenAILiveSocketSuite) TestABackendToolWithEveryPropertyRequiredIsSentAsBefore() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{Tools: []llm.Tool{{Name: "get_weather", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []string{"city"},
	}}}})
	defer s.hangUp(provider, conn)

	s.NotContains(backendTools(<-fake.starts)[0].(map[string]any), "strict")
}

// TestABackendToolDecodedFromJSONWithEveryPropertyRequiredIsSentAsBefore: required as []any, as
// a decoded MCP schema has it, counts as much as []string.
func (s *OpenAILiveSocketSuite) TestABackendToolDecodedFromJSONWithEveryPropertyRequiredIsSentAsBefore() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{Tools: []llm.Tool{{Name: "get_weather", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []any{"city"},
	}}}})
	defer s.hangUp(provider, conn)

	s.NotContains(backendTools(<-fake.starts)[0].(map[string]any), "strict")
}

func (s *OpenAILiveSocketSuite) TestASessionAskingForNothingExtraSendsNothingExtra() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer s.hangUp(provider, conn)

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
	defer s.hangUp(provider, conn)

	pcm := sts.PcmData{Samples: make([]int16, 1600), SampleRate: sts.InputSampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, sts.Participant{ID: "alice"}))

	frame := s.nextFrame(conn)
	s.Equal("session.input_audio.append", frame["type"])
	decoded, err := base64.StdEncoding.DecodeString(frame["audio"].(string))
	s.Require().NoError(err)
	s.Len(decoded, len(audio.Resample(pcm, SampleRate, 1).Bytes()), "100ms at 16 kHz goes up as 100ms at 24 kHz")
}

func (s *OpenAILiveSocketSuite) TestCallerAudioPassesThroughDuringAnOpenReply() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{ReplyGap: 5 * time.Second})
	defer func() {
		_ = conn.Close()
		_ = provider.Close()
	}()

	var transcript string
	nextAudio := func() sts.AudioChunk {
		timer := time.NewTimer(5 * time.Second)
		defer timer.Stop()
		for {
			select {
			case event, open := <-provider.Events():
				if !open {
					s.FailNow("provider closed its event stream")
				}
				switch typed := event.(type) {
				case sts.AudioChunk:
					return typed
				case sts.InputTranscript:
					transcript += typed.Text
				case sts.Error:
					s.FailNowf("provider rejected response audio", "context=%q", typed.Context)
				case sts.ResponseComplete:
					s.FailNow("response completed before the next output chunk")
				}
			case <-timer.C:
				s.FailNow("timed out waiting for provider audio")
			}
		}
	}

	firstAudio := sts.PcmData{Samples: []int16{400, -401}, SampleRate: SampleRate, Channels: 1}
	s.Require().NoError(conn.WriteJSON(serverEvent{Type: eventAudioDelta, Delta: base64.StdEncoding.EncodeToString(firstAudio.Bytes())}))
	first := nextAudio()
	s.Equal(firstAudio.Samples, first.Audio.Samples)

	caller := sts.PcmData{Samples: make([]int16, 1600), SampleRate: sts.InputSampleRate, Channels: 1}
	for i := range caller.Samples {
		caller.Samples[i] = int16(700 - i%1300)
	}
	processDone := make(chan error, 1)
	go func() { processDone <- provider.ProcessAudio(caller, sts.Participant{ID: "caller"}) }()
	input := s.nextFrame(conn)
	select {
	case err := <-processDone:
		s.Require().NoError(err)
	case <-time.After(5 * time.Second):
		s.FailNow("caller audio processing did not finish after forwarding the chunk")
	}
	s.Require().Equal(eventAudioAppend, input["type"])
	decoded, err := base64.StdEncoding.DecodeString(input["audio"].(string))
	s.Require().NoError(err)
	s.Equal(audio.Resample(caller, SampleRate, 1).Bytes(), decoded, "the complete caller chunk must reach the provider while its reply is open")

	const callerTranscript = "I have a new question for you."
	s.Require().NoError(conn.WriteJSON(serverEvent{Type: eventInputTranscript, Delta: callerTranscript}))
	secondAudio := sts.PcmData{Samples: []int16{-900, 901}, SampleRate: SampleRate, Channels: 1}
	s.Require().NoError(conn.WriteJSON(serverEvent{Type: eventAudioDelta, Delta: base64.StdEncoding.EncodeToString(secondAudio.Bytes())}))
	second := nextAudio()

	s.Equal(callerTranscript, transcript)
	s.Equal(first.ResponseID, second.ResponseID, "caller audio must not roll the live response over")
	s.Equal(first.Generation, second.Generation, "caller audio must not start a new generation")
	s.Equal(first.Index+1, second.Index, "the next output chunk keeps its place in the original response")
	s.Equal(secondAudio.Samples, second.Audio.Samples, "the distinctive second chunk must arrive intact")
}

func (s *OpenAILiveSocketSuite) TestAToolAnswerContinuesTheBackend() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer s.hangUp(provider, conn)

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
	defer s.hangUp(provider, conn)

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
	defer s.hangUp(provider, conn)

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

	// The silence the model streams between replies is not it talking.
	var hush serverEvent
	s.Require().NoError(json.Unmarshal([]byte(quiet(240)), &hush))
	provider.handleMessage(hush)
	provider.mu.Lock()
	carrying := provider.carrying
	provider.mu.Unlock()
	s.True(carrying, "the filler goes on while the model is silent")

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
	defer s.hangUp(provider, conn)

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

// TestCloseWaitsForSessionClosed: session.closed carries the session's final billed seconds
// and comes after session.close, so Close reads it before dropping the socket.
func (s *OpenAILiveSocketSuite) TestCloseWaitsForSessionClosed() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	closed := make(chan error, 1)
	go func() { closed <- provider.Close() }()
	s.Equal(map[string]any{"type": "session.close"}, s.nextFrame(conn))
	time.Sleep(500 * time.Millisecond)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage,
		[]byte(`{"type":"session.closed","reason":"close_requested","usage":{"seconds":29}}`)))
	select {
	case err := <-closed:
		s.Require().NoError(err)
	case <-time.After(closeGrace):
		s.FailNow("Close kept waiting after session.closed arrived")
	}

	var disconnected []sts.Disconnected
	for event := range provider.Events() {
		if typed, ok := event.(sts.Disconnected); ok {
			disconnected = append(disconnected, typed)
		}
	}
	s.Require().Len(disconnected, 1, "session.closed is read before the socket is dropped")
	s.Equal("close_requested", disconnected[0].Reason)
	s.True(disconnected[0].Clean)
}

// TestTheSessionsSecondsAndItsBackendAreReported uses what a recorded 30-second session on
// a gpt-5.6-luna backend was sent: a backend response of 890 tokens in and 31 out, 10 of them
// thinking, the session's seconds at 13 and 28, and 29 as it closed. The seconds are sent as
// numbers with a fraction, as the Live API sent "seconds":2.0 when probed on 2026-10-23.
func (s *OpenAILiveSocketSuite) TestTheSessionsSecondsAndItsBackendAreReported() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{Backend: "gpt-5.6-luna"})

	for _, frame := range []string{
		`{"type":"session.usage.updated","usage":{"seconds":13.0},"context_window":{"usage_ratio":0.0096}}`,
		`{"type":"response.event","event":{"type":"response.completed","response":{"id":"resp_1",` +
			`"model":"gpt-5.6-luna-2026-09-30","status":"completed","service_tier":"default","usage":{"input_tokens":890,` +
			`"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":31,` +
			`"output_tokens_details":{"reasoning_tokens":10},"total_tokens":921}}}}`,
		`{"type":"session.usage.updated","usage":{"seconds":27.6}}`,
	} {
		s.Require().NoError(conn.WriteMessage(websocket.TextMessage, []byte(frame)))
	}
	closed := make(chan error, 1)
	go func() { closed <- provider.Close() }()
	s.Equal(map[string]any{"type": "session.close"}, s.nextFrame(conn))
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage,
		[]byte(`{"type":"session.closed","reason":"close_requested","usage":{"seconds":29.0}}`)))
	s.Require().NoError(<-closed)

	var seconds []int64
	var backend []sts.BackendUsage
	var disconnected int
	for event := range provider.Events() {
		switch typed := event.(type) {
		case sts.SessionUsage:
			seconds = append(seconds, typed.Seconds)
		case sts.BackendUsage:
			backend = append(backend, typed)
		case sts.Disconnected:
			disconnected++
		}
	}
	s.Equal([]int64{13, 28, 29}, seconds, "each running total as the vendor counts it, rounded, the last as it closed")
	s.Equal(1, disconnected, "session.closed is read, seconds and all")
	s.Equal([]sts.BackendUsage{{
		ResponseID:  "resp_1",
		Model:       "gpt-5.6-luna",
		ServiceTier: "default",
		Usage: llm.Usage{
			InputTokens:         890,
			OutputTokens:        31,
			OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: 10},
			TotalTokens:         921,
		},
	}}, backend, "reported as the model the session delegated to")
	s.True(CapabilitiesFor(DefaultModel).SessionBilled)
}

// TestCloseGivesUpAfterTwoSeconds: a server that never answers session.close holds Close
// for closeGrace and no longer.
func (s *OpenAILiveSocketSuite) TestCloseGivesUpAfterTwoSeconds() {
	fake := newFakeLive("")
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	began := time.Now()
	s.Require().NoError(provider.Close())
	waited := time.Since(began)

	s.Equal(map[string]any{"type": "session.close"}, s.nextFrame(conn))
	s.GreaterOrEqual(waited, closeGrace)
	s.Less(waited, closeGrace+time.Second)
}

// hangUp drops the server's side before closing the provider. A test that is not about
// closing has no server answering session.close, and would otherwise wait out closeGrace.
func (s *OpenAILiveSocketSuite) hangUp(provider *STS, conn *websocket.Conn) {
	_ = conn.Close()
	_ = provider.Close()
}

func keys(object map[string]any) []string {
	var found []string
	for key := range object {
		found = append(found, key)
	}
	return found
}
