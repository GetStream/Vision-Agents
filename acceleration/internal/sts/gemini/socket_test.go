package gemini

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
)

// GeminiSocketSuite exercises the provider against a Live API that is answering, which is
// where the setup frame, the audio encoding and the resume live.
type GeminiSocketSuite struct {
	suite.Suite
}

func TestGeminiSocketSuite(t *testing.T) {
	suite.Run(t, new(GeminiSocketSuite))
}

// fakeLive is a Live API that accepts sessions one after another. It completes the setup
// exchange itself, so a test starts from a connection that is ready for audio, and it keeps
// every setup it was sent so a resumed session can be checked for its handle.
type fakeLive struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	setups chan setup
	done   chan struct{}
}

func newFakeLive() *fakeLive {
	fake := &fakeLive{
		conns:  make(chan *websocket.Conn, 4),
		setups: make(chan setup, 4),
		done:   make(chan struct{}),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}

		_, raw, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var frame clientMessage
		if err := json.Unmarshal(raw, &frame); err != nil || frame.Setup == nil {
			return
		}
		fake.setups <- *frame.Setup

		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)); err != nil {
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
func (s *GeminiSocketSuite) connect(fake *fakeLive, settings Options) (*STS, *websocket.Conn) {
	settings.URL = fake.endpoint()
	if settings.APIKey == "" {
		settings.APIKey = "test-key"
	}
	if settings.HandshakeTimeout == 0 {
		settings.HandshakeTimeout = 5 * time.Second
	}
	provider, err := New(settings)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	return provider, s.nextConn(fake)
}

func (s *GeminiSocketSuite) nextConn(fake *fakeLive) *websocket.Conn {
	select {
	case conn := <-fake.conns:
		return conn
	case <-time.After(5 * time.Second):
		s.FailNow("the provider never connected")
		return nil
	}
}

// nextFrame reads the next message the provider sent upstream.
func (s *GeminiSocketSuite) nextFrame(conn *websocket.Conn) clientMessage {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)

	var frame clientMessage
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

func (s *GeminiSocketSuite) serve(conn *websocket.Conn, message serverMessage) {
	raw, err := json.Marshal(message)
	s.Require().NoError(err)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, raw))
}

// nextEvent waits for the next event of the wanted kind.
func (s *GeminiSocketSuite) nextEvent(provider *STS, wanted func(sts.Event) bool) sts.Event {
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				s.FailNow("the session ended first")
				return nil
			}
			if wanted(event) {
				return event
			}
		case <-deadline:
			s.FailNow("timed out waiting for an event")
			return nil
		}
	}
}

func (s *GeminiSocketSuite) TestTheSessionAsksForAudioTranscriptsToolsAndAVoice() {
	fake := newFakeLive()
	defer fake.close()
	on := true
	provider, _ := s.connect(fake, Options{
		Voice:            "Kore",
		Instructions:     "Be brief.",
		Tools:            []llm.Tool{{Name: "get_weather", Description: "Weather.", Parameters: map[string]any{"type": "object"}}},
		InputTranscript:  true,
		OutputTranscript: true,
		ThinkingLevel:    "low",
		ProactiveAudio:   &on,
	})
	defer func() { _ = provider.Close() }()

	opened := <-fake.setups
	s.Equal("models/"+DefaultModel, opened.Model)
	s.Equal([]string{"AUDIO"}, opened.GenerationConfig.ResponseModalities)
	s.Equal("Kore", opened.GenerationConfig.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig.VoiceName)
	s.Equal("low", opened.GenerationConfig.ThinkingConfig.ThinkingLevel)
	s.Equal("Be brief.", opened.SystemInstruction.Parts[0].Text)
	s.Require().Len(opened.Tools, 1)
	s.Equal("get_weather", opened.Tools[0].FunctionDeclarations[0].Name)
	s.NotNil(opened.InputAudioTranscription)
	s.NotNil(opened.OutputAudioTranscription)
	s.Require().NotNil(opened.SessionResumption, "always asked for, or the session could never be resumed")
	s.Empty(opened.SessionResumption.Handle)
	s.True(opened.Proactivity.ProactiveAudio)
}

func (s *GeminiSocketSuite) TestASessionAskingForNothingExtraSendsNothingExtra() {
	fake := newFakeLive()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	opened := <-fake.setups
	s.Nil(opened.GenerationConfig.SpeechConfig, "no voice, so the model's own")
	s.Nil(opened.InputAudioTranscription, "not asked for, so not billed for")
	s.Nil(opened.SystemInstruction)
	s.Nil(opened.RealtimeInputConfig)
}

func (s *GeminiSocketSuite) TestAudioAndFramesArriveAsWhatTheyWere() {
	fake := newFakeLive()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	samples := []int16{1, -2, 3, -4}
	pcm := sts.PcmData{Samples: samples, SampleRate: sts.InputSampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, sts.Participant{ID: "alice"}))

	frame := s.nextFrame(conn)
	s.Require().NotNil(frame.RealtimeInput)
	s.Equal("audio/pcm;rate=16000", frame.RealtimeInput.Audio.MimeType)
	decoded, err := base64.StdEncoding.DecodeString(frame.RealtimeInput.Audio.Data)
	s.Require().NoError(err)
	s.Equal(pcm.Bytes(), decoded)

	s.Require().NoError(provider.SendFrame(llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8}}))
	still := s.nextFrame(conn)
	s.Equal("image/jpeg", still.RealtimeInput.Video.MimeType)
}

func (s *GeminiSocketSuite) TestATypedTurnAndAPromptAreSentComplete() {
	fake := newFakeLive()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.SendText("hello", sts.Participant{ID: "alice"}))
	turn := s.nextFrame(conn)
	s.Require().NotNil(turn.ClientContent)
	s.True(turn.ClientContent.TurnComplete, "an incomplete turn would never be answered")
	s.Equal("user", turn.ClientContent.Turns[0].Role)
	s.Equal("hello", turn.ClientContent.Turns[0].Parts[0].Text)

	s.Require().NoError(provider.Prompt("Greet the caller."))
	prompt := s.nextFrame(conn)
	s.Equal("Greet the caller.", prompt.ClientContent.Turns[0].Parts[0].Text)
}

func (s *GeminiSocketSuite) TestAToolAnswerCarriesTheNameItWasCalledUnder() {
	fake := newFakeLive()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverMessage{ToolCall: &toolCall{FunctionCalls: []functionCall{{ID: "call_1", Name: "get_weather"}}}})
	s.nextEvent(provider, func(event sts.Event) bool { _, ok := event.(sts.ToolCall); return ok })

	s.Require().NoError(provider.Answer("call_1", `{"forecast":"sunny"}`, nil))
	answer := s.nextFrame(conn)
	s.Require().NotNil(answer.ToolResponse)
	s.Equal("call_1", answer.ToolResponse.FunctionResponses[0].ID)
	s.Equal("get_weather", answer.ToolResponse.FunctionResponses[0].Name)
	s.Equal(`{"forecast":"sunny"}`, answer.ToolResponse.FunctionResponses[0].Response["output"])
}

func (s *GeminiSocketSuite) TestTheSessionResumesWithItsHandleWhenTheServerHangsUp() {
	fake := newFakeLive()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()
	<-fake.setups

	s.serve(conn, serverMessage{SessionResumptionUpdate: &resumptionUpdate{NewHandle: "h-1", Resumable: true}})
	s.serve(conn, serverMessage{GoAway: &goAway{TimeLeft: "5s"}})
	s.nextEvent(provider, func(event sts.Event) bool { _, ok := event.(sts.SessionExpiring); return ok })

	// The server cuts the connection the way it does every ten minutes.
	s.Require().NoError(conn.WriteControl(websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseInternalServerErr, "session timeout"), time.Now().Add(time.Second)))
	conn.Close()

	dropped := s.nextEvent(provider, func(event sts.Event) bool { _, ok := event.(sts.Disconnected); return ok })
	s.True(dropped.(sts.Disconnected).Clean, "a connection cut so it can be resumed is not a failure")
	s.Equal("resuming", dropped.(sts.Disconnected).Reason)

	s.nextConn(fake)
	resumed := <-fake.setups
	s.Require().NotNil(resumed.SessionResumption)
	s.Equal("h-1", resumed.SessionResumption.Handle, "the new connection has to pick the conversation up where it was")
	s.nextEvent(provider, func(event sts.Event) bool { _, ok := event.(sts.Connected); return ok })
}
