package openaicompat

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type socketSession struct {
	header http.Header
	frames chan map[string]any
}

type BasetenSocketSuite struct {
	suite.Suite
	server   *httptest.Server
	sessions chan socketSession
}

func TestBasetenSocketSuite(t *testing.T) { suite.Run(t, new(BasetenSocketSuite)) }

func (s *BasetenSocketSuite) SetupSuite() {
	s.sessions = make(chan socketSession, 8)
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		frames := make(chan map[string]any, 8)
		s.sessions <- socketSession{header: r.Header, frames: frames}
		for {
			var message map[string]any
			if conn.ReadJSON(&message) != nil {
				return
			}
			frames <- message
			if message["type"] == "input_audio_buffer.append" {
				raw, _ := fixtures.ReadFile("testdata/baseten_partial.json")
				if conn.WriteMessage(websocket.TextMessage, raw) != nil {
					return
				}
			}
			if message["type"] == "input_audio_buffer.commit" {
				raw, _ := fixtures.ReadFile("testdata/baseten_flush.json")
				_ = conn.WriteMessage(websocket.TextMessage, raw)
			}
		}
	}))
}

func (s *BasetenSocketSuite) connect() (*basetenRealtime, socketSession) {
	p, err := newBasetenRealtime(Options{BaseURL: s.server.URL, APIKey: "secret"})
	s.Require().NoError(err)
	s.Require().NoError(p.Start(context.Background()))
	session := <-s.sessions
	s.IsType(stt.Connected{}, <-p.Events())
	return p, session
}

func (s *BasetenSocketSuite) TestSetupAndAudioUseTheBasetenProtocol() {
	p, session := s.connect()
	s.Equal("Api-Key secret", session.header.Get("Authorization"))
	setup := <-session.frames
	s.Equal("auto", setup["whisper_params"].(map[string]any)["audio_language"])
	s.Equal(true, setup["streaming_params"].(map[string]any)["enable_partial_transcripts"])
	s.NotContains(setup, "streaming_vad_config", "the endpoint owns its VAD defaults")
	pcm := stt.PcmData{Samples: []int16{1, -2, 3, -4}, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(p.ProcessAudio(pcm, stt.Participant{ID: "alice"}))
	audio := <-session.frames
	s.Equal("input_audio_buffer.append", audio["type"])
	decoded, err := base64.StdEncoding.DecodeString(audio["audio"].(string))
	s.Require().NoError(err)
	s.Equal(pcm.Bytes(), decoded)
	transcript := (<-p.Events()).(stt.Transcript)
	s.Equal("in a quiet village", transcript.Text)
	s.Equal("alice", transcript.Participant.ID)
	s.Require().NoError(p.Close())
}

func (s *BasetenSocketSuite) TestHangingUpRetainsTheFinalTranscript() {
	p, session := s.connect()
	<-session.frames
	s.Require().NoError(p.Close())
	s.Equal("input_audio_buffer.commit", (<-session.frames)["type"])
	var transcripts []stt.Transcript
	for event := range p.Events() {
		if transcript, ok := event.(stt.Transcript); ok {
			transcripts = append(transcripts, transcript)
		}
	}
	s.Require().Len(transcripts, 1)
	s.Equal("the tail of the call", transcripts[0].Text)
	s.True(transcripts[0].Final())
	s.Require().NoError(p.Close())
	s.Error(p.ProcessAudio(stt.PcmData{Samples: []int16{1}, SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{}))
}
