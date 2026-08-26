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

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// GeminiSocketSuite exercises the provider against a Live API that is answering, which is
// where the handshake, the audio encoding and the wait for the tail of a call live.
type GeminiSocketSuite struct {
	suite.Suite
}

func TestGeminiSocketSuite(t *testing.T) {
	suite.Run(t, new(GeminiSocketSuite))
}

// fakeLive is a Live API that accepts one session. It completes the setup exchange itself,
// so a test starts from a connection that is ready for audio.
type fakeLive struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	setups chan setup
	url    chan string
	done   chan struct{}
}

func newFakeLive() *fakeLive {
	fake := &fakeLive{
		conns:  make(chan *websocket.Conn, 1),
		setups: make(chan setup, 1),
		url:    make(chan string, 1),
		done:   make(chan struct{}),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.url <- r.URL.String()
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
func (s *GeminiSocketSuite) connect(fake *fakeLive, options Options) (*STT, *websocket.Conn) {
	options.URL = fake.endpoint()
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	// The fake answers at once when it answers at all, so no test should pay the wait a
	// real call allows for the tail.
	if options.FlushTimeout == 0 {
		options.FlushTimeout = 500 * time.Millisecond
	}
	provider, err := New(options)
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

// speak sends one chunk of audio.
func (s *GeminiSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
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

// says is a frame carrying more of what the caller is saying, with an optional boundary.
func says(text string, over bool) string {
	frame, _ := json.Marshal(serverMessage{ServerContent: &serverContent{
		InputTranscription: &transcription{Text: text},
		TurnComplete:       over,
	}})
	return string(frame)
}

func (s *GeminiSocketSuite) TestTheSessionAsksForTranscriptionAndNoSpokenReply() {
	fake := newFakeLive()
	defer fake.close()
	s.connect(fake, Options{})

	opened := <-fake.setups
	s.Equal("models/"+DefaultModel, opened.Model)
	s.Equal([]string{"TEXT"}, opened.GenerationConfig.ResponseModalities,
		"a spoken reply is neither wanted nor free")
	s.NotNil(opened.InputAudioTranscription, "without this the session hears nothing back")
	s.Nil(opened.SystemInstruction, "nothing was asked for, so nothing should be said")
}

func (s *GeminiSocketSuite) TestTheKeyIsSentBecauseTheresNoHeaderForIt() {
	fake := newFakeLive()
	defer fake.close()
	s.connect(fake, Options{APIKey: "sec ret"})

	s.Contains(<-fake.url, "key=sec+ret")
}

func (s *GeminiSocketSuite) TestKeytermsReachTheModelAsAnInstruction() {
	fake := newFakeLive()
	defer fake.close()
	s.connect(fake, Options{Keyterms: []string{"Vision Agents"}, LanguageHints: []string{"es"}})

	opened := <-fake.setups
	s.Require().NotNil(opened.SystemInstruction)
	said := opened.SystemInstruction.Parts[0].Text
	s.Contains(said, "Vision Agents")
	s.Contains(said, "es")
}

func (s *GeminiSocketSuite) TestAudioArrivesAsThePcmItWasGiven() {
	fake := newFakeLive()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	samples := []int16{1, -2, 3, -4}
	s.speak(provider, samples)

	frame := s.nextFrame(conn)
	s.Require().NotNil(frame.RealtimeInput)
	s.Require().NotNil(frame.RealtimeInput.Audio)
	s.Equal("audio/pcm;rate=16000", frame.RealtimeInput.Audio.MimeType)

	decoded, err := base64.StdEncoding.DecodeString(frame.RealtimeInput.Audio.Data)
	s.Require().NoError(err)
	want := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Equal(want.Bytes(), decoded)
}

func (s *GeminiSocketSuite) TestClosingWaitsForTheTailOfTheLastTurnEvenAfterAnEarlierOne() {
	// An earlier turn is not the last one. Treating it as the tail would have Close
	// return before the words somebody just said were transcribed.
	fake := newFakeLive()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	// The server settles one turn during the call, then holds the next until it is told
	// the audio has stopped.
	served := make(chan struct{})
	go func() {
		defer close(served)
		if err := conn.WriteMessage(websocket.TextMessage, []byte(says("hello", true))); err != nil {
			return
		}
		for {
			if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
				return
			}
			_, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var frame clientMessage
			if err := json.Unmarshal(raw, &frame); err != nil {
				return
			}
			if frame.RealtimeInput != nil && frame.RealtimeInput.AudioStreamEnd {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(says("goodbye", true)))
				return
			}
		}
	}()

	s.speak(provider, []int16{1, 2, 3})
	// Let the earlier turn finish before closing, so the wait below is unambiguously for
	// the tail rather than for a boundary that was already on its way.
	s.Equal("hello", s.nextFinal(provider))

	s.Require().NoError(provider.Close())
	<-served

	s.Equal("goodbye", s.nextFinal(provider),
		"closing should wait for the words that had not been transcribed yet")
}

// nextFinal returns the text of the next settled turn.
func (s *GeminiSocketSuite) nextFinal(provider *STT) string {
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				s.FailNow("the session ended before the turn settled")
				return ""
			}
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				return transcript.Text
			}
		case <-deadline:
			s.FailNow("timed out waiting for a settled turn")
			return ""
		}
	}
}
