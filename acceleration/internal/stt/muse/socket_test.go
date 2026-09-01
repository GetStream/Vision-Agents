package muse

import (
	"context"
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

// MuseSocketSuite exercises the provider against a server that is answering, which is
// where the session identifier, the setup frame, the audio encoding and the wait for the
// tail of a call live.
type MuseSocketSuite struct {
	suite.Suite
}

func TestMuseSocketSuite(t *testing.T) {
	suite.Run(t, new(MuseSocketSuite))
}

// fakeSTT is a Muse socket that accepts one session. It says nothing on its own, as the
// real one does: the setup frame is not acknowledged.
type fakeSTT struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	// requests carries what the provider asked for, which for this API is only the query
	// string: the credentials travel in the setup frame.
	requests chan *http.Request
	done     chan struct{}
}

func newFakeSTT() *fakeSTT {
	fake := &fakeSTT{
		conns:    make(chan *websocket.Conn, 1),
		requests: make(chan *http.Request, 1),
		done:     make(chan struct{}),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.requests <- r
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakeSTT) endpoint() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeSTT) close() {
	close(f.done)
	f.server.Close()
}

// connect returns a started provider and the server side of its connection.
func (s *MuseSocketSuite) connect(fake *fakeSTT, options Options) (*STT, *websocket.Conn) {
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
func (s *MuseSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream.
func (s *MuseSocketSuite) nextFrame(conn *websocket.Conn) (int, []byte) {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	messageType, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	return messageType, raw
}

// setup reads the opening frame and decodes it.
func (s *MuseSocketSuite) setup(conn *websocket.Conn) setupFrame {
	messageType, raw := s.nextFrame(conn)
	s.Require().Equal(websocket.TextMessage, messageType)

	var frame setupFrame
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

func (s *MuseSocketSuite) TestTheSessionIsConfiguredByTheOpeningFrame() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{
		Mode:          ModeDiarization,
		Keyterms:      []string{"Acme Mobile", "eSIM"},
		LanguageHints: []string{"English", "French"},
	})
	defer func() { _ = provider.Close() }()

	frame := s.setup(conn)
	s.Equal(DefaultModel, frame.Model)
	s.Equal(encoding16k, frame.AudioEncoding, "the router decodes to 16kHz")
	s.Equal(ModeDiarization, frame.Mode)
	s.Equal(partialModeCumulative, frame.PartialMode,
		"deltas would leave the consumer to stitch the turn back together")
	s.False(frame.EmitAudioProgress)
	s.Equal([]string{"Acme Mobile", "eSIM"}, frame.Keywords)
	s.Equal([]string{"English", "French"}, frame.LanguageBias)
}

func (s *MuseSocketSuite) TestTheKeyTravelsInTheOpeningFrameRatherThanAHeader() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	s.Equal("Bearer secret", s.setup(conn).Authorization.AccessToken)
	s.Empty((<-fake.requests).Header.Get("Authorization"))
}

func (s *MuseSocketSuite) TestTheSessionIsIdentifiedOnTheQueryString() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.NotEmpty((<-fake.requests).URL.Query().Get("sessionId"))
}

func (s *MuseSocketSuite) TestAudioArrivesAsTheRawPcmItWasGiven() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()
	s.setup(conn)

	samples := []int16{1, -2, 3, -4}
	s.speak(provider, samples)

	messageType, raw := s.nextFrame(conn)
	s.Equal(websocket.BinaryMessage, messageType, "this API takes bytes, not base64")
	want := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Equal(want.Bytes(), raw)
}

func (s *MuseSocketSuite) TestClosingWaitsForTheWordsTheServerWasStillHolding() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})
	s.setup(conn)

	// The server holds the tail until it is told the audio has stopped, which is the case
	// Close exists for: a caller cut off mid-sentence has no silence to end their turn.
	served := make(chan struct{})
	go func() {
		defer close(served)
		for {
			if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
				return
			}
			messageType, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if messageType != websocket.TextMessage {
				continue
			}
			var control struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(raw, &control); err != nil {
				return
			}
			if control.Type == controlTypeEndStream {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(
					`{"type":"speechComplete","turnId":1,"transcript":"In a quiet village.","audioProcessedMs":4000}`))
				_ = conn.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
		}
	}()

	s.speak(provider, []int16{1, 2, 3})
	s.Require().NoError(provider.Close())
	<-served

	s.Equal("In a quiet village.", s.finalText(provider),
		"closing should not cut off the words that had not been transcribed yet")
}

func (s *MuseSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	s.speak(provider, []int16{1, 2, 3})

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

func (s *MuseSocketSuite) TestAServerThatHangsUpMidCallIsReported() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()
	s.setup(conn)

	s.Require().NoError(conn.WriteMessage(websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "")))

	deadline := time.After(5 * time.Second)
	for {
		select {
		case event := <-provider.Events():
			if disconnected, ok := event.(stt.Disconnected); ok {
				s.True(disconnected.Clean)
				return
			}
		case <-deadline:
			s.FailNow("the provider never noticed the server hang up")
			return
		}
	}
}

// finalText is the text of the settled turn among everything the session emitted. Close
// has already run, so the channel is closed and reading it to the end terminates.
func (s *MuseSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
