package grok

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

// GrokSocketSuite exercises the provider against a server that is answering, which is
// where the query string, the credentials, the audio encoding and the wait for the tail of
// a call live.
type GrokSocketSuite struct {
	suite.Suite
}

func TestGrokSocketSuite(t *testing.T) {
	suite.Run(t, new(GrokSocketSuite))
}

// fakeSTT is an xAI socket that accepts one session. It reports itself ready as the real
// one does, so a test starts from a connection that is ready for audio.
type fakeSTT struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	// requests carries what the provider asked for: the query string and the headers.
	requests chan *http.Request
	// ready is the frame the server opens with. Empty means transcript.created.
	ready string
	done  chan struct{}
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

		opening := fake.ready
		if opening == "" {
			opening = `{"type":"transcript.created"}`
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(opening)); err != nil {
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
func (s *GrokSocketSuite) connect(fake *fakeSTT, options Options) (*STT, *websocket.Conn) {
	provider := s.build(fake, options)
	s.Require().NoError(provider.Start(s.ctx()))

	select {
	case conn := <-fake.conns:
		return provider, conn
	case <-time.After(5 * time.Second):
		s.FailNow("the provider never connected")
		return nil, nil
	}
}

// build returns an unstarted provider pointed at the fake.
func (s *GrokSocketSuite) build(fake *fakeSTT, options Options) *STT {
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
	return provider
}

func (s *GrokSocketSuite) ctx() context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	return ctx
}

// speak sends one chunk of audio.
func (s *GrokSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream.
func (s *GrokSocketSuite) nextFrame(conn *websocket.Conn) (int, []byte) {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	messageType, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	return messageType, raw
}

func (s *GrokSocketSuite) TestTheSessionIsConfiguredOnTheQueryString() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{
		Language: "de",
		Keyterms: []string{"Vision Agents"},
	})
	defer func() { _ = provider.Close() }()

	opened := (<-fake.requests).URL.Query()
	s.Equal("16000", opened.Get("sample_rate"))
	s.Equal("pcm", opened.Get("encoding"))
	s.Equal("true", opened.Get("interim_results"),
		"without this nothing is written down until the caller stops")
	s.Equal("de", opened.Get("language"))
	s.Equal([]string{"Vision Agents"}, opened["keyterm"])
}

func (s *GrokSocketSuite) TestTheKeyGoesInTheAuthorizationHeader() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	s.Equal("Bearer secret", (<-fake.requests).Header.Get("Authorization"))
}

func (s *GrokSocketSuite) TestAudioArrivesAsTheRawPcmItWasGiven() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	samples := []int16{1, -2, 3, -4}
	s.speak(provider, samples)

	messageType, raw := s.nextFrame(conn)
	s.Equal(websocket.BinaryMessage, messageType, "this API takes bytes, not base64")
	want := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Equal(want.Bytes(), raw)
}

func (s *GrokSocketSuite) TestStartWaitsForTheServerToBeReady() {
	// Audio sent before the recogniser is up is audio nobody is listening to.
	fake := newFakeSTT()
	fake.ready = `{"type":"transcript.partial","text":"too soon"}`
	defer fake.close()

	err := s.build(fake, Options{}).Start(s.ctx())

	s.ErrorContains(err, "expected \"transcript.created\"")
}

func (s *GrokSocketSuite) TestStartReportsAHandshakeTheServerRejected() {
	fake := newFakeSTT()
	fake.ready = `{"type":"error","message":"invalid api key"}`
	defer fake.close()

	err := s.build(fake, Options{}).Start(s.ctx())

	s.ErrorContains(err, "invalid api key")
}

func (s *GrokSocketSuite) TestClosingWaitsForTheWordsTheServerWasStillHolding() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})

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
			if control.Type == controlTypeAudioDone {
				_ = conn.WriteMessage(websocket.TextMessage,
					[]byte(`{"type":"transcript.done","text":"In a quiet village.","duration":4}`))
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

func (s *GrokSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	s.speak(provider, []int16{1, 2, 3})

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

// finalText is the text of the settled turn among everything the session emitted. Close
// has already run, so the channel is closed and reading it to the end terminates.
func (s *GrokSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
