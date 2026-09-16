package cartesia

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// CartesiaSocketSuite exercises the provider against a server that is answering, which is
// where the query string, the credentials, the audio encoding and the wait for the tail of
// a call live. This API is configured entirely by the query string, so what goes on the
// wire at setup is a URL rather than a frame.
type CartesiaSocketSuite struct {
	suite.Suite
}

func TestCartesiaSocketSuite(t *testing.T) {
	suite.Run(t, new(CartesiaSocketSuite))
}

// fakeSTT is a Cartesia turns socket that accepts one session.
type fakeSTT struct {
	server   *httptest.Server
	conns    chan *websocket.Conn
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
func (s *CartesiaSocketSuite) connect(fake *fakeSTT, options Options) (*STT, *websocket.Conn) {
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
func (s *CartesiaSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream.
func (s *CartesiaSocketSuite) nextFrame(conn *websocket.Conn) (int, []byte) {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	messageType, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	return messageType, raw
}

func (s *CartesiaSocketSuite) TestTheSessionIsConfiguredByTheQueryString() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	query := (<-fake.requests).URL.Query()
	s.Equal(DefaultModel, query.Get("model"))
	s.Equal(encodingPCM16, query.Get("encoding"))
	s.Equal(strconv.Itoa(stt.SampleRate), query.Get("sample_rate"))
	s.Equal(APIVersion, query.Get("cartesia_version"),
		"the endpoint refuses a request that does not name the version it was written against")
}

func (s *CartesiaSocketSuite) TestTheKeyTravelsInAHeader() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	request := <-fake.requests
	s.Equal("secret", request.Header.Get(apiKeyHeader))
	s.Empty(request.URL.Query().Get("access_token"),
		"the token in the query string is for a browser, not for the router")
}

func (s *CartesiaSocketSuite) TestEachKeytermIsItsOwnParameter() {
	// The parameter is singular and one term per copy is how this endpoint reads a list.
	// Joining them would boost one phrase nobody said.
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{Keyterms: []string{"Ink 2", "Cartesia"}})
	defer func() { _ = provider.Close() }()

	s.Equal([]string{"Ink 2", "Cartesia"}, (<-fake.requests).URL.Query()["keyterm"])
}

func (s *CartesiaSocketSuite) TestTurnDetectionSettingsAreSentWhenAskedFor() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{
		TurnEndTimeoutMs:      800,
		TurnStartThreshold:    0.7,
		TurnEagerEndThreshold: 0.45,
		TurnEndThreshold:      0.25,
	})
	defer func() { _ = provider.Close() }()

	query := (<-fake.requests).URL.Query()
	s.Equal("800", query.Get("turn_end_timeout_ms"))
	s.Equal("0.7", query.Get("turn_start_threshold"))
	s.Equal("0.45", query.Get("turn_eager_end_threshold"))
	s.Equal("0.25", query.Get("turn_end_threshold"))
}

func (s *CartesiaSocketSuite) TestTurnDetectionIsLeftToTheServerWhenNobodyAsked() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	query := (<-fake.requests).URL.Query()
	s.Empty(query.Get("turn_end_timeout_ms"))
	s.Empty(query.Get("turn_start_threshold"))
}

func (s *CartesiaSocketSuite) TestAudioArrivesAsTheRawPcmItWasGiven() {
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

func (s *CartesiaSocketSuite) TestClosingWaitsForTheWordsTheServerWasStillHolding() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})

	// The server buffers audio to transcribe it better, and only processes what it is
	// holding once it is told the session is over. That is the case Close exists for: a
	// caller cut off mid-sentence has no silence to end their turn with.
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
			var command struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(raw, &command); err != nil {
				return
			}
			if command.Type == controlTypeClose {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(
					`{"type":"turn.end","transcript":"In a quiet village.","request_id":"r1"}`))
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

func (s *CartesiaSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	s.speak(provider, []int16{1, 2, 3})

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

func (s *CartesiaSocketSuite) TestAServerThatHangsUpMidCallIsReported() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

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
func (s *CartesiaSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
