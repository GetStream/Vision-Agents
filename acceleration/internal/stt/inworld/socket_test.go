package inworld

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

// InworldSocketSuite exercises the provider against a server that is answering, which is
// where the opening frame, the credentials, the audio envelope and the wait for the tail
// of a call live.
type InworldSocketSuite struct {
	suite.Suite
}

func TestInworldSocketSuite(t *testing.T) {
	suite.Run(t, new(InworldSocketSuite))
}

// fakeSTT is an Inworld streaming socket that accepts one session. It says nothing on its
// own, as the real one does: the opening frame is not acknowledged.
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
func (s *InworldSocketSuite) connect(fake *fakeSTT, options Options) (*STT, *websocket.Conn) {
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
func (s *InworldSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream and decodes it as the
// envelope this protocol wraps everything in.
func (s *InworldSocketSuite) nextFrame(conn *websocket.Conn) clientMessage {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	messageType, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	s.Require().Equal(websocket.TextMessage, messageType)

	var frame clientMessage
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

func (s *InworldSocketSuite) TestTheSessionIsConfiguredByTheOpeningFrame() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{
		Keyterms:                     []string{"Acme Mobile", "eSIM"},
		LanguageHints:                []string{"en", "fr"},
		EndOfTurnConfidenceThreshold: 0.7,
		MinEndOfTurnSilenceMs:        480,
		InactivityTimeoutSeconds:     30,
	})
	defer func() { _ = provider.Close() }()

	held := s.nextFrame(conn).TranscribeConfig
	s.Require().NotNil(held, "an unwrapped config closes the socket without an error frame")
	s.Equal(modelVendor+DefaultModel, held.ModelID,
		"the gateway names a model by the vendor serving it, and routing names that separately")
	s.Equal(encodingLinear16, held.AudioEncoding)
	s.Equal(stt.SampleRate, held.SampleRateHertz)
	s.Equal(1, held.NumberOfChannels)
	s.Equal([]string{"Acme Mobile", "eSIM"}, held.Prompts)
	s.Equal("en", held.Language, "the field takes one code, so a list of hints has one answer")
	s.Equal(0.7, held.EndOfTurnConfidenceThreshold)
	s.Equal(30, held.InactivityTimeoutSeconds)
	s.Require().NotNil(held.InworldSttV1Config)
	s.Equal(480, held.InworldSttV1Config.MinEndOfTurnSilenceWhenConfident)
}

func (s *InworldSocketSuite) TestTurnDetectionIsLeftToTheServerWhenNobodyAsked() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	held := s.nextFrame(conn).TranscribeConfig
	s.Require().NotNil(held)
	s.Nil(held.InworldSttV1Config,
		"sending an empty block would state defaults the server already has")
}

func (s *InworldSocketSuite) TestTheKeyTravelsInAHeader() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	s.Equal("Basic secret", (<-fake.requests).Header.Get("Authorization"))
}

func (s *InworldSocketSuite) TestAudioArrivesBase64EncodedInsideAJsonFrame() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()
	s.nextFrame(conn)

	samples := []int16{1, -2, 3, -4}
	s.speak(provider, samples)

	chunk := s.nextFrame(conn).AudioChunk
	s.Require().NotNil(chunk)
	want := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Equal(base64.StdEncoding.EncodeToString(want.Bytes()), chunk.Content)
}

func (s *InworldSocketSuite) TestClosingWaitsForTheWordsTheServerWasStillHolding() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})
	s.nextFrame(conn)

	// A caller cut off mid-sentence gives the server no silence to end their turn with,
	// so the turn is settled by being told the turn is over rather than by hearing it.
	served := make(chan struct{})
	go func() {
		defer close(served)
		for {
			if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
				return
			}
			_, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var command map[string]json.RawMessage
			if err := json.Unmarshal(raw, &command); err != nil {
				return
			}
			if _, ok := command["endTurn"]; !ok {
				continue
			}
			_ = conn.WriteMessage(websocket.TextMessage, []byte(
				`{"result":{"transcription":{"transcript":"In a quiet village.","isFinal":true}}}`))
			_ = conn.WriteMessage(websocket.TextMessage, []byte(
				`{"result":{"usage":{"transcribedAudioMs":4000,"modelId":"inworld/inworld-stt-1"}}}`))
			return
		}
	}()

	s.speak(provider, []int16{1, 2, 3})
	s.Require().NoError(provider.Close())
	<-served

	s.Equal("In a quiet village.", s.finalText(provider),
		"closing should not cut off the words that had not been transcribed yet")
}

func (s *InworldSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	s.speak(provider, []int16{1, 2, 3})

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

func (s *InworldSocketSuite) TestAServerThatHangsUpMidCallIsReported() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()
	s.nextFrame(conn)

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
func (s *InworldSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
