package togetherparakeet

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

// TogetherParakeetSocketSuite exercises the provider against a server that is answering,
// which is where the handshake, the credentials, the audio encoding and the wait for the
// tail of a call live.
type TogetherParakeetSocketSuite struct {
	suite.Suite
}

func TestTogetherParakeetSocketSuite(t *testing.T) {
	suite.Run(t, new(TogetherParakeetSocketSuite))
}

// fakeRealtime is a Together realtime socket that accepts one session. It opens the session
// as the real one does, so a test starts from a connection that is ready for audio.
type fakeRealtime struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	// requests carries what the provider asked for: the query string and the headers.
	requests chan *http.Request
	// opening is the frame the server starts with. Empty means session.created.
	opening string
	done    chan struct{}
}

func newFakeRealtime() *fakeRealtime {
	fake := &fakeRealtime{
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

		opening := fake.opening
		if opening == "" {
			opening = `{"type":"session.created"}`
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(opening)); err != nil {
			return
		}
		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakeRealtime) endpoint() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeRealtime) close() {
	close(f.done)
	f.server.Close()
}

// connect returns a started provider and the server side of its connection.
func (s *TogetherParakeetSocketSuite) connect(fake *fakeRealtime, options Options) (*STT, *websocket.Conn) {
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
func (s *TogetherParakeetSocketSuite) build(fake *fakeRealtime, options Options) *STT {
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

func (s *TogetherParakeetSocketSuite) ctx() context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	return ctx
}

// speak sends one chunk of audio.
func (s *TogetherParakeetSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream.
func (s *TogetherParakeetSocketSuite) nextFrame(conn *websocket.Conn) clientMessage {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)

	var frame clientMessage
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

func (s *TogetherParakeetSocketSuite) TestTheSessionAsksToTranscribeWithTheStreamingModel() {
	fake := newFakeRealtime()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	opened := (<-fake.requests).URL.Query()
	s.Equal("transcription", opened.Get("intent"),
		"this endpoint also serves conversations, and one would try to answer back")
	s.Equal(DefaultModel, opened.Get("model"))
	s.Equal(audioFormat, opened.Get("input_audio_format"))
}

func (s *TogetherParakeetSocketSuite) TestTheKeyAndTheProtocolVersionGoInTheHeaders() {
	fake := newFakeRealtime()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	header := (<-fake.requests).Header
	s.Equal("Bearer secret", header.Get("Authorization"))
	s.Equal("realtime=v1", header.Get("OpenAI-Beta"))
}

func (s *TogetherParakeetSocketSuite) TestAudioArrivesBase64EncodedInsideAFrame() {
	fake := newFakeRealtime()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	samples := []int16{1, -2, 3, -4}
	s.speak(provider, samples)

	frame := s.nextFrame(conn)
	s.Equal(clientTypeAppend, frame.Type)

	decoded, err := base64.StdEncoding.DecodeString(frame.Audio)
	s.Require().NoError(err)
	want := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Equal(want.Bytes(), decoded)
}

func (s *TogetherParakeetSocketSuite) TestStartWaitsForTheSessionToOpen() {
	fake := newFakeRealtime()
	fake.opening = `{"type":"conversation.item.input_audio_transcription.delta","delta":"too soon"}`
	defer fake.close()

	err := s.build(fake, Options{}).Start(s.ctx())

	s.ErrorContains(err, "expected \"session.created\"")
}

func (s *TogetherParakeetSocketSuite) TestStartReportsAHandshakeTheServerRejected() {
	fake := newFakeRealtime()
	fake.opening = `{"type":"error","error":{"message":"model not available"}}`
	defer fake.close()

	err := s.build(fake, Options{}).Start(s.ctx())

	s.ErrorContains(err, "model not available")
}

func (s *TogetherParakeetSocketSuite) TestClosingCommitsTheAudioTheServerHadNotTranscribed() {
	fake := newFakeRealtime()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})

	// The server holds the tail until it is told to transcribe what it has, which is the
	// case Close exists for: a caller cut off mid-sentence never paused.
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
			var frame clientMessage
			if err := json.Unmarshal(raw, &frame); err != nil {
				return
			}
			if frame.Type == clientTypeCommit {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(
					`{"type":"conversation.item.input_audio_transcription.completed",`+
						`"transcript":"In a quiet village."}`))
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

func (s *TogetherParakeetSocketSuite) TestClosingASettledCallDoesNotWaitOutTheTimeout() {
	// Nothing is outstanding once an utterance has settled, and no further one is coming.
	// Waiting the full timeout here is that long spent on every hangup.
	fake := newFakeRealtime()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: time.Minute})

	go func() {
		_ = conn.WriteMessage(websocket.TextMessage, []byte(
			`{"type":"conversation.item.input_audio_transcription.completed",`+
				`"transcript":"In a quiet village."}`))
	}()

	s.speak(provider, []int16{1, 2, 3})
	s.Require().Equal("In a quiet village.", s.nextFinal(provider))

	closing := time.Now()
	s.Require().NoError(provider.Close())
	s.Less(time.Since(closing), 5*time.Second,
		"a call whose last utterance already settled should not wait for another")
}

func (s *TogetherParakeetSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeRealtime()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	// A delta and no completed is the server still working, which is what earns the full
	// timeout rather than the grace a settled call gets.
	provider.handleMessage(delta("in a quiet"))
	s.speak(provider, []int16{1, 2, 3})

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

// nextFinal returns the text of the next settled utterance.
func (s *TogetherParakeetSocketSuite) nextFinal(provider *STT) string {
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

// finalText is the text of the settled turn among everything the session emitted. Close
// has already run, so the channel is closed and reading it to the end terminates.
func (s *TogetherParakeetSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
