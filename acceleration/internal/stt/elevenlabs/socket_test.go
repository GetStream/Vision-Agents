package elevenlabs

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

// ElevenlabsSocketSuite exercises the provider against a server that is answering, which
// is where the query string, the credentials, the handshake, the audio envelope and the
// wait for the tail of a call live.
type ElevenlabsSocketSuite struct {
	suite.Suite
}

func TestElevenlabsSocketSuite(t *testing.T) {
	suite.Run(t, new(ElevenlabsSocketSuite))
}

// fakeSTT is a Scribe realtime socket that accepts one session and opens it the way the
// real one does, since the provider will not send audio until it has been opened.
type fakeSTT struct {
	server   *httptest.Server
	conns    chan *websocket.Conn
	requests chan *http.Request
	done     chan struct{}
	// greeting is what the server says first. Empty means the session_started frame the
	// provider is waiting for.
	greeting string
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
		greeting := fake.greeting
		if greeting == "" {
			greeting = `{"message_type":"session_started","session_id":"s1","config":{}}`
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(greeting)); err != nil {
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
func (s *ElevenlabsSocketSuite) connect(fake *fakeSTT, options Options) (*STT, *websocket.Conn) {
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
func (s *ElevenlabsSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream.
func (s *ElevenlabsSocketSuite) nextFrame(conn *websocket.Conn) clientMessage {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	messageType, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	s.Require().Equal(websocket.TextMessage, messageType)

	var frame clientMessage
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

func (s *ElevenlabsSocketSuite) TestTheSessionIsConfiguredByTheQueryString() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{
		Keyterms:                []string{"Scribe", "Acme Mobile"},
		LanguageHints:           []string{"en", "fr", "de"},
		VadSilenceThresholdSecs: 0.5,
		VadThreshold:            0.4,
		MinSpeechDurationMs:     120,
		MinSilenceDurationMs:    240,
	})
	defer func() { _ = provider.Close() }()

	query := (<-fake.requests).URL.Query()
	s.Equal(DefaultModel, query.Get("model_id"))
	s.Equal(audioFormat16k, query.Get("audio_format"))
	s.Equal(CommitOnVAD, query.Get("commit_strategy"))
	s.Equal("en", query.Get("language_code"))
	s.Equal([]string{"fr", "de"}, query["secondary_languages"],
		"the language the call is in is a different question from the ones it may switch into")
	s.Equal([]string{"Scribe", "Acme Mobile"}, query["keyterms"])
	s.Equal("0.5", query.Get("vad_silence_threshold_secs"))
	s.Equal("0.4", query.Get("vad_threshold"))
	s.Equal("120", query.Get("min_speech_duration_ms"))
	s.Equal("240", query.Get("min_silence_duration_ms"))
}

func (s *ElevenlabsSocketSuite) TestTurnDetectionIsLeftToTheServerWhenNobodyAsked() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	query := (<-fake.requests).URL.Query()
	s.Empty(query.Get("vad_silence_threshold_secs"))
	s.Empty(query.Get("min_silence_duration_ms"))
}

func (s *ElevenlabsSocketSuite) TestTheKeyTravelsInAHeader() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	request := <-fake.requests
	s.Equal("secret", request.Header.Get(apiKeyHeader))
	s.Empty(request.URL.Query().Get("token"),
		"the single-use token is for a browser, not for the router")
}

func (s *ElevenlabsSocketSuite) TestStartWaitsForTheServerToOpenTheSession() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	// Audio sent before the session opened is audio nobody is listening to, so the first
	// frame upstream should be the first chunk rather than anything sent while waiting.
	s.speak(provider, []int16{1, 2, 3})
	s.Equal(clientTypeAudioChunk, s.nextFrame(conn).MessageType)
}

func (s *ElevenlabsSocketSuite) TestARefusedSessionIsReportedFromStart() {
	fake := newFakeSTT()
	fake.greeting = `{"message_type":"auth_error","error":"invalid api key"}`
	defer fake.close()

	provider, err := New(Options{APIKey: "wrong", URL: fake.endpoint()})
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	s.ErrorContains(provider.Start(ctx), "invalid api key")
}

func (s *ElevenlabsSocketSuite) TestAudioArrivesBase64EncodedWithoutCommittingAnything() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	samples := []int16{1, -2, 3, -4}
	s.speak(provider, samples)

	frame := s.nextFrame(conn)
	want := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Equal(base64.StdEncoding.EncodeToString(want.Bytes()), frame.AudioBase64)
	s.Equal(stt.SampleRate, frame.SampleRate)
	s.False(frame.Commit, "committing every chunk would settle a turn mid-sentence")
}

func (s *ElevenlabsSocketSuite) TestClosingCommitsTheWordsTheServerWasStillHolding() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})

	// A caller cut off mid-sentence leaves no silence for the detector to settle on, so
	// the tail is committed by asking for it. That request is the only kind of frame this
	// protocol has: a chunk of audio carrying the commit flag.
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
			if !frame.Commit {
				continue
			}
			_ = conn.WriteMessage(websocket.TextMessage, []byte(
				`{"message_type":"committed_transcript","text":"In a quiet village."}`))
			return
		}
	}()

	s.speak(provider, []int16{1, 2, 3})
	s.Require().NoError(provider.Close())
	<-served

	s.Equal("In a quiet village.", s.finalText(provider),
		"closing should not cut off the words that had not been committed yet")
}

func (s *ElevenlabsSocketSuite) TestClosingIsNotSatisfiedByASegmentFromEarlierInTheCall() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 300 * time.Millisecond})

	s.speak(provider, []int16{1, 2, 3})
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, []byte(
		`{"message_type":"committed_transcript","text":"Hello."}`)))

	// Give the segment time to land before closing, so the wait below is for one that
	// comes after the audio stopped rather than for that one.
	deadline := time.After(5 * time.Second)
	for {
		heard := false
		select {
		case event := <-provider.Events():
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				heard = true
			}
		case <-deadline:
			s.FailNow("the first segment never arrived")
		}
		if heard {
			break
		}
	}

	closing := time.Now()
	s.Require().NoError(provider.Close())
	s.GreaterOrEqual(time.Since(closing), 300*time.Millisecond,
		"a segment that settled before the hangup says nothing about the tail")
}

func (s *ElevenlabsSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	s.speak(provider, []int16{1, 2, 3})

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

func (s *ElevenlabsSocketSuite) TestAServerThatHangsUpMidCallIsReported() {
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
func (s *ElevenlabsSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
