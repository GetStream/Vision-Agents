package fish

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// FishSocketSuite exercises the provider against a server that is answering, which is
// where the headers, the config, the framing of the audio and the wait for the tail of a
// call live.
type FishSocketSuite struct {
	suite.Suite
}

func TestFishSocketSuite(t *testing.T) {
	suite.Run(t, new(FishSocketSuite))
}

// fakeSTT is a realtime socket that accepts one session, reads its config and opens it
// the way the real one does, since the provider will not send audio until it is open.
type fakeSTT struct {
	server   *httptest.Server
	conns    chan *websocket.Conn
	requests chan *http.Request
	configs  chan map[string]any
	done     chan struct{}
	// greeting is the server's answer to the config. Empty means session_started.
	greeting []byte
}

func newFakeSTT() *fakeSTT {
	fake := &fakeSTT{
		conns:    make(chan *websocket.Conn, 1),
		requests: make(chan *http.Request, 1),
		configs:  make(chan map[string]any, 1),
		done:     make(chan struct{}),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.requests <- r
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		_, raw, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var config map[string]any
		if err := msgpack.Unmarshal(raw, &config); err != nil {
			return
		}
		fake.configs <- config
		greeting := fake.greeting
		if greeting == nil {
			greeting = frame("session_started.json")
		}
		if err := conn.WriteMessage(websocket.BinaryMessage, greeting); err != nil {
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
func (s *FishSocketSuite) connect(fake *fakeSTT, options Options) (*STT, *websocket.Conn) {
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

// speak sends this many milliseconds of audio as one chunk.
func (s *FishSocketSuite) speak(provider *STT, ms int) {
	pcm := stt.PcmData{
		Samples:    make([]int16, stt.SampleRate*ms/1000),
		SampleRate: stt.SampleRate,
		Channels:   1,
	}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextMessage reads the next message the provider sent upstream.
func (s *FishSocketSuite) nextMessage(conn *websocket.Conn) map[string]any {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	messageType, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	s.Require().Equal(websocket.BinaryMessage, messageType, "every message is MessagePack in a binary frame")
	var message map[string]any
	s.Require().NoError(msgpack.Unmarshal(raw, &message))
	return message
}

func (s *FishSocketSuite) TestTheKeyAndTheModelTravelInTheHeaders() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	request := <-fake.requests
	s.Equal("Bearer secret", request.Header.Get("Authorization"))
	s.Equal(DefaultModel, request.Header.Get("model"))
}

func (s *FishSocketSuite) TestACallIsConfiguredForPlainWordsAsTheyAreHeard() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	config := <-fake.configs
	s.Equal("config", config["type"])
	audio := config["audio"].(map[string]any)
	s.Equal("pcm_s16le", audio["format"])
	s.EqualValues(stt.SampleRate, audio["sample_rate"])
	s.EqualValues(1, audio["channels"])
	s.Equal(map[string]any{"drafts": true, "diarize": false}, config["results"],
		"drafts are the interim transcripts, and diarization would write speaker tags into the text")
	s.Equal(map[string]any{"mode": ModeAuto, "eager": false}, config["turn_detection"],
		"a silence nobody asked for is left to the mode")
	s.NotContains(config, "context", "no keyterms were asked for")
}

func (s *FishSocketSuite) TestTheTurnDetectionAndKeytermsAskedForAreConfigured() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{
		Keyterms:         []string{"Mia", "hues of gold"},
		Mode:             ModeVAD,
		SilenceTimeoutMs: 900,
		Eager:            true,
	})
	defer func() { _ = provider.Close() }()

	config := <-fake.configs
	turnDetection := config["turn_detection"].(map[string]any)
	s.Equal(ModeVAD, turnDetection["mode"])
	s.Equal(true, turnDetection["eager"])
	s.EqualValues(900, turnDetection["silence_timeout_ms"])
	s.Equal(map[string]any{"terms": []any{"Mia", "hues of gold"}}, config["context"])
}

func (s *FishSocketSuite) TestARefusedConfigIsReportedFromStart() {
	fake := newFakeSTT()
	fake.greeting = frame("error_invalid_config.json")
	defer fake.close()

	provider, err := New(Options{APIKey: "test-key", URL: fake.endpoint()})
	s.Require().NoError(err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = provider.Start(ctx)
	s.ErrorContains(err, "session refused: invalid_config: turn_detection.silence_timeout_ms must be between 3000 and 60000 in auto mode")
}

func (s *FishSocketSuite) TestAudioGoesUpAsItArrives() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.speak(provider, 20)

	message := s.nextMessage(conn)
	s.Equal("audio", message["type"])
	s.Len(message["data"], 2*stt.SampleRate*20/1000, "a call frame is sent as it is, 16-bit samples")
}

func (s *FishSocketSuite) TestKeytermsCanBeReplacedOnAnOpenSession() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{Keyterms: []string{"Mia"}})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.SetKeyterms([]string{"Fish Audio", " "}))

	s.Equal(map[string]any{"type": "configure", "context": map[string]any{"terms": []any{"Fish Audio"}}},
		s.nextMessage(conn))
}

func (s *FishSocketSuite) TestClearingTheKeytermsSendsAnEmptyList() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{Keyterms: []string{"Mia"}})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.SetKeyterms(nil))

	s.Equal(map[string]any{"terms": []any{}}, s.nextMessage(conn)["context"],
		"the new list replaces the old one, so an empty one is what clears it")
}

func (s *FishSocketSuite) TestClosingAsksForTheTurnToSettleAndWaitsForIt() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})

	// A caller cut off mid-sentence leaves no silence for the turn detector to settle on,
	// so the tail is settled by audio_done, and the server answers with the turn and then
	// with session_done.
	served := make(chan []string, 1)
	go func() {
		var types []string
		defer func() { served <- types }()
		for {
			if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
				return
			}
			_, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var message map[string]any
			if err := msgpack.Unmarshal(raw, &message); err != nil {
				return
			}
			types = append(types, message["type"].(string))
			if message["type"] != clientTypeAudioDone {
				continue
			}
			for _, name := range []string{"user_turn_taken.json", "committed.json", "session_done.json"} {
				_ = conn.WriteMessage(websocket.BinaryMessage, frame(name))
			}
			return
		}
	}()

	s.speak(provider, 100)
	s.Require().NoError(provider.Close())

	s.Equal([]string{"audio", "audio_done"}, <-served)
	s.Equal("In a quiet village where the sky brushes the fields in hues of gold, young Mia "+
		"discovered a map leading to forgotten treasures.", s.finalText(provider),
		"closing should not cut off the words that had not settled yet")
}

func (s *FishSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	s.speak(provider, 100)

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

func (s *FishSocketSuite) TestAServerThatHangsUpMidCallIsReported() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(conn.WriteMessage(websocket.CloseMessage,
		websocket.FormatCloseMessage(4413, "max duration")))

	deadline := time.After(5 * time.Second)
	for {
		select {
		case event := <-provider.Events():
			if failure, ok := event.(stt.Error); ok {
				s.True(failure.Fatal)
				s.Equal("read", failure.Context)
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
func (s *FishSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
