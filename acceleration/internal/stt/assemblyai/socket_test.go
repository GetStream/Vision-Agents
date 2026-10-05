package assemblyai

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

// AssemblyAISocketSuite exercises the provider against a server that is answering, which
// is where the query string, the credentials, the handshake, the framing of the audio and
// the wait for the tail of a call live.
type AssemblyAISocketSuite struct {
	suite.Suite
}

func TestAssemblyAISocketSuite(t *testing.T) {
	suite.Run(t, new(AssemblyAISocketSuite))
}

// fakeSTT is a streaming socket that accepts one session and opens it the way the real one
// does, since the provider will not send audio until it has been opened.
type fakeSTT struct {
	server   *httptest.Server
	conns    chan *websocket.Conn
	requests chan *http.Request
	done     chan struct{}
	// greeting is what the server says first. Empty means the Begin frame of a session
	// opened on the default model and preset.
	greeting []byte
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
		if greeting == nil {
			greeting = frame("begin_max_accuracy.json")
		}
		if err := conn.WriteMessage(websocket.TextMessage, greeting); err != nil {
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
func (s *AssemblyAISocketSuite) connect(fake *fakeSTT, options Options) (*STT, *websocket.Conn) {
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

// startAgainst opens a provider on a server that greets it with this frame, and returns
// what Start said.
func (s *AssemblyAISocketSuite) startAgainst(greeting string, options Options) error {
	fake := newFakeSTT()
	fake.greeting = frame(greeting)
	defer fake.close()

	options.APIKey = "test-key"
	options.URL = fake.endpoint()
	provider, err := New(options)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return provider.Start(ctx)
}

// speak sends this many milliseconds of audio as one chunk.
func (s *AssemblyAISocketSuite) speak(provider *STT, ms int) {
	pcm := stt.PcmData{
		Samples:    make([]int16, stt.SampleRate*ms/1000),
		SampleRate: stt.SampleRate,
		Channels:   1,
	}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream.
func (s *AssemblyAISocketSuite) nextFrame(conn *websocket.Conn) (int, []byte) {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	messageType, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	return messageType, raw
}

// nextAudio reads the next frame and requires it to be audio, returning how long it is.
func (s *AssemblyAISocketSuite) nextAudio(conn *websocket.Conn) int {
	messageType, raw := s.nextFrame(conn)
	s.Require().Equal(websocket.BinaryMessage, messageType, "audio goes up as raw binary, not JSON")
	return len(raw) * 1000 / (2 * stt.SampleRate)
}

func (s *AssemblyAISocketSuite) TestTheSessionIsConfiguredByTheQueryString() {
	fake := newFakeSTT()
	fake.greeting = frame("begin_min_latency.json")
	defer fake.close()
	provider, _ := s.connect(fake, Options{
		Keyterms:         []string{"Mia", "hues of gold"},
		LanguageHints:    []string{"en", "es"},
		Mode:             ModeMinLatency,
		MinTurnSilenceMs: 160,
		MaxTurnSilenceMs: 1200,
	})
	defer func() { _ = provider.Close() }()

	query := (<-fake.requests).URL.Query()
	s.Equal(DefaultModel, query.Get("speech_model"))
	s.Equal("16000", query.Get("sample_rate"))
	s.Equal("pcm_s16le", query.Get("encoding"))
	s.Equal(ModeMinLatency, query.Get("mode"))
	s.Equal(`["Mia","hues of gold"]`, query.Get("keyterms_prompt"),
		"a list is one parameter holding a JSON array, which is the only way the server reads it")
	s.Equal(`["en","es"]`, query.Get("language_codes"))
	s.Equal("160", query.Get("min_turn_silence"))
	s.Equal("1200", query.Get("max_turn_silence"))
}

func (s *AssemblyAISocketSuite) TestACallGetsThePresetThatDoesNotEndATurnMidSentence() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	query := (<-fake.requests).URL.Query()
	s.Equal(ModeMaxAccuracy, query.Get("mode"),
		"the server's own preset ends a turn at a short pause in the middle of a sentence")
	for _, name := range []string{"min_turn_silence", "max_turn_silence", "keyterms_prompt", "language_codes"} {
		s.Emptyf(query.Get(name), "%s was not asked for", name)
	}
}

func (s *AssemblyAISocketSuite) TestTheKeyTravelsAsItIsInTheAuthorizationHeader() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	request := <-fake.requests
	s.Equal("secret", request.Header.Get("Authorization"),
		"the server refuses a key with Bearer in front of it as an invalid key")
	s.Empty(request.URL.Query().Get("token"), "the temporary token is for a browser, not for the router")
}

func (s *AssemblyAISocketSuite) TestARefusedSessionIsReportedFromStart() {
	err := s.startAgainst("error_invalid_key.json", Options{})

	s.ErrorContains(err, "Invalid API key")
	s.ErrorContains(err, "code 1008")
}

func (s *AssemblyAISocketSuite) TestASessionOpenedOnAnotherModelIsRefused() {
	// The server ignores a parameter it does not recognise rather than refusing it, so a
	// session on the wrong model is only visible in what Begin says it applied.
	err := s.startAgainst("begin_other_model.json", Options{})

	s.ErrorContains(err, "asked for model universal-3-6-pro and the server opened universal-streaming-english")
}

func (s *AssemblyAISocketSuite) TestASessionThatDidNotApplyTheAskedPresetIsRefused() {
	err := s.startAgainst("begin_balanced.json", Options{Mode: ModeMaxAccuracy})

	s.ErrorContains(err, "asked for mode max_accuracy and the server applied balanced")
}

func (s *AssemblyAISocketSuite) TestCallAudioIsGatheredIntoFramesTheServerAccepts() {
	// A call arrives 20 ms at a time and the server ends a session that is sent a frame
	// shorter than 50 ms, so nothing goes up until there is enough of it.
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.speak(provider, 20)
	s.speak(provider, 20)
	s.speak(provider, 20)

	s.Equal(60, s.nextAudio(conn), "three call frames make the first one long enough to send")
}

func (s *AssemblyAISocketSuite) TestAChunkLongerThanTheServerTakesIsSplit() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer func() { _ = provider.Close() }()

	s.speak(provider, 1500)

	s.Equal(1000, s.nextAudio(conn))
	s.Equal(500, s.nextAudio(conn))
}

func (s *AssemblyAISocketSuite) TestClosingSendsTheLastAudioAndAsksForTheTurnToSettle() {
	fake := newFakeSTT()
	defer fake.close()
	provider, conn := s.connect(fake, Options{FlushTimeout: 5 * time.Second})

	// A caller cut off mid-sentence leaves no silence for the turn detector to settle on,
	// so the tail is settled by terminating, and the server answers with the turn and
	// then with Termination, after which nothing else comes.
	served := make(chan []int, 1)
	go func() {
		var audio []int
		defer func() { served <- audio }()
		for {
			if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
				return
			}
			messageType, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if messageType == websocket.BinaryMessage {
				audio = append(audio, len(raw)*1000/(2*stt.SampleRate))
				continue
			}
			var message struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(raw, &message); err != nil || message.Type != clientTypeTerminate {
				return
			}
			_ = conn.WriteMessage(websocket.TextMessage, frame("turn_final.json"))
			_ = conn.WriteMessage(websocket.TextMessage, frame("termination.json"))
			return
		}
	}()

	s.speak(provider, 20)
	s.Require().NoError(provider.Close())

	s.Equal([]int{minChunkMs}, <-served,
		"the 20 ms still waiting goes up padded to the shortest frame the server takes")
	s.Equal("In a quiet village where the sky brushes the fields in hues of gold,", s.finalText(provider),
		"closing should not cut off the words that had not settled yet")
}

func (s *AssemblyAISocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	fake := newFakeSTT()
	defer fake.close()
	provider, _ := s.connect(fake, Options{FlushTimeout: 200 * time.Millisecond})

	s.speak(provider, 100)

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
}

func (s *AssemblyAISocketSuite) TestAServerThatHangsUpMidCallIsReported() {
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
func (s *AssemblyAISocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}
