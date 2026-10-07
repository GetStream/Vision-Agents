package microsoft

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// MicrosoftSocketSuite exercises the provider against a server that is answering, which is
// where the credentials, the session configuration, the audio encoding, the commit at the
// end of a turn and the wait for the tail of a call live.
type MicrosoftSocketSuite struct {
	suite.Suite
	fake *fakeFoundry
}

func TestMicrosoftSocketSuite(t *testing.T) {
	suite.Run(t, new(MicrosoftSocketSuite))
}

// fakeFoundry is a Foundry realtime socket. It opens a session and accepts its
// configuration as the real one does, so a test starts from a connection ready for audio.
type fakeFoundry struct {
	server *httptest.Server

	mu sync.Mutex
	// opening is the frame the server starts with, and answer its reply to session.update.
	opening string
	answer  string
	// requests carries the upgrade request, updates the configuration frame the provider
	// sent, and conns the server side of each connection once it is configured.
	requests chan *http.Request
	updates  chan clientMessage
	conns    chan *websocket.Conn
}

func (f *fakeFoundry) reset() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.opening = `{"type":"session.created","session":{"id":"sess_1"}}`
	f.answer = `{"type":"session.updated"}`
	f.requests = make(chan *http.Request, 1)
	f.updates = make(chan clientMessage, 1)
	f.conns = make(chan *websocket.Conn, 1)
}

func (f *fakeFoundry) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	opening, answer := f.opening, f.answer
	requests, updates, conns := f.requests, f.updates, f.conns
	f.mu.Unlock()

	requests <- r
	upgrader := websocket.Upgrader{}
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	if err := conn.WriteMessage(websocket.TextMessage, []byte(opening)); err != nil {
		return
	}

	_, raw, err := conn.ReadMessage()
	if err != nil {
		return
	}
	var update clientMessage
	if err := json.Unmarshal(raw, &update); err != nil {
		return
	}
	updates <- update
	if err := conn.WriteMessage(websocket.TextMessage, []byte(answer)); err != nil {
		return
	}
	conns <- conn
}

func (s *MicrosoftSocketSuite) SetupSuite() {
	s.fake = &fakeFoundry{}
	s.fake.reset()
	s.fake.server = httptest.NewServer(http.HandlerFunc(s.fake.serve))
}

func (s *MicrosoftSocketSuite) SetupTest() {
	s.fake.reset()
}

// build returns an unstarted provider pointed at the fake.
func (s *MicrosoftSocketSuite) build(options Options) *STT {
	options.Endpoint = s.fake.server.URL
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

// connect returns a started provider and the server side of its connection.
func (s *MicrosoftSocketSuite) connect(options Options) (*STT, *websocket.Conn) {
	provider := s.build(options)
	s.Require().NoError(provider.Start(s.ctx()))

	select {
	case conn := <-s.fake.conns:
		return provider, conn
	case <-time.After(5 * time.Second):
		s.FailNow("the provider never connected")
		return nil, nil
	}
}

func (s *MicrosoftSocketSuite) ctx() context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	return ctx
}

// speak sends one chunk of audio.
func (s *MicrosoftSocketSuite) speak(provider *STT, samples []int16) {
	pcm := stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, stt.Participant{ID: "alice", UserID: "alice"}))
}

// nextFrame reads the next message the provider sent upstream.
func (s *MicrosoftSocketSuite) nextFrame(conn *websocket.Conn) clientMessage {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)

	var frame clientMessage
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

// say sends a server frame down the connection.
func (s *MicrosoftSocketSuite) say(conn *websocket.Conn, frame string) {
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, []byte(frame)))
}

// nextFinal returns the text of the next settled turn.
func (s *MicrosoftSocketSuite) nextFinal(provider *STT) string {
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

// finalText is the text of the last settled turn among everything the session emitted.
// Close has already run, so the channel is closed and reading it to the end terminates.
func (s *MicrosoftSocketSuite) finalText(provider *STT) string {
	var settled string
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			settled = transcript.Text
		}
	}
	return settled
}

func (s *MicrosoftSocketSuite) TestTheSessionOpensOnTheRealtimePathAskingToTranscribe() {
	provider, _ := s.connect(Options{})
	defer func() { _ = provider.Close() }()

	opened := <-s.fake.requests
	s.Equal(realtimePath, opened.URL.Path)
	s.Equal("transcription", opened.URL.Query().Get("intent"),
		"the realtime socket also serves conversations, and one would try to answer back")
}

func (s *MicrosoftSocketSuite) TestTheKeyGoesInTheAPIKeyHeader() {
	provider, _ := s.connect(Options{APIKey: "secret"})
	defer func() { _ = provider.Close() }()

	header := (<-s.fake.requests).Header
	s.Equal("secret", header.Get("api-key"))
	s.Empty(header.Get("Authorization"), "a key is not an Entra bearer token")
}

func (s *MicrosoftSocketSuite) TestTheSessionIsConfiguredForTheDeploymentBeforeAnyAudio() {
	provider, _ := s.connect(Options{Deployment: "mai-stream", Language: "de"})
	defer func() { _ = provider.Close() }()

	update := <-s.fake.updates
	s.Equal(clientTypeUpdate, update.Type)
	s.Require().NotNil(update.Session)
	s.Equal("transcription", update.Session.Type)
	s.Equal("mai-stream", update.Session.Audio.Input.Transcription.Model)
	s.Require().NotNil(update.Session.Audio.Input.Transcription.Language)
	s.Equal("de", *update.Session.Audio.Input.Transcription.Language)
	s.Equal(audioFormat{Type: "audio/pcm", Rate: stt.SampleRate}, update.Session.Audio.Input.Format)
}

func (s *MicrosoftSocketSuite) TestAudioArrivesBase64EncodedInsideAFrame() {
	provider, conn := s.connect(Options{})
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

func (s *MicrosoftSocketSuite) TestStartWaitsForTheSessionToOpen() {
	s.fake.mu.Lock()
	s.fake.opening = `{"type":"conversation.item.input_audio_transcription.delta","delta":"too soon"}`
	s.fake.mu.Unlock()

	err := s.build(Options{}).Start(s.ctx())

	s.ErrorContains(err, "expected \"session.created\"")
}

func (s *MicrosoftSocketSuite) TestStartReportsAConfigurationTheServerRejected() {
	s.fake.mu.Lock()
	s.fake.answer = `{"type":"error","error":{"message":"deployment not found"}}`
	s.fake.mu.Unlock()

	err := s.build(Options{}).Start(s.ctx())

	s.ErrorContains(err, "deployment not found")
}

// TestTheTurnIsCommittedOnceTheWordsStopChanging is the boundary itself. The server never
// commits on its own, so a turn the provider did not commit would never settle.
func (s *MicrosoftSocketSuite) TestTheTurnIsCommittedOnceTheWordsStopChanging() {
	provider, conn := s.connect(Options{TurnGrace: 50 * time.Millisecond})
	defer func() { _ = provider.Close() }()

	s.speak(provider, []int16{1, 2, 3})
	s.Equal(clientTypeAppend, s.nextFrame(conn).Type)

	s.say(conn, `{"type":"conversation.item.input_audio_transcription.delta","delta":"Hello"}`)
	s.say(conn, `{"type":"conversation.item.input_audio_transcription.intermediate","intermediate":" there"}`)
	s.Equal(clientTypeCommit, s.nextFrame(conn).Type, "silence should commit the turn")

	s.say(conn, `{"type":"input_audio_buffer.committed"}`)
	s.say(conn, `{"type":"conversation.item.input_audio_transcription.completed","transcript":"Hello there!"}`)
	s.Equal("Hello there!", s.nextFinal(provider))
}

func (s *MicrosoftSocketSuite) TestClosingCommitsTheAudioTheServerHadNotTranscribed() {
	provider, conn := s.connect(Options{FlushTimeout: 5 * time.Second, TurnGrace: time.Hour})

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

func (s *MicrosoftSocketSuite) TestClosingASettledCallDoesNotWaitOutTheTimeout() {
	// Nothing is outstanding once a turn has settled, and no further one is coming.
	provider, conn := s.connect(Options{FlushTimeout: time.Minute})

	s.speak(provider, []int16{1, 2, 3})
	s.say(conn, `{"type":"conversation.item.input_audio_transcription.completed","transcript":"In a quiet village."}`)
	s.Require().Equal("In a quiet village.", s.nextFinal(provider))

	closing := time.Now()
	s.Require().NoError(provider.Close())
	s.Less(time.Since(closing), 5*time.Second,
		"a call whose last turn already settled should not wait for another")
}

func (s *MicrosoftSocketSuite) TestClosingGivesUpOnAServerThatNeverAnswers() {
	provider, _ := s.connect(Options{FlushTimeout: 200 * time.Millisecond, TurnGrace: time.Hour})

	provider.handleMessage(delta("in a quiet"))
	s.speak(provider, []int16{1, 2, 3})

	closing := time.Now()
	s.Require().NoError(provider.Close())

	s.Less(time.Since(closing), 3*time.Second, "a silent server should not hold up a hangup")
	s.Equal("in a quiet", s.finalText(provider), "what was heard is still settled on hangup")
}
