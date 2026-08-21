package cartesia

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// fakeCartesia is a WebSocket server that speaks the context protocol, so the provider can
// be driven over a real connection without an API key.
type fakeCartesia struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	done   chan struct{}
	// request is what the client dialled, so the query string and headers can be asserted.
	request chan *http.Request
}

func newFakeCartesia() *fakeCartesia {
	fake := &fakeCartesia{
		conns:   make(chan *websocket.Conn, 1),
		done:    make(chan struct{}),
		request: make(chan *http.Request, 1),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.request <- r.Clone(context.Background())
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakeCartesia) baseURL() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeCartesia) close() {
	close(f.done)
	f.server.Close()
}

// accept returns the server side of the connection the provider opened.
func (f *fakeCartesia) accept() *websocket.Conn {
	select {
	case conn := <-f.conns:
		return conn
	case <-time.After(5 * time.Second):
		return nil
	}
}

// speak sends a frame of PCM audio for a context, the way the real server does.
func speak(conn *websocket.Conn, contextID string, samples []int16) error {
	pcm := audio.PcmData{Samples: samples, SampleRate: DefaultSampleRate, Channels: 1}
	payload, err := json.Marshal(map[string]any{
		"type":       typeChunk,
		"data":       base64.StdEncoding.EncodeToString(pcm.Bytes()),
		"context_id": contextID,
		"done":       false,
	})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

func finish(conn *websocket.Conn, contextID string) error {
	payload, err := json.Marshal(map[string]any{
		"type": typeDone, "context_id": contextID, "done": true,
	})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

type CartesiaSuite struct {
	suite.Suite
}

func TestCartesiaSuite(t *testing.T) {
	suite.Run(t, new(CartesiaSuite))
}

// newTTS returns a provider that is wired up but never connected.
func (s *CartesiaSuite) newTTS(options Options) *TTS {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// connect returns a started provider and the server side of its connection.
func (s *CartesiaSuite) connect(fake *fakeCartesia, options Options) (*TTS, *websocket.Conn) {
	options.BaseURL = fake.baseURL()
	provider := s.newTTS(options)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))

	conn := fake.accept()
	s.Require().NotNil(conn, "the provider should have connected")
	return provider, conn
}

// collect reads events until the predicate is satisfied or the wait runs out.
func (s *CartesiaSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
	var events []tts.Event
	deadline := time.After(5 * time.Second)

	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
			if until(event) {
				return events
			}
		case <-deadline:
			s.FailNow("timed out waiting for events")
			return events
		}
	}
}

// generations reads the generation frames the provider sent.
func (s *CartesiaSuite) generations(conn *websocket.Conn, want int) []generation {
	var messages []generation
	for len(messages) < want {
		s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
		_, raw, err := conn.ReadMessage()
		s.Require().NoError(err)

		var message generation
		s.Require().NoError(json.Unmarshal(raw, &message))
		messages = append(messages, message)
	}
	return messages
}

func (s *CartesiaSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("CARTESIA_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *CartesiaSuite) TestNewFallsBackToEnv() {
	s.T().Setenv("CARTESIA_API_KEY", "from-env")
	s.T().Setenv("CARTESIA_VOICE_ID", "voice-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
	s.Equal("voice-from-env", provider.options.VoiceID)
}

func (s *CartesiaSuite) TestNewDefaultsToSonicAndAVoice() {
	s.T().Setenv("CARTESIA_VOICE_ID", "")

	provider := s.newTTS(Options{})
	s.Equal(DefaultModel, provider.Model())
	s.Equal(DefaultVoiceID, provider.options.VoiceID)
	s.Equal(DefaultSampleRate, provider.SampleRate())
	s.Equal(ProviderName, provider.Provider())
	s.True(provider.Streaming(), "the model generates from partial text")
}

func (s *CartesiaSuite) TestNewRejectsASampleRateTheOutputFormatCannotCarry() {
	_, err := New(Options{APIKey: "k", SampleRate: 12_345})
	s.ErrorContains(err, "sample rate 12345 is not one of")
}

func (s *CartesiaSuite) TestTheEndpointPinsTheAPIVersion() {
	// Cartesia dates its API rather than numbering it, and rejects a request that does
	// not say which shapes it was written against.
	url := s.newTTS(Options{}).url()

	s.Contains(url, "/tts/websocket")
	s.Contains(url, "cartesia_version="+apiVersion)
}

func (s *CartesiaSuite) TestTheAPIKeyTravelsAsAHeader() {
	fake := newFakeCartesia()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret-key"})
	defer provider.Close()

	request := <-fake.request
	s.Equal("secret-key", request.Header.Get("X-API-Key"))
	s.NotContains(request.URL.RawQuery, "secret-key", "a key in the query string ends up in logs")
}

func (s *CartesiaSuite) TestSynthesizeFailsBeforeStart() {
	err := s.newTTS(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *CartesiaSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newTTS(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	for range provider.Events() {
	}
}

func (s *CartesiaSuite) TestAnUtteranceStreamsTextOnOneContextAndThenEndsIt() {
	fake := newFakeCartesia()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1", SampleRate: 16_000})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello"}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: " world"}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Final: true}))

	messages := s.generations(conn, 3)

	s.Equal("hello", messages[0].Transcript)
	s.Equal("u1", messages[0].ContextID)
	s.Equal("v1", messages[0].Voice.ID)
	s.Equal(DefaultModel, messages[0].ModelID)
	s.Equal(outputFormat{Container: "raw", Encoding: "pcm_s16le", SampleRate: 16_000},
		messages[0].OutputFormat)
	s.True(messages[0].Continue, "more of the turn is still coming")

	s.Equal(" world", messages[1].Transcript)
	s.True(messages[1].Continue)

	s.Empty(messages[2].Transcript, "the turn ends with nothing left to say")
	s.False(messages[2].Continue, "ending the context is what makes the tail generate at once")
}

func (s *CartesiaSuite) TestTheServerIsToldNotToBufferOnTopOfTheAgent() {
	// The agent already waits for a whole sentence before sending it. Letting Cartesia
	// wait again for more text would add its buffer delay to every reply.
	fake := newFakeCartesia()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello.", Final: true}))

	s.Zero(s.generations(conn, 1)[0].MaxBufferDelayMs)
}

func (s *CartesiaSuite) TestAudioAndTheDoneFrameSettleTheSynthesis() {
	fake := newFakeCartesia()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello there", Final: true}))

	// 2400 samples at 24 kHz is 100 ms of speech.
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.Require().NoError(finish(conn, "u1"))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var chunks []tts.AudioChunk
	var complete tts.SynthesisComplete
	var started tts.SynthesisStarted
	for _, event := range events {
		switch typed := event.(type) {
		case tts.SynthesisStarted:
			started = typed
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			complete = typed
		}
	}

	s.Equal("u1", started.SynthesisID, "the caller should learn the utterance was accepted")
	s.Equal(DefaultVoiceID, started.Voice)
	s.Require().Len(chunks, 2)
	s.Equal(0, chunks[0].Index)
	s.Equal(1, chunks[1].Index, "chunks should be numbered so playback order survives")
	s.Equal(DefaultSampleRate, chunks[0].Audio.SampleRate)

	s.Equal("u1", complete.SynthesisID)
	s.EqualValues(len("hello there"), complete.Characters)
	s.InDelta(200.0, complete.AudioDurationMs, 1.0)
	s.Positive(complete.TimeToFirstByteMs, "the wait for the first sound is the number that matters")
	s.False(complete.Interrupted)
}

func (s *CartesiaSuite) TestInterruptCancelsTheContextAndDropsAudioStillOnTheWire() {
	fake := newFakeCartesia()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "a long answer", Final: true}))
	s.generations(conn, 1)
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})

	s.Require().NoError(provider.Interrupt())

	// Audio the server had already sent must not reach the call after barge-in.
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete := events[len(events)-1].(tts.SynthesisComplete)

	s.True(complete.Interrupted)
	s.InDelta(100.0, complete.AudioDurationMs, 1.0,
		"only the audio heard before the interruption counts")

	var cancelled cancellation
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	s.Require().NoError(json.Unmarshal(raw, &cancelled))
	s.Equal("u1", cancelled.ContextID)
	s.True(cancelled.Cancel)
}

func (s *CartesiaSuite) TestAnUtteranceCannotChangeVoiceHalfway() {
	// Cartesia binds a context to the voice it was opened with, so a change partway
	// through would be one answer said in two voices.
	fake := newFakeCartesia()
	defer fake.close()
	provider, _ := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello"}))

	err := provider.Synthesize(tts.Request{ID: "u1", Text: " world", Voice: "v2"})
	s.ErrorContains(err, "being said in voice v1, not v2")
}

func (s *CartesiaSuite) TestAnUtteranceCanPickItsOwnVoice() {
	fake := newFakeCartesia()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "session-voice"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{
		ID: "u1", Text: "hello", Voice: "chosen-voice", Final: true,
	}))

	s.Equal("chosen-voice", s.generations(conn, 1)[0].Voice.ID)
}

func (s *CartesiaSuite) TestAServerErrorIsReportedAndSettlesTheUtterance() {
	// An error ends the context upstream, so an utterance left in flight after one would
	// never be settled and the turn would never finish being spoken.
	fake := newFakeCartesia()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))

	payload, err := json.Marshal(map[string]any{
		"type": typeError, "context_id": "u1", "done": true,
		"title": "Invalid model", "error_code": "model_not_found",
	})
	s.Require().NoError(err)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, payload))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var failure tts.Error
	for _, event := range events {
		if typed, ok := event.(tts.Error); ok {
			failure = typed
		}
	}
	s.Equal("u1", failure.SynthesisID)
	s.ErrorContains(failure.Err, "Invalid model")
	s.ErrorContains(failure.Err, "model_not_found")
	s.False(failure.Fatal, "one rejected utterance does not end the session")
}

func (s *CartesiaSuite) TestALostConnectionSettlesWhatWasInFlight() {
	fake := newFakeCartesia()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.Require().NoError(conn.Close())

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete := events[len(events)-1].(tts.SynthesisComplete)

	s.Equal("u1", complete.SynthesisID)
	s.True(complete.Interrupted, "work cut short by a dropped connection is not work completed")
}
