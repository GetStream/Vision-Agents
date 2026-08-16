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

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// fakeElevenLabs is a WebSocket server that speaks the multi-stream protocol, so the
// provider can be driven over a real connection without an API key.
type fakeElevenLabs struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	done   chan struct{}
	// url is what the client dialled, so the query string can be asserted.
	url chan string
}

func newFakeElevenLabs() *fakeElevenLabs {
	fake := &fakeElevenLabs{
		conns: make(chan *websocket.Conn, 1),
		done:  make(chan struct{}),
		url:   make(chan string, 1),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.url <- r.URL.String()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakeElevenLabs) baseURL() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeElevenLabs) close() {
	close(f.done)
	f.server.Close()
}

// accept returns the server side of the connection the provider opened.
func (f *fakeElevenLabs) accept() *websocket.Conn {
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
		"audio":     base64.StdEncoding.EncodeToString(pcm.Bytes()),
		"contextId": contextID,
	})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

func finish(conn *websocket.Conn, contextID string) error {
	payload, err := json.Marshal(map[string]any{"contextId": contextID, "isFinal": true})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

type ElevenLabsSuite struct {
	suite.Suite
}

func TestElevenLabsSuite(t *testing.T) {
	suite.Run(t, new(ElevenLabsSuite))
}

// newTTS returns a provider that is wired up but never connected.
func (s *ElevenLabsSuite) newTTS(options Options) *TTS {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// connect returns a started provider and the server side of its connection.
func (s *ElevenLabsSuite) connect(fake *fakeElevenLabs, options Options) (*TTS, *websocket.Conn) {
	options.BaseURL = fake.baseURL()
	// The fake never answers close_socket, and no test is about that wait.
	if options.CloseTimeout == 0 {
		options.CloseTimeout = 50 * time.Millisecond
	}
	provider := s.newTTS(options)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))

	conn := fake.accept()
	s.Require().NotNil(conn, "the provider should have connected")
	return provider, conn
}

// collect reads events until the predicate is satisfied or the wait runs out.
func (s *ElevenLabsSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
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

// clientMessages reads the frames the provider sent, until the server read times out.
func (s *ElevenLabsSuite) clientMessages(conn *websocket.Conn, want int) []clientMessage {
	var messages []clientMessage
	for len(messages) < want {
		s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
		_, raw, err := conn.ReadMessage()
		s.Require().NoError(err)

		var message clientMessage
		s.Require().NoError(json.Unmarshal(raw, &message))
		messages = append(messages, message)
	}
	return messages
}

func (s *ElevenLabsSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("ELEVENLABS_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *ElevenLabsSuite) TestNewFallsBackToEnv() {
	s.T().Setenv("ELEVENLABS_API_KEY", "from-env")
	s.T().Setenv("ELEVENLABS_VOICE_ID", "voice-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
	s.Equal("voice-from-env", provider.options.VoiceID)
}

func (s *ElevenLabsSuite) TestNewDefaultsToTheLowLatencyModelAndAVoice() {
	s.T().Setenv("ELEVENLABS_VOICE_ID", "")

	provider := s.newTTS(Options{})
	s.Equal(DefaultModel, provider.Model())
	s.Equal(DefaultVoiceID, provider.options.VoiceID)
	s.Equal(DefaultSampleRate, provider.SampleRate())
	s.Equal(ProviderName, provider.Provider())
	s.True(provider.Streaming(), "the model generates from partial text")
}

func (s *ElevenLabsSuite) TestNewRejectsASampleRateTheOutputFormatCannotCarry() {
	_, err := New(Options{APIKey: "k", SampleRate: 12_345})
	s.ErrorContains(err, "sample rate 12345 is not one of")
}

func (s *ElevenLabsSuite) TestTheEndpointCarriesTheVoiceModelAndFormat() {
	url := s.newTTS(Options{VoiceID: "v1", SampleRate: 16_000}).url()

	s.Contains(url, "/v1/text-to-speech/v1/multi-stream-input")
	s.Contains(url, "model_id="+DefaultModel)
	s.Contains(url, "output_format=pcm_16000")
	s.Contains(url, "auto_mode=true")
}

func (s *ElevenLabsSuite) TestALanguageIsOnlySentToAModelThatAcceptsOne() {
	multilingual := s.newTTS(Options{Model: "eleven_multilingual_v2", Language: "ES"}).url()
	s.Contains(multilingual, "language_code=es", "the code should be lowercased")

	monolingual := s.newTTS(Options{Model: "eleven_monolingual_v1", Language: "es"}).url()
	s.NotContains(monolingual, "language_code", "sending one here would be rejected upstream")
}

func (s *ElevenLabsSuite) TestSynthesizeFailsBeforeStart() {
	err := s.newTTS(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *ElevenLabsSuite) TestSynthesizeRejectsAVoiceTheConnectionCannotSpeak() {
	provider := s.newTTS(Options{VoiceID: "bound-voice"})

	err := provider.Synthesize(tts.Request{Text: "hello", Voice: "other-voice", Final: true})
	s.ErrorContains(err, "the connection is bound to voice bound-voice")
}

func (s *ElevenLabsSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newTTS(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	for range provider.Events() {
	}
}

func (s *ElevenLabsSuite) TestAnUtteranceOpensAContextStreamsTextAndCloses() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello"}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "world", Final: true}))

	messages := s.clientMessages(conn, 4)

	s.Equal(" ", messages[0].Text, "the context is opened with a space")
	s.Equal("u1", messages[0].ContextID)
	s.Require().NotNil(messages[0].Generation, "the first chunk threshold should be tuned down")

	s.Equal("hello ", messages[1].Text, "deltas need a trailing space to stay separate words")
	s.Equal("world ", messages[2].Text)
	s.True(messages[3].CloseContext, "a final request should close the context")
	s.Equal("u1", messages[3].ContextID)
}

func (s *ElevenLabsSuite) TestAudioAndTheFinalFrameSettleTheSynthesis() {
	fake := newFakeElevenLabs()
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
	var started bool
	for _, event := range events {
		switch typed := event.(type) {
		case tts.SynthesisStarted:
			started = true
			s.Equal("u1", typed.SynthesisID)
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			complete = typed
		}
	}

	s.True(started, "the caller should learn the utterance was accepted")
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

func (s *ElevenLabsSuite) TestInterruptEndsTheUtteranceAndDropsAudioStillOnTheWire() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "a long sentence", Final: true}))
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))

	// Wait for the first chunk so the interrupt lands mid-utterance.
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})

	s.Require().NoError(provider.Interrupt())

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted, "barge-in should be visible in the stat row")
	s.InDelta(100.0, complete.AudioDurationMs, 1.0, "only the audio that played should be billed")

	// Audio the server had already generated must not reach the caller.
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	select {
	case event := <-provider.Events():
		s.Failf("stale audio", "an interrupted utterance should stay silent, got %T", event)
	case <-time.After(200 * time.Millisecond):
	}
}

func (s *ElevenLabsSuite) TestAServerErrorIsReportedWithoutKillingTheSession() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage,
		[]byte(`{"contextId":"u1","error":"voice not found"}`)))

	events := s.collect(provider, func(event tts.Event) bool {
		_, failed := event.(tts.Error)
		return failed
	})
	failure := events[len(events)-1].(tts.Error)

	s.ErrorContains(failure.Err, "voice not found")
	s.Equal("u1", failure.SynthesisID)
	s.False(failure.Fatal, "one bad utterance should not close the connection")
}

func (s *ElevenLabsSuite) TestCloseSettlesAnUtteranceThatNeverFinished() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))

	collected := make(chan []tts.Event, 1)
	go func() {
		var events []tts.Event
		for event := range provider.Events() {
			events = append(events, event)
		}
		collected <- events
	}()

	s.Require().NoError(provider.Close())

	var events []tts.Event
	select {
	case events = <-collected:
	case <-time.After(5 * time.Second):
		s.FailNow("closing should close the event channel")
	}

	var completes []tts.SynthesisComplete
	for _, event := range events {
		if complete, ok := event.(tts.SynthesisComplete); ok {
			completes = append(completes, complete)
		}
	}
	s.Require().Len(completes, 1, "an unfinished utterance should still be accounted for")
	s.True(completes[0].Interrupted)
}

func (s *ElevenLabsSuite) TestALostConnectionSettlesWhatWasInFlight() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))

	// A rejection closes the socket, so nothing in flight will ever finish on its own.
	s.Require().NoError(conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.ClosePolicyViolation, "voice does not exist"),
		time.Now().Add(time.Second),
	))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var failure tts.Error
	var sawFailure bool
	for _, event := range events {
		if typed, ok := event.(tts.Error); ok {
			failure, sawFailure = typed, true
		}
	}
	s.Require().True(sawFailure, "a dropped connection should be reported")
	s.True(failure.Fatal)

	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted, "the caller should not be left waiting for audio")
}

func (s *ElevenLabsSuite) TestSatisfiesTTSInterface() {
	var _ tts.TTS = s.newTTS(Options{})
}
