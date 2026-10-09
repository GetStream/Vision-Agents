package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
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

func (s *ElevenLabsSuite) TestThisSocketsModelsAreNeverOfferedAudioTags() {
	// A model that cannot act a tag reads the word inside the brackets out, so asking for
	// one would make every reply worse rather than better.
	flash := s.newTTS(Options{Model: DefaultModel})

	s.False(flash.Performs())
	s.Empty(flash.Prompt())
}

func (s *ElevenLabsSuite) TestNewRejectsASampleRateTheOutputFormatCannotCarry() {
	_, err := New(Options{APIKey: "k", SampleRate: 12_345})
	s.ErrorContains(err, "sample rate 12345 is not one of")
}

func (s *ElevenLabsSuite) TestASpeedOutsideTheVendorsRangeIsRefused() {
	_, err := New(Options{APIKey: "k", Speed: 1.5})
	s.ErrorContains(err, "speed 1.5 is outside 0.7 to 1.2")

	_, err = New(Options{APIKey: "k", Speed: 0.5})
	s.ErrorContains(err, "speed 0.5 is outside 0.7 to 1.2")
}

func (s *ElevenLabsSuite) TestEveryContextIsOpenedAtTheSpeedAskedFor() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1", Speed: 0.9})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))

	opened := s.clientMessages(conn, 1)[0]
	s.Require().NotNil(opened.VoiceSettings)
	s.Equal(0.9, opened.VoiceSettings.Speed)
}

func (s *ElevenLabsSuite) TestNoSpeedIsSentWhenNoneWasAskedFor() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))

	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	s.NotContains(string(raw), `"speed"`, "the voice keeps its own pace unless told otherwise")
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
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "u1"
	})
	var failure *tts.Error
	var complete *tts.SynthesisComplete
	for _, event := range events {
		switch typed := event.(type) {
		case tts.Error:
			failure = &typed
		case tts.SynthesisComplete:
			complete = &typed
		}
	}
	s.Require().NotNil(failure)
	s.ErrorContains(failure.Err, "voice not found")
	s.Equal("u1", failure.SynthesisID)
	s.False(failure.Fatal, "one bad utterance should not close the connection")
	s.Require().NotNil(complete)
	s.True(complete.Interrupted, "rejected contexts are terminal and must be accounted for")
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
	s.Equal("u1", failure.SynthesisID, "connection loss is attributed to the active utterance")
	s.False(failure.Fatal, "the session can reconnect for a later synthesis")

	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted, "the caller should not be left waiting for audio")
}

func (s *ElevenLabsSuite) TestIdleSocketLossReconnectsForANewSynthesis() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, firstConn := s.connect(fake, Options{})
	defer provider.Close()
	oldSocket := provider.conn
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	s.Require().NoError(firstConn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "idle"),
		time.Now().Add(time.Second),
	))
	disconnected := s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Disconnected)
		return ok
	})
	s.Require().IsType(tts.Disconnected{}, disconnected[len(disconnected)-1])

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "after-idle", Text: "hello", Final: true}))
	secondConn := fake.accept()
	s.Require().NotNil(secondConn, "a later utterance should establish a replacement socket")
	<-fake.url
	messages := s.clientMessages(secondConn, 3)
	s.Equal("after-idle", messages[0].ContextID)
	s.Equal("after-idle", messages[1].ContextID)
	s.Equal("after-idle", messages[2].ContextID)

	// A late frame delivered by the old reader must not be attributed to the new socket.
	lateAudio := base64.StdEncoding.EncodeToString([]byte{0, 0, 0, 0})
	provider.handleMessage(oldSocket, serverMessage{ContextID: "after-idle", Audio: lateAudio, IsFinal: true})
	s.Require().NoError(speak(secondConn, "after-idle", make([]int16, 240)))
	s.Require().NoError(finish(secondConn, "after-idle"))
	events := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "after-idle"
	})
	var chunks int
	for _, event := range events {
		if chunk, ok := event.(tts.AudioChunk); ok && chunk.SynthesisID == "after-idle" {
			chunks++
		}
	}
	s.Equal(1, chunks, "only the replacement connection's audio should be forwarded")
}

func (s *ElevenLabsSuite) TestDisconnectInterruptsOnceAndNeverReopensTheFailedID() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, firstConn := s.connect(fake, Options{})
	defer provider.Close()
	oldSocket := provider.conn
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "partial", Text: "hello"}))
	s.clientMessages(firstConn, 2)
	s.Require().NoError(speak(firstConn, "partial", make([]int16, 240)))
	s.collect(provider, func(event tts.Event) bool {
		chunk, ok := event.(tts.AudioChunk)
		return ok && chunk.SynthesisID == "partial"
	})
	firstConn.Close()

	failed := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "partial"
	})
	var errorsForID, completesForID int
	for _, event := range failed {
		switch typed := event.(type) {
		case tts.Error:
			if typed.SynthesisID == "partial" {
				errorsForID++
			}
		case tts.SynthesisComplete:
			if typed.SynthesisID == "partial" {
				completesForID++
				s.True(typed.Interrupted)
			}
		}
	}
	s.Equal(1, errorsForID)
	s.Equal(1, completesForID)

	// These late input frames belong to the already failed stream and must be discarded.
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "partial", Text: "late", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "next", Text: "new", Final: true}))
	secondConn := fake.accept()
	s.Require().NotNil(secondConn)
	<-fake.url
	messages := s.clientMessages(secondConn, 3)
	for _, message := range messages {
		s.Equal("next", message.ContextID, "failed input must never be replayed")
		s.NotContains(message.Text, "late")
	}

	lateAudio := base64.StdEncoding.EncodeToString([]byte{0, 0, 0, 0})
	provider.handleMessage(oldSocket, serverMessage{ContextID: "next", Audio: lateAudio, IsFinal: true})
	s.Require().NoError(speak(secondConn, "next", make([]int16, 240)))
	s.Require().NoError(finish(secondConn, "next"))
	newEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "next"
	})
	for _, event := range newEvents {
		switch typed := event.(type) {
		case tts.AudioChunk:
			s.Equal("next", typed.SynthesisID)
		case tts.Error:
			s.NotEqual("partial", typed.SynthesisID, "a terminal old ID must not fail again")
		case tts.SynthesisComplete:
			s.NotEqual("partial", typed.SynthesisID, "a terminal old ID must not complete again")
		}
	}
	select {
	case event := <-provider.Events():
		switch typed := event.(type) {
		case tts.AudioChunk:
			s.NotEqual("partial", typed.SynthesisID)
		case tts.Error:
			s.NotEqual("partial", typed.SynthesisID)
		case tts.SynthesisComplete:
			s.NotEqual("partial", typed.SynthesisID, "the failed ID must have exactly one completion")
		}
	case <-time.After(100 * time.Millisecond):
	}
}

func (s *ElevenLabsSuite) TestContextFailureSettlesOnlyThatIDAndTheSocketStaysUsable() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()
	clientBefore := provider.Client()
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "rejected", Text: "bad", Final: true}))
	s.clientMessages(conn, 3)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage,
		[]byte(`{"contextId":"rejected","error":"context rejected"}`)))
	failed := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "rejected"
	})
	var errs, dones int
	for _, event := range failed {
		switch typed := event.(type) {
		case tts.Error:
			if typed.SynthesisID == "rejected" {
				errs++
			}
		case tts.SynthesisComplete:
			if typed.SynthesisID == "rejected" {
				dones++
				s.True(typed.Interrupted)
			}
		}
	}
	s.Equal(1, errs)
	s.Equal(1, dones)

	// Same-ID late final is suppressed; a different ID proceeds on the same socket.
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "rejected", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "good", Text: "hello", Final: true}))
	s.clientMessages(conn, 3)
	s.Require().NoError(speak(conn, "good", make([]int16, 240)))
	s.Require().NoError(finish(conn, "good"))
	s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "good"
	})
	s.Same(clientBefore, provider.Client(), "a context rejection should not replace a healthy socket")
	s.NotNil(conn, "the same server-side connection remains open")
}

func (s *ElevenLabsSuite) TestSessionFailureWithoutContextSettlesEachActiveID() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	for _, id := range []string{"one", "two"} {
		s.Require().NoError(provider.Synthesize(tts.Request{ID: id, Text: id}))
	}
	s.clientMessages(conn, 4)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage,
		[]byte(`{"error":"upstream session rejected"}`)))

	completions := map[string]bool{}
	errorsByID := map[string]int{}
	deadline := time.After(5 * time.Second)
	for len(completions) < 2 {
		select {
		case event := <-provider.Events():
			switch typed := event.(type) {
			case tts.Error:
				s.NotEmpty(typed.SynthesisID, "session failure is attributed per active utterance")
				errorsByID[typed.SynthesisID]++
			case tts.SynthesisComplete:
				completions[typed.SynthesisID] = typed.Interrupted
			}
		case <-deadline:
			s.FailNow("timed out waiting for per-ID terminal events")
		}
	}
	s.Equal(map[string]bool{"one": true, "two": true}, completions)
	s.Equal(map[string]int{"one": 1, "two": 1}, errorsByID)
}

func (s *ElevenLabsSuite) TestConcurrentNewIDsShareOneReplacementSocket() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, firstConn := s.connect(fake, Options{})
	defer provider.Close()
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})
	firstConn.Close()
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Disconnected)
		return ok
	})

	results := make(chan error, 2)
	for _, id := range []string{"new-a", "new-b"} {
		id := id
		go func() {
			results <- provider.Synthesize(tts.Request{ID: id, Text: id, Final: true})
		}()
	}
	s.NoError(<-results)
	s.NoError(<-results)
	secondConn := fake.accept()
	s.Require().NotNil(secondConn)
	<-fake.url
	messages := s.clientMessages(secondConn, 6)
	seen := map[string]map[string]bool{}
	for _, message := range messages {
		if seen[message.ContextID] == nil {
			seen[message.ContextID] = map[string]bool{}
		}
		if message.CloseContext {
			seen[message.ContextID]["closed"] = true
		} else if message.Text == " " {
			seen[message.ContextID]["opened"] = true
		} else {
			seen[message.ContextID]["text"] = true
		}
	}
	s.Require().Len(seen, 2, "both new IDs should use the same replacement connection")
	for _, id := range []string{"new-a", "new-b"} {
		s.True(seen[id]["opened"])
		s.True(seen[id]["text"])
		s.True(seen[id]["closed"])
	}

	for _, id := range []string{"new-a", "new-b"} {
		s.NoError(speak(secondConn, id, make([]int16, 240)))
		s.NoError(finish(secondConn, id))
	}
	done := map[string]bool{}
	deadline := time.After(5 * time.Second)
	for len(done) < 2 {
		select {
		case event := <-provider.Events():
			if complete, ok := event.(tts.SynthesisComplete); ok {
				done[complete.SynthesisID] = true
			}
		case <-deadline:
			s.FailNow("timed out waiting for both replacement utterances")
		}
	}
}

func (s *ElevenLabsSuite) TestCloseCancelsAnInProgressHandshake() {
	requestSeen := make(chan struct{}, 1)
	releaseHandler := make(chan struct{})
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestSeen <- struct{}{}
		<-releaseHandler
		_, _ = upgrader.Upgrade(w, r, nil)
	}))
	defer func() {
		close(releaseHandler)
		server.Close()
	}()

	provider := s.newTTS(Options{
		BaseURL:          "ws" + strings.TrimPrefix(server.URL, "http"),
		HandshakeTimeout: 15 * time.Second,
		CloseTimeout:     100 * time.Millisecond,
	})
	startResult := make(chan error, 1)
	go func() { startResult <- provider.Start(context.Background()) }()
	select {
	case <-requestSeen:
	case <-time.After(2 * time.Second):
		s.FailNow("the test server did not receive the handshake")
	}

	closed := make(chan error, 1)
	startedClose := time.Now()
	go func() { closed <- provider.Close() }()
	select {
	case err := <-closed:
		s.NoError(err)
		s.Less(time.Since(startedClose), time.Second, "Close should cancel DialContext before waiting")
	case <-time.After(time.Second):
		s.FailNow("Close did not cancel the stalled handshake")
	}
	select {
	case err := <-startResult:
		s.Error(err, "the canceled handshake should not install a socket")
	case <-time.After(time.Second):
		s.FailNow("Start did not return after its lifetime was canceled")
	}
}

func (s *ElevenLabsSuite) TestInitialDialContextObservesCallerAndSessionCancellation() {
	for _, cancelSession := range []bool{false, true} {
		caller, cancelCaller := context.WithCancel(context.Background())
		session, cancelSessionLifetime := context.WithCancel(context.Background())
		dialCtx, release := mergeDialContext(caller, session)
		if cancelSession {
			cancelSessionLifetime()
		} else {
			cancelCaller()
		}
		select {
		case <-dialCtx.Done():
		case <-time.After(time.Second):
			s.Fail("the initial dial must stop when either its caller or session closes")
		}
		release()
		cancelCaller()
		cancelSessionLifetime()
	}
}

func (s *ElevenLabsSuite) TestFinishedDialCancellationCannotCloseANewerAttempt() {
	firstClient, firstPeer := net.Pipe()
	defer firstPeer.Close()
	first := &dialAttempt{}
	s.Require().True(first.attach(firstClient))
	s.True(first.finish())

	secondClient, secondPeer := net.Pipe()
	defer secondClient.Close()
	defer secondPeer.Close()
	second := &dialAttempt{}
	s.Require().True(second.attach(secondClient))

	// A cancellation callback from the completed generation must be scoped to that
	// generation and leave a newer replacement transport usable.
	first.close()
	writeDone := make(chan error, 1)
	go func() {
		_, err := secondClient.Write([]byte{0x7f})
		writeDone <- err
	}()
	read := make([]byte, 1)
	_, err := secondPeer.Read(read)
	s.NoError(err)
	s.Equal(byte(0x7f), read[0])
	s.NoError(<-writeDone)
}

func (s *ElevenLabsSuite) TestCloseCancelsAStalledReplacementHandshake() {
	requestSeen := make(chan struct{}, 2)
	firstServerConn := make(chan *websocket.Conn, 1)
	releaseFirst := make(chan struct{})
	releaseSecond := make(chan struct{})
	var releaseSecondOnce sync.Once
	lateServerConn := make(chan *websocket.Conn, 1)
	secondHandlerDone := make(chan struct{}, 1)
	var requests int32
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&requests, 1)
		requestSeen <- struct{}{}
		if n == 1 {
			conn, err := upgrader.Upgrade(w, r, nil)
			if err == nil {
				firstServerConn <- conn
				<-releaseFirst
			}
			return
		}
		<-releaseSecond
		conn, err := upgrader.Upgrade(w, r, nil)
		if err == nil {
			lateServerConn <- conn
		}
		secondHandlerDone <- struct{}{}
	}))
	defer func() {
		close(releaseFirst)
		releaseSecondOnce.Do(func() { close(releaseSecond) })
		server.Close()
	}()

	provider := s.newTTS(Options{
		BaseURL:          "ws" + strings.TrimPrefix(server.URL, "http"),
		HandshakeTimeout: 15 * time.Second,
		CloseTimeout:     100 * time.Millisecond,
	})
	s.Require().NoError(provider.Start(context.Background()))
	<-requestSeen
	firstConn := <-firstServerConn
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})
	firstConn.Close()
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Disconnected)
		return ok
	})

	synthesizeDone := make(chan error, 1)
	go func() {
		synthesizeDone <- provider.Synthesize(tts.Request{ID: "during-reconnect", Text: "hello", Final: true})
	}()
	select {
	case <-requestSeen:
	case <-time.After(2 * time.Second):
		s.FailNow("the replacement handshake did not start")
	}
	closed := make(chan error, 1)
	go func() { closed <- provider.Close() }()
	select {
	case err := <-closed:
		s.NoError(err)
	case <-time.After(time.Second):
		s.FailNow("Close did not cancel the replacement handshake")
	}
	select {
	case err := <-synthesizeDone:
		s.Error(err, "the in-progress synthesis must not succeed on a late connection")
	case <-time.After(time.Second):
		s.FailNow("Synthesize did not return after Close canceled the dial")
	}
	s.Nil(provider.Client(), "Close must not leave a replacement socket installed")

	releaseSecondOnce.Do(func() { close(releaseSecond) })
	select {
	case <-secondHandlerDone:
	case <-time.After(time.Second):
		s.FailNow("the stalled server handshake did not finish after release")
	}
	select {
	case lateConn := <-lateServerConn:
		// The server can finish writing its upgrade response after the client has
		// already canceled and closed its raw transport. That must not install a
		// client-side socket or announce a replacement generation.
		_ = lateConn.Close()
	default:
	}
	s.Nil(provider.Client(), "a late server handshake must not install a client socket")
}

func (s *ElevenLabsSuite) TestStartContextDoesNotDisableSessionReconnect() {
	fake := newFakeElevenLabs()
	defer fake.close()

	startCtx, cancelStart := context.WithCancel(context.Background())
	provider := s.newTTS(Options{BaseURL: fake.baseURL(), CloseTimeout: 100 * time.Millisecond})
	s.Require().NoError(provider.Start(startCtx))
	firstConn := fake.accept()
	<-fake.url
	s.Require().NotNil(firstConn)
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})
	cancelStart()

	s.Require().NoError(firstConn.Close())
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Disconnected)
		return ok
	})

	s.NoError(provider.Synthesize(tts.Request{ID: "after-start-context", Text: "hello", Final: true}))
	secondConn := fake.accept()
	s.Require().NotNil(secondConn, "the session-owned lifetime should allow reconnect")
	messages := s.clientMessages(secondConn, 3)
	s.Equal("after-start-context", messages[0].ContextID)
	s.Equal("after-start-context", messages[1].ContextID)
	s.Equal("after-start-context", messages[2].ContextID)
	s.NoError(speak(secondConn, "after-start-context", make([]int16, 240)))
	s.NoError(finish(secondConn, "after-start-context"))
	s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "after-start-context"
	})
	s.NoError(provider.Close())
}

func (s *ElevenLabsSuite) TestCloseUnblocksAudioWhenEventBufferIsFull() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, serverConn := s.connect(fake, Options{CloseTimeout: 100 * time.Millisecond})
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	s.NoError(provider.Synthesize(tts.Request{ID: "stalled-audio", Text: "hello"}))
	s.clientMessages(serverConn, 2)
	s.collect(provider, func(event tts.Event) bool {
		started, ok := event.(tts.SynthesisStarted)
		return ok && started.SynthesisID == "stalled-audio"
	})
	for i := 0; i < cap(provider.emitter.Events()); i++ {
		provider.emitter.Send(tts.Connected{Provider: ProviderName, Model: provider.Model(), At: time.Now()})
	}
	s.NoError(speak(serverConn, "stalled-audio", make([]int16, 240)))

	provider.mu.Lock()
	current := provider.active["stalled-audio"]
	provider.mu.Unlock()
	s.Require().NotNil(current)
	blocked := false
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if !current.eventMu.TryLock() {
			blocked = true
			break
		}
		current.eventMu.Unlock()
		time.Sleep(time.Millisecond)
	}
	s.Require().True(blocked, "audio forwarding should be blocked behind the full event buffer")

	closed := make(chan error, 1)
	started := time.Now()
	go func() { closed <- provider.Close() }()
	select {
	case err := <-closed:
		s.NoError(err)
		s.Less(time.Since(started), time.Second, "Close must be bounded when the consumer stops draining")
	case <-time.After(time.Second):
		s.FailNow("Close remained blocked on an event send")
	}
	for range provider.Events() {
	}
}

func (s *ElevenLabsSuite) TestCloseUnblocksStartedWhenEventBufferIsFull() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, _ := s.connect(fake, Options{CloseTimeout: 100 * time.Millisecond})
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})
	for i := 0; i < cap(provider.emitter.Events()); i++ {
		provider.emitter.Send(tts.Connected{Provider: ProviderName, Model: provider.Model(), At: time.Now()})
	}

	synthesizeDone := make(chan error, 1)
	go func() {
		synthesizeDone <- provider.Synthesize(tts.Request{ID: "stalled-started", Text: "hello", Final: true})
	}()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		provider.mu.Lock()
		_, active := provider.active["stalled-started"]
		provider.mu.Unlock()
		if active {
			break
		}
		time.Sleep(time.Millisecond)
	}
	provider.mu.Lock()
	_, active := provider.active["stalled-started"]
	provider.mu.Unlock()
	s.Require().True(active)

	closed := make(chan error, 1)
	started := time.Now()
	go func() { closed <- provider.Close() }()
	select {
	case err := <-closed:
		s.NoError(err)
		s.Less(time.Since(started), time.Second, "Close must release a blocked Started send")
	case <-time.After(time.Second):
		s.FailNow("Close remained blocked waiting for Started")
	}
	select {
	case <-synthesizeDone:
	case <-time.After(time.Second):
		s.FailNow("Synthesize remained blocked after Close")
	}
	for range provider.Events() {
	}
}

func (s *ElevenLabsSuite) TestSatisfiesTTSInterface() {
	var _ tts.TTS = s.newTTS(Options{})
}
