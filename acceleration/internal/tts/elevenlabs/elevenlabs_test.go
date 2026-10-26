package elevenlabs

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
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

// quotaRefusals is how many socket frames of a spent quota the pool of ELEVENLABS_API_KEY
// has counted as refused.
func quotaRefusals(t *testing.T) float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != "router_keypool_refused_attempts_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["key_var"] == "ELEVENLABS_API_KEY" && labels["status"] == "quota_exceeded" {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
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
	s.ErrorIs(err, ErrRefused, "the caller's mistake, not the provider's fault")
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

// openedVoiceSettings is the voice_settings of the first frame the provider sent, as it
// went on the wire.
func (s *ElevenLabsSuite) openedVoiceSettings(options Options) string {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, options)
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))

	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	var opened struct {
		VoiceSettings json.RawMessage `json:"voice_settings"`
	}
	s.Require().NoError(json.Unmarshal(raw, &opened))
	return string(opened.VoiceSettings)
}

func (s *ElevenLabsSuite) TestAContextOpenedWithNothingAskedSendsWhatItAlwaysHas() {
	// The frame this provider sent before stability and similarity could be asked for,
	// captured on the commit before they could.
	s.JSONEq(`{"stability":0.5,"similarity_boost":0.8}`, s.openedVoiceSettings(Options{VoiceID: "v1"}))
}

func (s *ElevenLabsSuite) TestEveryContextIsOpenedWithTheStabilityAndSimilarityAskedFor() {
	stability, similarity := 0.3, 0.6

	s.JSONEq(`{"stability":0.3,"similarity_boost":0.6}`,
		s.openedVoiceSettings(Options{VoiceID: "v1", Stability: &stability, Similarity: &similarity}))
}

func (s *ElevenLabsSuite) TestAStabilityOfZeroIsSentRatherThanTakenForNone() {
	// Zero is the most variable the voice can be, not an absent setting.
	stability := 0.0

	s.JSONEq(`{"stability":0,"similarity_boost":0.8}`,
		s.openedVoiceSettings(Options{VoiceID: "v1", Stability: &stability}))
}

func (s *ElevenLabsSuite) TestAVoiceSettingOutsideZeroToOneIsRefused() {
	above, below := 1.5, -0.2

	_, err := New(Options{APIKey: "k", Stability: &above})
	s.ErrorIs(err, ErrRefused, "the caller's mistake, not the provider's fault")
	s.ErrorContains(err, "stability 1.5 is outside 0 to 1")

	_, err = New(Options{APIKey: "k", Similarity: &below})
	s.ErrorContains(err, "similarity -0.2 is outside 0 to 1")
}

func (s *ElevenLabsSuite) TestEveryContextIsOpenedWithTheStyleAskedFor() {
	style := 0.4

	s.JSONEq(`{"stability":0.5,"similarity_boost":0.8,"style":0.4}`,
		s.openedVoiceSettings(Options{VoiceID: "v1", Style: &style}))
}

func (s *ElevenLabsSuite) TestAStyleOfZeroIsSentRatherThanTakenForNone() {
	// Zero is ElevenLabs' own default and a style a caller can ask for by name.
	style := 0.0

	s.JSONEq(`{"stability":0.5,"similarity_boost":0.8,"style":0}`,
		s.openedVoiceSettings(Options{VoiceID: "v1", Style: &style}))
}

func (s *ElevenLabsSuite) TestAStyleOfOneIsSentRatherThanRefused() {
	// One is the top of the documented 0 to 1 range, so it is a style a caller can ask for.
	style := 1.0

	s.JSONEq(`{"stability":0.5,"similarity_boost":0.8,"style":1}`,
		s.openedVoiceSettings(Options{VoiceID: "v1", Style: &style}))
}

func (s *ElevenLabsSuite) TestAStyleOutsideZeroToOneIsRefused() {
	// Live on 2026-10-16 ElevenLabs refused 1.5 and -0.2 on Flash v2.5, "expected to be
	// greater or equal to 0.0 and less or equal to 1.0".
	above, below := 1.5, -0.2

	_, err := New(Options{APIKey: "k", Style: &above})
	s.ErrorIs(err, ErrStyle, "a refused style is told apart from the other refusals")
	s.ErrorIs(err, ErrRefused, "the caller's mistake, not the provider's fault")
	s.ErrorContains(err, "style 1.5 is outside 0 to 1")

	_, err = New(Options{APIKey: "k", Style: &below})
	s.ErrorContains(err, "style -0.2 is outside 0 to 1")
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

func (s *ElevenLabsSuite) TestSynthesizeDoesNotReconnectAfterStartFailed() {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := s.newTTS(Options{BaseURL: "ws" + strings.TrimPrefix(server.URL, "http")})
	defer provider.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Error(provider.Start(ctx))
	s.ErrorContains(provider.Synthesize(tts.Request{ID: "after-failed-start", Text: "hello"}), "not started")
	s.Equal(int32(1), requests.Load(), "Synthesize must not silently retry a failed initial Start")
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

func (s *ElevenLabsSuite) TestIdleSocketLossReconnectsForANewSynthesis() {
	fake := newFakeElevenLabs()
	defer fake.close()

	startCtx, cancelStart := context.WithCancel(context.Background())
	provider := s.newTTS(Options{BaseURL: fake.baseURL()})
	defer provider.Close()
	s.Require().NoError(provider.Start(startCtx))
	firstConn := fake.accept()
	s.Require().NotNil(firstConn)
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})
	cancelStart()

	firstSamples := []int16{1200, -2300, 3100, -4200}
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "first", Text: "hello", Final: true}))
	firstMessages := s.clientMessages(firstConn, 3)
	s.Equal("first", firstMessages[0].ContextID)
	s.Equal("first", firstMessages[1].ContextID)
	s.Equal("hello ", firstMessages[1].Text)
	s.True(firstMessages[2].CloseContext)
	s.Require().NoError(speak(firstConn, "first", firstSamples))
	s.Require().NoError(finish(firstConn, "first"))
	firstEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "first"
	})
	var firstChunks []tts.AudioChunk
	var firstComplete tts.SynthesisComplete
	for _, event := range firstEvents {
		switch typed := event.(type) {
		case tts.AudioChunk:
			firstChunks = append(firstChunks, typed)
		case tts.SynthesisComplete:
			firstComplete = typed
		}
	}
	s.Require().Len(firstChunks, 1)
	s.Equal(firstSamples, firstChunks[0].Audio.Samples)
	s.False(firstComplete.Interrupted)
	s.Equal(int64(len([]rune("hello"))), firstComplete.Characters)

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
	s.True(disconnected[len(disconnected)-1].(tts.Disconnected).Clean)

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "after-idle", Text: "recovered", Final: true}))
	secondConn := fake.accept()
	s.Require().NotNil(secondConn, "a later synthesis should open a replacement socket")
	<-fake.url
	secondMessages := s.clientMessages(secondConn, 3)
	s.Equal("after-idle", secondMessages[0].ContextID)
	s.Equal(" ", secondMessages[0].Text)
	s.Equal("after-idle", secondMessages[1].ContextID)
	s.Equal("recovered ", secondMessages[1].Text)
	s.True(secondMessages[2].CloseContext)

	secondSamples := []int16{500, -600, 700, -800}
	s.Require().NoError(speak(secondConn, "after-idle", secondSamples))
	s.Require().NoError(finish(secondConn, "after-idle"))
	secondEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "after-idle"
	})
	var secondChunks []tts.AudioChunk
	var secondComplete tts.SynthesisComplete
	for _, event := range secondEvents {
		switch typed := event.(type) {
		case tts.AudioChunk:
			secondChunks = append(secondChunks, typed)
		case tts.SynthesisComplete:
			secondComplete = typed
		}
	}
	s.Require().Len(secondChunks, 1)
	s.Equal(secondSamples, secondChunks[0].Audio.Samples)
	s.False(secondComplete.Interrupted)
	s.Equal(int64(len([]rune("recovered"))), secondComplete.Characters)
}

func (s *ElevenLabsSuite) TestDisconnectSettlesOnceAndDoesNotReplayAFailedID() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, firstConn := s.connect(fake, Options{})
	defer provider.Close()
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "failed", Text: "accepted"}))
	s.clientMessages(firstConn, 2)
	s.Require().NoError(firstConn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "idle with work in flight"),
		time.Now().Add(time.Second),
	))

	failedEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "failed"
	})
	var failedCompletes []tts.SynthesisComplete
	for _, event := range failedEvents {
		if complete, ok := event.(tts.SynthesisComplete); ok && complete.SynthesisID == "failed" {
			failedCompletes = append(failedCompletes, complete)
		}
	}
	s.Require().Len(failedCompletes, 1, "a lost socket must account for an active ID once")
	s.True(failedCompletes[0].Interrupted)

	// A delayed delta for the failed ID is not new work and must not be replayed.
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "failed", Text: "late", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "new", Text: "fresh", Final: true}))
	secondConn := fake.accept()
	s.Require().NotNil(secondConn)
	<-fake.url
	messages := s.clientMessages(secondConn, 3)
	s.Equal("new", messages[0].ContextID)
	s.Equal(" ", messages[0].Text)
	s.Equal("new", messages[1].ContextID)
	s.Equal("fresh ", messages[1].Text)
	s.True(messages[2].CloseContext)

	samples := []int16{900, -1000, 1100, -1200}
	s.Require().NoError(speak(secondConn, "new", samples))
	s.Require().NoError(finish(secondConn, "new"))
	newEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "new"
	})
	var chunks []tts.AudioChunk
	var newComplete tts.SynthesisComplete
	var newCompletions int
	for _, event := range newEvents {
		switch typed := event.(type) {
		case tts.SynthesisStarted:
			s.NotEqual("failed", typed.SynthesisID, "the failed ID must not start again")
		case tts.AudioChunk:
			s.NotEqual("failed", typed.SynthesisID, "the failed ID must not produce more audio")
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			if typed.SynthesisID == "failed" {
				s.Fail("the failed ID must not be completed more than once")
			}
			if typed.SynthesisID == "new" {
				newComplete = typed
				newCompletions++
			}
		case tts.Error:
			s.NotEqual("failed", typed.SynthesisID, "late failures for the old ID must stay silent")
		}
	}
	s.Require().Len(chunks, 1)
	s.Equal(1, newCompletions, "the fresh ID should settle exactly once")
	s.Equal(samples, chunks[0].Audio.Samples)
	s.False(newComplete.Interrupted)
}

func (s *ElevenLabsSuite) TestInterruptDoesNotReopenAnOldID() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "interrupted", Text: "first piece"}))
	s.clientMessages(conn, 2)
	s.Require().NoError(provider.Interrupt())
	interrupt := s.clientMessages(conn, 1)
	s.Equal("interrupted", interrupt[0].ContextID)
	s.True(interrupt[0].CloseContext)
	s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "interrupted"
	})

	// Late text under an interrupted ID cannot start another context.
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "interrupted", Text: "late piece", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "next", Text: "new turn", Final: true}))
	messages := s.clientMessages(conn, 3)
	s.Equal("next", messages[0].ContextID)
	s.Equal(" ", messages[0].Text)
	s.Equal("next", messages[1].ContextID)
	s.Equal("new turn ", messages[1].Text)
	s.True(messages[2].CloseContext)
}

func (s *ElevenLabsSuite) TestAWriteFailureAnnouncesAndSettlesBeforeReconnect() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, _ := s.connect(fake, Options{})
	defer provider.Close()
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	s.Require().NoError(provider.Client().SetWriteDeadline(time.Now().Add(-time.Second)))
	err := provider.Synthesize(tts.Request{ID: "write-failed", Text: "accepted", Final: true})
	s.Error(err, "the expired client write deadline should fail the first context frame")

	failedEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "write-failed"
	})
	var startedAt, completeAt int
	var completions int
	for index, event := range failedEvents {
		switch typed := event.(type) {
		case tts.SynthesisStarted:
			if typed.SynthesisID == "write-failed" {
				startedAt = index + 1
			}
		case tts.SynthesisComplete:
			if typed.SynthesisID == "write-failed" {
				completeAt = index + 1
				completions++
				s.True(typed.Interrupted)
			}
		}
	}
	s.Require().Positive(startedAt, "a locally accepted attempt is announced before any terminal event")
	s.Require().Positive(completeAt)
	s.Less(startedAt, completeAt)
	s.Equal(1, completions)

	// The failed ID is terminal; only a different new ID may use the replacement socket.
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "write-failed", Text: "late", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "fresh", Text: "new", Final: true}))
	secondConn := fake.accept()
	s.Require().NotNil(secondConn)
	<-fake.url
	messages := s.clientMessages(secondConn, 3)
	s.Equal("fresh", messages[0].ContextID)
	s.Equal(" ", messages[0].Text)
	s.Equal("fresh", messages[1].ContextID)
	s.Equal("new ", messages[1].Text)
	s.True(messages[2].CloseContext)

	samples := []int16{120, -340, 560, -780}
	s.Require().NoError(speak(secondConn, "fresh", samples))
	s.Require().NoError(finish(secondConn, "fresh"))
	freshEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == "fresh"
	})
	var freshChunks []tts.AudioChunk
	for _, event := range freshEvents {
		if chunk, ok := event.(tts.AudioChunk); ok {
			freshChunks = append(freshChunks, chunk)
		}
	}
	s.Require().Len(freshChunks, 1)
	s.Equal(samples, freshChunks[0].Audio.Samples)
}

func (s *ElevenLabsSuite) TestAStaleSocketFinalCannotSettleAReusedID() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, firstConn := s.connect(fake, Options{})
	defer provider.Close()
	<-fake.url
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Connected)
		return ok
	})

	const id = "reused"
	s.Require().NoError(provider.Synthesize(tts.Request{ID: id, Text: "first", Final: true}))
	s.clientMessages(firstConn, 3)
	s.Require().NoError(speak(firstConn, id, []int16{100, -200, 300, -400}))
	s.Require().NoError(finish(firstConn, id))
	s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == id
	})

	provider.mu.Lock()
	oldSocket := provider.conn
	provider.mu.Unlock()
	s.Require().NoError(firstConn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "replace"),
		time.Now().Add(time.Second),
	))
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.Disconnected)
		return ok
	})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: id, Text: "second", Final: true}))
	secondConn := fake.accept()
	s.Require().NotNil(secondConn)
	<-fake.url
	started := s.collect(provider, func(event tts.Event) bool {
		started, ok := event.(tts.SynthesisStarted)
		return ok && started.SynthesisID == id
	})
	s.NotEmpty(started)
	s.clientMessages(secondConn, 3)

	provider.handleMessage(oldSocket, serverMessage{ContextID: id, IsFinal: true})
	select {
	case event := <-provider.Events():
		s.Failf("stale socket event", "old generation must not affect the replacement synthesis: %T", event)
	case <-time.After(100 * time.Millisecond):
	}

	samples := []int16{500, -600, 700, -800}
	s.Require().NoError(speak(secondConn, id, samples))
	s.Require().NoError(finish(secondConn, id))
	newEvents := s.collect(provider, func(event tts.Event) bool {
		complete, ok := event.(tts.SynthesisComplete)
		return ok && complete.SynthesisID == id
	})
	var chunks []tts.AudioChunk
	var complete tts.SynthesisComplete
	for _, event := range newEvents {
		switch typed := event.(type) {
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			complete = typed
		}
	}
	s.Require().Len(chunks, 1)
	s.Equal(samples, chunks[0].Audio.Samples)
	s.False(complete.Interrupted)
}

func (s *ElevenLabsSuite) TestCloseCancelsAStalledReplacementHandshake() {
	requestSeen := make(chan struct{}, 2)
	firstServerConn := make(chan *websocket.Conn, 1)
	releaseFirst := make(chan struct{})
	releaseSecond := make(chan struct{})
	var releaseSecondOnce sync.Once
	secondHandlerDone := make(chan struct{}, 1)
	var requests atomic.Int32
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestSeen <- struct{}{}
		if requests.Add(1) == 1 {
			conn, err := upgrader.Upgrade(w, r, nil)
			if err == nil {
				firstServerConn <- conn
				<-releaseFirst
			}
			return
		}
		<-releaseSecond
		_, _ = upgrader.Upgrade(w, r, nil)
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
	defer provider.Close()
	s.Require().NoError(provider.Start(context.Background()))
	<-requestSeen
	firstConn := <-firstServerConn
	s.Require().NoError(firstConn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "idle"),
		time.Now().Add(time.Second),
	))
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
	case err := <-synthesizeDone:
		s.FailNow("new synthesis failed before a replacement handshake started: %v", err)
	case <-time.After(2 * time.Second):
		s.FailNow("new synthesis did not start a replacement handshake")
	}
	interruptDone := make(chan error, 1)
	go func() { interruptDone <- provider.Interrupt() }()
	select {
	case err := <-interruptDone:
		s.NoError(err, "an interrupt with no active utterance should not wait for reconnect")
	case <-time.After(time.Second):
		s.FailNow("an interrupt with no active utterance waited for the stalled handshake")
	}

	closed := make(chan error, 1)
	go func() { closed <- provider.Close() }()
	select {
	case err := <-closed:
		s.NoError(err)
	case <-time.After(time.Second):
		s.FailNow("Close did not cancel the stalled replacement handshake")
	}
	select {
	case err := <-synthesizeDone:
		s.Error(err, "a synthesis blocked in a canceled dial must not succeed")
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
	s.Nil(provider.Client(), "a late handshake must not install a socket after Close")
}

func (s *ElevenLabsSuite) TestSatisfiesTTSInterface() {
	var _ tts.TTS = s.newTTS(Options{})
}

func (s *ElevenLabsSuite) TestAKeyWithNoQuotaLeftIsCountedAsRefused() {
	spent, spare := "elevenlabs-"+rand.Text(), "elevenlabs-"+rand.Text()
	fake := newFakeElevenLabs()
	defer fake.close()
	before := quotaRefusals(s.T())
	provider, conn := s.connect(fake, Options{APIKey: spent + "," + spare})
	defer provider.Close()

	s.Require().NoError(conn.WriteMessage(websocket.TextMessage,
		[]byte(`{"message":"This request exceeds your API key quota.","error":"quota_exceeded","code":1008}`)))
	s.collect(provider, func(event tts.Event) bool {
		_, failed := event.(tts.Error)
		return failed
	})

	s.Equal(before+1, quotaRefusals(s.T()))
}
