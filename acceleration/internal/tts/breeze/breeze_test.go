package breeze

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// fakeDeployment speaks the protocol in acceleration/deploy/breeze-tts-2, so the provider
// can be driven over a real connection without a GPU.
type fakeDeployment struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	done   chan struct{}
	// sampleRate is what the handshake claims to generate at.
	sampleRate int
	// rejectHandshake makes the server answer the metadata frame with an error.
	rejectHandshake bool
}

func newFakeDeployment() *fakeDeployment {
	fake := &fakeDeployment{
		conns:      make(chan *websocket.Conn, 1),
		done:       make(chan struct{}),
		sampleRate: DefaultSampleRate,
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}

		// The opening metadata frame is answered before the test takes over.
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		reply := map[string]any{"type": "ready", "sample_rate": fake.sampleRate}
		if fake.rejectHandshake {
			reply = map[string]any{"type": "error", "error": "sample_rate must be 24000"}
		}
		payload, _ := json.Marshal(reply)
		if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
			return
		}

		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakeDeployment) baseURL() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeDeployment) close() {
	close(f.done)
	f.server.Close()
}

func (f *fakeDeployment) accept() *websocket.Conn {
	select {
	case conn := <-f.conns:
		return conn
	case <-time.After(5 * time.Second):
		return nil
	}
}

// speak sends a frame of PCM audio, the way the deployment does.
func speak(conn *websocket.Conn, samples []int16) error {
	pcm := audio.PcmData{Samples: samples, SampleRate: DefaultSampleRate, Channels: 1}
	return conn.WriteMessage(websocket.BinaryMessage, pcm.Bytes())
}

func finish(conn *websocket.Conn, id string, cancelled bool) error {
	payload, err := json.Marshal(map[string]any{
		"type": "final", "id": id, "cancelled": cancelled,
	})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

type BreezeSuite struct {
	suite.Suite
}

func TestBreezeSuite(t *testing.T) {
	suite.Run(t, new(BreezeSuite))
}

// newTTS returns a provider that is wired up but never connected.
func (s *BreezeSuite) newTTS(options Options) *TTS {
	if options.URL == "" {
		options.URL = "wss://example.invalid/websocket"
	}
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// connect returns a started provider and the server side of its connection.
func (s *BreezeSuite) connect(fake *fakeDeployment, options Options) (*TTS, *websocket.Conn) {
	options.URL = fake.baseURL() + "/websocket"
	// The fake only finals when a test tells it to, and no test is about that wait.
	if options.DrainTimeout == 0 {
		options.DrainTimeout = 100 * time.Millisecond
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
func (s *BreezeSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
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

// clientFrames reads the frames the provider sent.
func (s *BreezeSuite) clientFrames(conn *websocket.Conn, want int) []controlFrame {
	var frames []controlFrame
	for len(frames) < want {
		s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
		_, raw, err := conn.ReadMessage()
		s.Require().NoError(err)

		var frame controlFrame
		s.Require().NoError(json.Unmarshal(raw, &frame))
		frames = append(frames, frame)
	}
	return frames
}

// settled reads to the end of one utterance and returns how it finished.
func (s *BreezeSuite) settled(provider *TTS) tts.SynthesisComplete {
	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete, ok := events[len(events)-1].(tts.SynthesisComplete)
	s.Require().True(ok, "the last event should settle the utterance")
	return complete
}

func (s *BreezeSuite) TestNewRequiresAURL() {
	s.T().Setenv("BREEZE_WS_URL", "")

	_, err := New(Options{APIKey: "k"})
	s.ErrorContains(err, "websocket url is required")
}

func (s *BreezeSuite) TestNewRejectsANonWebsocketURL() {
	_, err := New(Options{URL: "https://example.com", APIKey: "k"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *BreezeSuite) TestNewRequiresAnAPIKey() {
	s.T().Setenv("BASETEN_API_KEY", "")

	_, err := New(Options{URL: "wss://example.com"})
	s.ErrorContains(err, "api key is required")
}

func (s *BreezeSuite) TestNewFallsBackToEnv() {
	s.T().Setenv("BREEZE_WS_URL", "wss://from-env/websocket")
	s.T().Setenv("BASETEN_API_KEY", "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("wss://from-env/websocket", provider.options.URL)
	s.Equal("key-from-env", provider.options.APIKey)
}

func (s *BreezeSuite) TestNewRejectsAReferenceTranscriptWithoutItsAudio() {
	_, err := New(Options{URL: "wss://x/y", APIKey: "k", ReferenceText: "hello"})
	s.ErrorContains(err, "reference text needs reference audio")
}

func (s *BreezeSuite) TestTheDefaultsAreWhatTheDeploymentServes() {
	provider := s.newTTS(Options{})
	s.Equal(DefaultModel, provider.Model())
	s.Equal(ProviderName, provider.Provider())
	s.Equal(24_000, provider.SampleRate())
	s.True(provider.Streaming(), "the deployment accumulates deltas and generates on flush")
}

func (s *BreezeSuite) TestThisVoiceActsADirectionRatherThanReadingItOut() {
	provider := s.newTTS(Options{})

	s.True(provider.Performs())
	s.Contains(provider.Prompt(), "[sigh]",
		"the model should be told the syntax, in the brackets the agent writes")
	s.Contains(provider.Prompt(), "those four",
		"unlike an open tag set, this model performs a fixed list")
}

func (s *BreezeSuite) TestARejectedHandshakeFailsTheStart() {
	fake := newFakeDeployment()
	defer fake.close()
	fake.rejectHandshake = true

	provider := s.newTTS(Options{URL: fake.baseURL() + "/websocket"})
	err := provider.Start(context.Background())
	s.ErrorContains(err, "handshake rejected")
}

func (s *BreezeSuite) TestASampleRateMismatchFailsTheStartRatherThanPlayingBackWrong() {
	fake := newFakeDeployment()
	defer fake.close()
	fake.sampleRate = 44_100

	provider := s.newTTS(Options{URL: fake.baseURL() + "/websocket"})
	err := provider.Start(context.Background())
	s.ErrorContains(err, "deployment generates at 44100 Hz, session wants 24000")
}

func (s *BreezeSuite) TestTheVoiceDescriptionTravelsAsAnInstruction() {
	// A voice here is prose rather than an id in a library, and the deployment passes it
	// to the model as the instruction that designs the speaker.
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{Voice: "a warm, unhurried narrator"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))

	frames := s.clientFrames(conn, 2)
	s.Equal(controlFlush, frames[1].Type)
	s.Equal("a warm, unhurried narrator", frames[1].Instruction)
}

func (s *BreezeSuite) TestARequestCanAskForADifferentVoiceThanTheSession() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{Voice: "a warm narrator"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{
		ID: "u1", Text: "hello", Voice: "a brisk newsreader", Final: true,
	}))

	frames := s.clientFrames(conn, 2)
	s.Equal("a brisk newsreader", frames[1].Instruction,
		"the voice is chosen per flush, so a request may override the session")
}

func (s *BreezeSuite) TestWithoutADescriptionTheModelUsesItsOwnVoice() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))

	frames := s.clientFrames(conn, 2)
	s.Empty(frames[1].Instruction, "an empty instruction must not be sent as one")
}

func (s *BreezeSuite) TestACloneTravelsWithTheTranscriptThatExplainsIt() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{
		ReferenceAudio: "/voices/alice.wav",
		ReferenceText:  "the exact words in the clip",
		Voice:          "sound tired",
	})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))

	flush := s.clientFrames(conn, 2)[1]
	s.Equal("/voices/alice.wav", flush.ReferenceAudio)
	s.Equal("the exact words in the clip", flush.ReferenceText)
	s.Equal("sound tired", flush.Instruction,
		"with a clip the reference decides who speaks and the description how")
}

func (s *BreezeSuite) TestDeltasAreSentAsTheyArriveAndFlushedAtTheEnd() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello "}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "there", Final: true}))

	frames := s.clientFrames(conn, 3)
	s.Equal(controlText, frames[0].Type)
	s.Equal("hello ", frames[0].Text, "a delta should go upstream immediately")
	s.Equal("u1", frames[0].ID)
	s.Equal("there", frames[1].Text)
	s.Equal(controlFlush, frames[2].Type)
	s.Equal("u1", frames[2].ID)
}

func (s *BreezeSuite) TestADirectionIsLeftForTheDeploymentToTranslate() {
	// Breeze writes an English vocal event in parentheses, but a direction can be split
	// across two deltas, so the rewrite happens where the utterance is whole again.
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "[si"}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "gh] fine", Final: true}))

	frames := s.clientFrames(conn, 3)
	s.Equal("[si", frames[0].Text)
	s.Equal("gh] fine", frames[1].Text, "half a direction is not something this end can act on")
}

func (s *BreezeSuite) TestAudioAndTheFinalFrameSettleTheSynthesis() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello there", Final: true}))

	// A tenth of a second of speech, twice.
	s.Require().NoError(speak(conn, make([]int16, DefaultSampleRate/10)))
	s.Require().NoError(speak(conn, make([]int16, DefaultSampleRate/10)))
	s.Require().NoError(finish(conn, "u1", false))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var chunks []tts.AudioChunk
	var complete tts.SynthesisComplete
	for _, event := range events {
		switch typed := event.(type) {
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			complete = typed
		}
	}

	s.Require().Len(chunks, 2)
	s.Equal(0, chunks[0].Index)
	s.Equal(1, chunks[1].Index, "chunks should be numbered so playback order survives")
	s.Equal(DefaultSampleRate, chunks[0].Audio.SampleRate)
	s.Equal("u1", chunks[0].SynthesisID)

	s.Equal("u1", complete.SynthesisID)
	s.EqualValues(len("hello there"), complete.Characters)
	s.InDelta(200.0, complete.AudioDurationMs, 1.0)
	s.False(complete.Interrupted)
}

func (s *BreezeSuite) TestInterruptCancelsEveryUtteranceInFlight() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "one", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u2", Text: "two", Final: true}))
	s.clientFrames(conn, 4)

	s.Require().NoError(provider.Interrupt())

	cancels := s.clientFrames(conn, 2)
	s.Equal(controlCancel, cancels[0].Type)
	s.ElementsMatch([]string{"u1", "u2"}, []string{cancels[0].ID, cancels[1].ID})

	s.Require().NoError(finish(conn, "u1", true))
	s.True(s.settled(provider).Interrupted, "barge-in should be visible in the stat row")
}

func (s *BreezeSuite) TestAServerErrorForOneUtteranceIsNotFatal() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage,
		[]byte(`{"type":"error","id":"u1","error":"generation failed"}`)))

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
	s.Require().True(sawFailure)
	s.ErrorContains(failure.Err, "generation failed")
	s.False(failure.Fatal, "one bad utterance should not close the connection")
}

func (s *BreezeSuite) TestCloseSettlesAnUtteranceTheDeploymentNeverFinished() {
	fake := newFakeDeployment()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.Require().NoError(speak(conn, make([]int16, DefaultSampleRate/10)))

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
	s.InDelta(100.0, completes[0].AudioDurationMs, 1.0, "the audio that did play should be billed")
}

func (s *BreezeSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newTTS(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	for range provider.Events() {
	}
}

func (s *BreezeSuite) TestSynthesizeFailsBeforeStart() {
	err := s.newTTS(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *BreezeSuite) TestSynthesizeFailsAfterClose() {
	provider := s.newTTS(Options{})
	s.Require().NoError(provider.Close())

	err := provider.Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "session closed")
}

func (s *BreezeSuite) TestSatisfiesTTSInterface() {
	var _ tts.TTS = s.newTTS(Options{})
}
