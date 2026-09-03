package inworld

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

// fakeInworld is a WebSocket server that speaks the context protocol, so the provider can
// be driven over a real connection without an API key.
type fakeInworld struct {
	server  *httptest.Server
	conns   chan *websocket.Conn
	done    chan struct{}
	request chan *http.Request
}

func newFakeInworld() *fakeInworld {
	fake := &fakeInworld{
		conns:   make(chan *websocket.Conn, 4),
		done:    make(chan struct{}),
		request: make(chan *http.Request, 4),
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

func (f *fakeInworld) baseURL() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeInworld) close() {
	close(f.done)
	f.server.Close()
}

func (f *fakeInworld) accept() *websocket.Conn {
	select {
	case conn := <-f.conns:
		return conn
	case <-time.After(5 * time.Second):
		return nil
	}
}

func speak(conn *websocket.Conn, contextID string, samples []int16) error {
	pcm := audio.PcmData{Samples: samples, SampleRate: DefaultSampleRate, Channels: 1}
	payload, err := json.Marshal(map[string]any{
		"result": map[string]any{
			"contextId":  contextID,
			"audioChunk": map[string]any{"audioContent": base64.StdEncoding.EncodeToString(pcm.Bytes())},
		},
	})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

func flushed(conn *websocket.Conn, contextID string) error {
	payload, err := json.Marshal(map[string]any{
		"result": map[string]any{
			"contextId":      contextID,
			"flushCompleted": map[string]any{},
		},
	})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

// closed is the server acknowledging a close_context, which is the last thing it sends for
// a context.
func closed(conn *websocket.Conn, contextID string) error {
	payload, err := json.Marshal(map[string]any{
		"result": map[string]any{
			"contextId":     contextID,
			"contextClosed": map[string]any{},
		},
	})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

type InworldSocketSuite struct {
	suite.Suite
}

func TestInworldSocketSuite(t *testing.T) {
	suite.Run(t, new(InworldSocketSuite))
}

func (s *InworldSocketSuite) newTTS(options Options) *TTS {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

func (s *InworldSocketSuite) connect(fake *fakeInworld, options Options) (*TTS, *websocket.Conn) {
	options.BaseURL = fake.baseURL()
	provider := s.newTTS(options)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))

	conn := fake.accept()
	s.Require().NotNil(conn, "the provider should have connected")
	return provider, conn
}

func (s *InworldSocketSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
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

func (s *InworldSocketSuite) frames(conn *websocket.Conn, want int) []clientFrame {
	var messages []clientFrame
	for len(messages) < want {
		s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
		_, raw, err := conn.ReadMessage()
		s.Require().NoError(err)

		var message clientFrame
		s.Require().NoError(json.Unmarshal(raw, &message))
		messages = append(messages, message)
	}
	return messages
}

func (s *InworldSocketSuite) TestTheAPIKeyTravelsAsBasicAuth() {
	fake := newFakeInworld()
	defer fake.close()
	provider, _ := s.connect(fake, Options{APIKey: "secret-key"})
	defer provider.Close()

	request := <-fake.request
	s.Equal("Basic secret-key", request.Header.Get("Authorization"))
	s.NotEmpty(request.Header.Get("X-Request-Id"))
	s.Equal(streamPath, request.URL.Path)
}

func (s *InworldSocketSuite) TestAnUtteranceCreatesThenStreamsThenFlushes() {
	fake := newFakeInworld()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "Clive", SampleRate: 16_000})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello"}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: " world"}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Final: true}))

	messages := s.frames(conn, 4)

	s.Require().NotNil(messages[0].Create)
	s.Equal("u1", messages[0].ContextID)
	s.Equal("Clive", messages[0].Create.VoiceID)
	s.Equal(DefaultModel, messages[0].Create.ModelID)
	s.Equal("PCM", messages[0].Create.AudioConfig.AudioEncoding)
	s.Equal(16_000, messages[0].Create.AudioConfig.SampleRateHertz)

	s.Require().NotNil(messages[1].SendText)
	s.Equal("hello", messages[1].SendText.Text)
	s.Equal("u1", messages[1].ContextID)

	s.Require().NotNil(messages[2].SendText)
	s.Equal(" world", messages[2].SendText.Text)

	s.Require().NotNil(messages[3].FlushContext)
	s.Equal("u1", messages[3].ContextID)
}

func (s *InworldSocketSuite) TestAudioAndFlushSettleTheSynthesis() {
	fake := newFakeInworld()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello there", Final: true}))
	s.frames(conn, 3)

	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.Require().NoError(flushed(conn, "u1"))

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

	s.Equal("u1", started.SynthesisID)
	s.Equal(DefaultVoiceID, started.Voice)
	s.Require().Len(chunks, 2)
	s.Equal(0, chunks[0].Index)
	s.Equal(1, chunks[1].Index)

	s.Equal("u1", complete.SynthesisID)
	s.EqualValues(len("hello there"), complete.Characters)
	s.InDelta(200.0, complete.AudioDurationMs, 1.0)
	s.Positive(complete.TimeToFirstByteMs)
	s.False(complete.Interrupted)
}

func (s *InworldSocketSuite) TestTheTailAfterAFlushIsStillSpoken() {
	// A flush says the text is in, not that the audio for it has all been sent. The rest
	// arrives while the context is being closed, and it is the end of a sentence the caller
	// is waiting to hear rather than something to drop as somebody else's.
	fake := newFakeInworld()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello there", Final: true}))
	s.frames(conn, 3)
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.Require().NoError(flushed(conn, "u1"))
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.Require().NoError(closed(conn, "u1"))

	// The tail arrives after the completion, so the completion cannot be what ends this.
	events := s.collect(provider, func(event tts.Event) bool {
		chunk, ok := event.(tts.AudioChunk)
		return ok && chunk.Index == 1
	})

	var chunks []tts.AudioChunk
	var completions int
	for _, event := range events {
		switch typed := event.(type) {
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			completions++
		}
	}
	s.Len(chunks, 2, "the tail of the utterance was dropped rather than spoken")
	s.Equal(1, completions, "the utterance was settled more than once")
}

func (s *InworldSocketSuite) TestAudioForAClosedContextIsDropped() {
	// Once the server says the context is closed, whatever keeps draining onto the shared
	// socket belongs to nothing, and speaking it would cut into whatever came next.
	fake := newFakeInworld()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.frames(conn, 3)
	s.Require().NoError(flushed(conn, "u1"))
	s.Require().NoError(closed(conn, "u1"))
	s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u2", Text: "next", Final: true}))
	s.frames(conn, 3)
	s.Require().NoError(speak(conn, "u2", make([]int16, 2400)))

	events := s.collect(provider, func(event tts.Event) bool {
		chunk, ok := event.(tts.AudioChunk)
		return ok && chunk.SynthesisID == "u2"
	})
	for _, event := range events {
		if chunk, ok := event.(tts.AudioChunk); ok {
			s.NotEqual("u1", chunk.SynthesisID,
				"audio for a closed context was spoken over the next utterance")
		}
	}
}

func (s *InworldSocketSuite) TestFlushClosesTheContextOnce() {
	fake := newFakeInworld()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello", Final: true}))
	s.frames(conn, 3)
	s.Require().NoError(flushed(conn, "u1"))
	s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	closed := s.frames(conn, 1)
	s.Require().NotNil(closed[0].CloseContext)
	s.Equal("u1", closed[0].ContextID)
}

func (s *InworldSocketSuite) TestInterruptClosesTheContextAndDropsAudioStillOnTheWire() {
	fake := newFakeInworld()
	defer fake.close()
	provider, conn := s.connect(fake, Options{})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "a long answer", Final: true}))
	s.frames(conn, 3)
	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})

	s.Require().NoError(provider.Interrupt())

	s.Require().NoError(speak(conn, "u1", make([]int16, 2400)))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete := events[len(events)-1].(tts.SynthesisComplete)

	s.True(complete.Interrupted)
	s.InDelta(100.0, complete.AudioDurationMs, 1.0,
		"only the audio heard before the interruption counts")

	var closed clientFrame
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	s.Require().NoError(json.Unmarshal(raw, &closed))
	s.Equal("u1", closed.ContextID)
	s.NotNil(closed.CloseContext)
}

func (s *InworldSocketSuite) TestAnUtteranceCannotChangeVoiceHalfway() {
	fake := newFakeInworld()
	defer fake.close()
	provider, _ := s.connect(fake, Options{VoiceID: "Sarah"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello"}))

	err := provider.Synthesize(tts.Request{ID: "u1", Text: " world", Voice: "Clive"})
	s.ErrorContains(err, "being said in voice Sarah, not Clive")
}

func (s *InworldSocketSuite) TestAnUtteranceCanPickItsOwnVoice() {
	fake := newFakeInworld()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "Sarah"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{
		ID: "u1", Text: "hello", Voice: "Clive", Final: true,
	}))

	s.Equal("Clive", s.frames(conn, 1)[0].Create.VoiceID)
}

func (s *InworldSocketSuite) TestALostConnectionSettlesWhatWasInFlight() {
	fake := newFakeInworld()
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
