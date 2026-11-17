package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// posted is one request the fake received.
type posted struct {
	path   string
	query  url.Values
	header http.Header
	body   generateRequest
}

// fakeGemini answers streamGenerateContent the way Google does, as server-sent events of
// base64 PCM, so the provider can be driven over a real request without an API key.
type fakeGemini struct {
	server *httptest.Server

	mu       sync.Mutex
	requests []posted

	// events is how many audio events to answer with, each samplesPerEvent long.
	events          int
	samplesPerEvent int
	// status, when set, is answered instead of audio.
	status int
	body   string
	// failAfter, when positive, sends an error event after that many audio events.
	failAfter int
	// pause is held between events, so a test can interrupt mid-response.
	pause time.Duration
}

func newFakeGemini() *fakeGemini {
	fake := &fakeGemini{events: 2, samplesPerEvent: 2400}

	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			return
		}
		var body generateRequest
		_ = json.Unmarshal(raw, &body)

		fake.mu.Lock()
		fake.requests = append(fake.requests, posted{
			path: r.URL.Path, query: r.URL.Query(), header: r.Header.Clone(), body: body,
		})
		status, failure := fake.status, fake.body
		events, samples, failAfter, pause := fake.events, fake.samplesPerEvent, fake.failAfter, fake.pause
		fake.mu.Unlock()

		if status != 0 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(failure))
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		send := func(event any) bool {
			payload, _ := json.Marshal(event)
			if _, err := fmt.Fprintf(w, "data: %s\r\n\r\n", payload); err != nil {
				return false
			}
			if flusher != nil {
				flusher.Flush()
			}
			return true
		}

		for sent := 0; sent < events; sent++ {
			if failAfter > 0 && sent == failAfter {
				send(map[string]any{"error": map[string]any{
					"code": 500, "message": "Internal error encountered.", "status": "INTERNAL",
				}})
				return
			}
			if !send(eventOf(part{InlineData: &blob{
				Data:     base64.StdEncoding.EncodeToString(silence(samples)),
				MimeType: "audio/l16; rate=24000; channels=1",
			}})) {
				return
			}
			if pause > 0 {
				select {
				case <-time.After(pause):
				case <-r.Context().Done():
					return
				}
			}
		}
		send(streamEvent{Candidates: []candidate{{
			Content: &content{Role: "model", Parts: []part{{Text: ""}}}, FinishReason: "STOP",
		}}})
	}))
	return fake
}

func (f *fakeGemini) posted() []posted {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]posted(nil), f.requests...)
}

type GeminiSocketSuite struct {
	suite.Suite
}

func TestGeminiSocketSuite(t *testing.T) {
	suite.Run(t, new(GeminiSocketSuite))
}

// start returns a started provider pointed at the fake.
func (s *GeminiSocketSuite) start(fake *fakeGemini, options Options) *TTS {
	options.BaseURL = fake.server.URL
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { _ = provider.Close() })
	return provider
}

// collect reads events until the predicate is satisfied or the wait runs out.
func (s *GeminiSocketSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
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

func (s *GeminiSocketSuite) settled(provider *TTS) []tts.Event {
	return s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
}

func (s *GeminiSocketSuite) TestTheKeyTravelsInAHeaderAndTheModelInThePath() {
	fake := newFakeGemini()
	defer fake.server.Close()
	provider := s.start(fake, Options{APIKey: "secret-key"})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Final: true}))
	s.settled(provider)

	requests := fake.posted()
	s.Require().Len(requests, 1)
	s.Equal("/models/gemini-3.8-flash-tts:streamGenerateContent", requests[0].path)
	s.Equal("sse", requests[0].query.Get("alt"))
	s.Empty(requests[0].query.Get("key"), "the key belongs in a header, not in a URL that gets logged")
	s.Equal("secret-key", requests[0].header.Get("x-goog-api-key"))
	s.Equal("application/json", requests[0].header.Get("Content-Type"))
}

func (s *GeminiSocketSuite) TestOneUtteranceIsOnePostAndStreamsItsAudioBack() {
	fake := newFakeGemini()
	defer fake.server.Close()
	provider := s.start(fake, Options{Voice: "Kore"})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello there", Final: true}))
	events := s.settled(provider)

	var started tts.SynthesisStarted
	var chunks []tts.AudioChunk
	for _, event := range events {
		switch typed := event.(type) {
		case tts.SynthesisStarted:
			started = typed
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		}
	}
	complete := events[len(events)-1].(tts.SynthesisComplete)

	s.Equal("u1", started.SynthesisID)
	s.Equal("Kore", started.Voice)
	s.Len(chunks, 2, "each event is played as it arrives rather than collected into one")
	s.Equal("u1", complete.SynthesisID)
	s.EqualValues(len("hello there"), complete.Characters)
	s.InDelta(200.0, complete.AudioDurationMs, 1.0)
	s.Positive(complete.TimeToFirstByteMs)
	s.False(complete.Interrupted)

	requests := fake.posted()
	s.Require().Len(requests, 1, "a whole utterance is a single request")
	s.Equal("hello there", requests[0].body.Contents[0].Parts[0].Text)
	s.Equal("Kore", requests[0].body.GenerationConfig.SpeechConfig.VoiceConfig.Voice)
	s.Equal("AUDIO_L16", requests[0].body.GenerationConfig.ResponseFormat.Audio.MimeType)
}

func (s *GeminiSocketSuite) TestDeltasAreBufferedIntoOneRequest() {
	fake := newFakeGemini()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello "}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "there"}))
	s.Empty(fake.posted(), "a partial utterance should not be sent")

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Final: true}))
	events := s.settled(provider)

	s.Equal("u1", events[len(events)-1].(tts.SynthesisComplete).SynthesisID)
	requests := fake.posted()
	s.Require().Len(requests, 1)
	s.Equal("hello there", requests[0].body.Contents[0].Parts[0].Text)
}

func (s *GeminiSocketSuite) TestAPartialWithoutAnIdIsRejected() {
	fake := newFakeGemini()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	s.ErrorContains(provider.Synthesize(tts.Request{Text: "hello"}), "a partial request needs an id")
}

func (s *GeminiSocketSuite) TestAnEmptyUtteranceIsRejected() {
	fake := newFakeGemini()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	s.ErrorContains(provider.Synthesize(tts.Request{Text: "   ", Final: true}), "nothing to say")
	s.Empty(fake.posted())
}

func (s *GeminiSocketSuite) TestARejectedRequestIsReportedAndStillSettles() {
	fake := newFakeGemini()
	defer fake.server.Close()
	fake.status, fake.body = http.StatusBadRequest, `{"error":{"message":"Request contains an invalid argument."}}`
	provider := s.start(fake, Options{Voice: "NoSuchVoice"})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Final: true}))
	events := s.settled(provider)

	var failure tts.Error
	for _, event := range events {
		if typed, ok := event.(tts.Error); ok {
			failure = typed
		}
	}
	s.Require().Error(failure.Err, "a rejected request should reach the caller")
	s.ErrorContains(failure.Err, "http 400")
	s.ErrorContains(failure.Err, "invalid argument")

	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted, "a failed synthesis produced no speech")
	s.Zero(complete.AudioDurationMs)
}

func (s *GeminiSocketSuite) TestAnErrorPartWayThroughIsReportedAndSettlesWhatWasSaid() {
	fake := newFakeGemini()
	defer fake.server.Close()
	fake.events, fake.failAfter = 4, 1
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Final: true}))
	events := s.settled(provider)

	var failure tts.Error
	for _, event := range events {
		if typed, ok := event.(tts.Error); ok {
			failure = typed
		}
	}
	s.ErrorContains(failure.Err, "INTERNAL: Internal error encountered.")
	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted)
	s.InDelta(100.0, complete.AudioDurationMs, 1.0, "only the audio that arrived is billed")
}

func (s *GeminiSocketSuite) TestInterruptStopsTheAudioMidResponse() {
	fake := newFakeGemini()
	defer fake.server.Close()
	// A long response delivered slowly, so the interrupt lands while it is streaming.
	fake.events, fake.pause = 100, 20*time.Millisecond
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "a very long sentence", Final: true}))
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})
	s.Require().NoError(provider.Interrupt())

	events := s.settled(provider)
	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted)
	s.Less(complete.AudioDurationMs, 10_000.0, "barge-in should not bill the whole utterance")
	for _, event := range events {
		_, failed := event.(tts.Error)
		s.False(failed, "barge-in is not a provider failure")
	}
}

func (s *GeminiSocketSuite) TestCloseSettlesWhatWasInFlight() {
	fake := newFakeGemini()
	defer fake.server.Close()
	fake.events, fake.pause = 100, 20*time.Millisecond
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "a long answer", Final: true}))
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})
	s.Require().NoError(provider.Close())

	var completions []tts.SynthesisComplete
	for event := range provider.Events() {
		if complete, ok := event.(tts.SynthesisComplete); ok {
			completions = append(completions, complete)
		}
	}
	s.Require().Len(completions, 1, "the utterance owes exactly one completion")
	s.Equal("u1", completions[0].SynthesisID)
	s.True(completions[0].Interrupted)
}
