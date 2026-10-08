package speechify

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// fakeSpeechify is an HTTP server that answers /v1/audio/stream the way Speechify does,
// so the provider can be driven over a real request without an API key.
type fakeSpeechify struct {
	server *httptest.Server

	mu sync.Mutex
	// requests is every body the provider posted.
	requests []streamRequest
	headers  []http.Header
	paths    []string

	// samples is how much audio to answer with, split into writes of samplesPerWrite.
	samples         int
	samplesPerWrite int
	// status, when set, is answered instead of audio.
	status int
	body   string
	// pause is held between writes, so a test can interrupt mid-response.
	pause time.Duration
}

func newFakeSpeechify() *fakeSpeechify {
	fake := &fakeSpeechify{samples: 2400, samplesPerWrite: 2400}

	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return
		}
		var request streamRequest
		_ = json.Unmarshal(body, &request)

		fake.mu.Lock()
		fake.requests = append(fake.requests, request)
		fake.headers = append(fake.headers, r.Header.Clone())
		fake.paths = append(fake.paths, r.URL.Path)
		status, failure := fake.status, fake.body
		samples, perWrite, pause := fake.samples, fake.samplesPerWrite, fake.pause
		fake.mu.Unlock()

		if status != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_, _ = w.Write([]byte(failure))
			return
		}

		var rate int
		_, _ = fmt.Sscanf(request.OutputFormat, "pcm_%d", &rate)
		w.Header().Set("Content-Type", fmt.Sprintf("audio/L16; rate=%d; channels=1", rate))
		flusher, _ := w.(http.Flusher)
		for written := 0; written < samples; written += perWrite {
			block := min(perWrite, samples-written)
			pcm := audio.PcmData{Samples: make([]int16, block), SampleRate: rate, Channels: 1}
			if _, err := w.Write(pcm.Bytes()); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			if pause > 0 {
				select {
				case <-time.After(pause):
				case <-r.Context().Done():
					return
				}
			}
		}
	}))
	return fake
}

func (f *fakeSpeechify) posted() []streamRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]streamRequest(nil), f.requests...)
}

type SpeechifySocketSuite struct {
	suite.Suite
}

func TestSpeechifySocketSuite(t *testing.T) {
	suite.Run(t, new(SpeechifySocketSuite))
}

// start returns a started provider pointed at the fake.
func (s *SpeechifySocketSuite) start(fake *fakeSpeechify, options Options) *TTS {
	options.APIKey = "test-key"
	options.BaseURL = fake.server.URL
	provider, err := New(options)
	s.Require().NoError(err)
	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { _ = provider.Close() })
	return provider
}

// settle reads events until the utterance in flight completes.
func (s *SpeechifySocketSuite) settle(provider *TTS) []tts.Event {
	return s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
}

// collect reads events until the predicate is satisfied or the wait runs out.
func (s *SpeechifySocketSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
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

// first returns the earliest event of type T.
func first[T tts.Event](s *SpeechifySocketSuite, events []tts.Event) T {
	for _, event := range events {
		if typed, ok := event.(T); ok {
			return typed
		}
	}
	var zero T
	s.FailNowf("event not emitted", "no %T among %d events", zero, len(events))
	return zero
}

func (s *SpeechifySocketSuite) TestOneUtteranceIsOnePostAndStreamsItsAudioBack() {
	fake := newFakeSpeechify()
	defer fake.server.Close()
	// 4800 samples at 24 kHz is 200 ms, arriving in two writes.
	fake.samples, fake.samplesPerWrite = 4800, 2400
	provider := s.start(fake, Options{VoiceID: "hugh_32"})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello there", Final: true}))

	events := s.settle(provider)
	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.EqualValues(len("hello there"), complete.Characters)
	s.InDelta(200.0, complete.AudioDurationMs, 1.0)
	s.Positive(complete.TimeToFirstByteMs)
	s.False(complete.Interrupted)

	posted := fake.posted()
	s.Require().Len(posted, 1, "a whole utterance is a single request")
	s.Equal(streamRequest{
		Input:        "hello there",
		VoiceID:      "hugh_32",
		Model:        DefaultModel,
		OutputFormat: "pcm_24000",
	}, posted[0])

	s.Equal("/v1/audio/stream", fake.paths[0])
	s.Equal("Bearer test-key", fake.headers[0].Get("Authorization"))
	s.Equal("audio/pcm", fake.headers[0].Get("Accept"))
	s.Equal(caller, fake.headers[0].Get("Speechify-Caller"))
}

func (s *SpeechifySocketSuite) TestARequestVoiceOverridesTheSession() {
	fake := newFakeSpeechify()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Voice: "imogen_32", Final: true}))
	events := s.settle(provider)

	started := first[tts.SynthesisStarted](s, events)
	s.Equal("imogen_32", started.Voice)
	s.Equal("imogen_32", fake.posted()[0].VoiceID)
}

func (s *SpeechifySocketSuite) TestDeltasAreBufferedIntoOneRequest() {
	fake := newFakeSpeechify()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello "}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "there"}))
	s.Empty(fake.posted(), "a partial utterance should not be sent")

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Final: true}))
	events := s.settle(provider)

	s.Equal("u1", events[len(events)-1].(tts.SynthesisComplete).SynthesisID)
	posted := fake.posted()
	s.Require().Len(posted, 1)
	s.Equal("hello there", posted[0].Input)
}

func (s *SpeechifySocketSuite) TestTheSampleRateIsAskedForAndReported() {
	fake := newFakeSpeechify()
	defer fake.server.Close()
	provider := s.start(fake, Options{SampleRate: 16_000})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Final: true}))
	events := s.settle(provider)

	s.Equal("pcm_16000", fake.posted()[0].OutputFormat)
	chunk := first[tts.AudioChunk](s, events)
	s.Equal(16_000, chunk.Audio.SampleRate)
	s.Equal(1, chunk.Audio.Channels)
	s.Equal(0, chunk.Index)
}

func (s *SpeechifySocketSuite) TestARefusedRequestIsReportedAndStillSettles() {
	fake := newFakeSpeechify()
	defer fake.server.Close()
	fake.status = http.StatusPaymentRequired
	fake.body = `{"error":{"code":"payment_required","message":"insufficient credits"}}`
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Final: true}))
	events := s.settle(provider)

	var failure tts.Error
	var sawFailure bool
	for _, event := range events {
		if typed, ok := event.(tts.Error); ok {
			failure, sawFailure = typed, true
		}
	}
	s.Require().True(sawFailure, "a rejected request should reach the caller")
	s.ErrorContains(failure.Err, "http 402")
	s.True(strings.Contains(failure.Err.Error(), "payment_required"))

	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted, "a failed synthesis produced no speech")
	s.Zero(complete.AudioDurationMs)
}

func (s *SpeechifySocketSuite) TestInterruptStopsTheAudioMidResponse() {
	fake := newFakeSpeechify()
	defer fake.server.Close()
	// A long response delivered slowly, so the interrupt lands while it is streaming.
	fake.samples, fake.samplesPerWrite = 240_000, 2400
	fake.pause = 20 * time.Millisecond
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "a very long sentence", Final: true}))
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})
	s.Require().NoError(provider.Interrupt())

	events := s.settle(provider)
	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted)
	s.Less(complete.AudioDurationMs, 10_000.0, "barge-in should not bill the whole utterance")
}
