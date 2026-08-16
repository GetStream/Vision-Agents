package fish

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// fakeFish is an HTTP server that answers /v1/tts the way Fish does, so the provider can
// be driven over a real request without an API key.
type fakeFish struct {
	server *httptest.Server

	mu sync.Mutex
	// requests is every body the provider posted.
	requests []synthesisRequest
	headers  []http.Header

	// samples is how much audio to answer with, split into writes of samplesPerWrite.
	samples         int
	samplesPerWrite int
	// status, when set, is answered instead of audio.
	status int
	body   string
	// pause is held between writes, so a test can interrupt mid-response.
	pause time.Duration
}

func newFakeFish() *fakeFish {
	fake := &fakeFish{samples: 2400, samplesPerWrite: 2400}

	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return
		}
		var request synthesisRequest
		_ = json.Unmarshal(body, &request)

		fake.mu.Lock()
		fake.requests = append(fake.requests, request)
		fake.headers = append(fake.headers, r.Header.Clone())
		status, failure := fake.status, fake.body
		samples, perWrite, pause := fake.samples, fake.samplesPerWrite, fake.pause
		fake.mu.Unlock()

		if status != 0 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(failure))
			return
		}

		flusher, _ := w.(http.Flusher)
		for written := 0; written < samples; written += perWrite {
			block := min(perWrite, samples-written)
			pcm := audio.PcmData{
				Samples:    make([]int16, block),
				SampleRate: request.SampleRate,
				Channels:   1,
			}
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

func (f *fakeFish) posted() []synthesisRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]synthesisRequest(nil), f.requests...)
}

func (f *fakeFish) sentHeaders() []http.Header {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]http.Header(nil), f.headers...)
}

type FishSuite struct {
	suite.Suite
}

func TestFishSuite(t *testing.T) {
	suite.Run(t, new(FishSuite))
}

// newTTS returns a provider that is wired up but never started.
func (s *FishSuite) newTTS(options Options) *TTS {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// start returns a started provider pointed at the fake.
func (s *FishSuite) start(fake *fakeFish, options Options) *TTS {
	options.BaseURL = fake.server.URL
	provider := s.newTTS(options)
	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { _ = provider.Close() })
	return provider
}

// collect reads events until the predicate is satisfied or the wait runs out.
func (s *FishSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
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

func (s *FishSuite) completion(provider *TTS) tts.SynthesisComplete {
	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete, ok := events[len(events)-1].(tts.SynthesisComplete)
	s.Require().True(ok)
	return complete
}

func (s *FishSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("FISH_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *FishSuite) TestNewFallsBackToEnv() {
	s.T().Setenv("FISH_API_KEY", "from-env")
	s.T().Setenv("FISH_VOICE_ID", "voice-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
	s.Equal("voice-from-env", provider.options.Voice)
}

func (s *FishSuite) TestNewDefaultsToS2Pro() {
	provider := s.newTTS(Options{})
	s.Equal(DefaultModel, provider.Model())
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultSampleRate, provider.SampleRate())
	s.False(provider.Streaming(), "one POST per utterance means whole sentences only")
}

func (s *FishSuite) TestSynthesizeFailsBeforeStart() {
	err := s.newTTS(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *FishSuite) TestAPartialWithoutAnIdIsRejected() {
	fake := newFakeFish()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	err := provider.Synthesize(tts.Request{Text: "hello"})
	s.ErrorContains(err, "a partial request needs an id")
}

func (s *FishSuite) TestAnEmptyUtteranceIsRejected() {
	fake := newFakeFish()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	err := provider.Synthesize(tts.Request{Text: "   ", Final: true})
	s.ErrorContains(err, "nothing to say")
}

func (s *FishSuite) TestOneUtteranceIsOnePostAndStreamsItsAudioBack() {
	fake := newFakeFish()
	defer fake.server.Close()
	// 4800 samples at 24 kHz is 200 ms, arriving in two writes.
	fake.samples, fake.samplesPerWrite = 4800, 2400
	provider := s.start(fake, Options{Voice: "ref-1", Latency: "balanced"})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello there", Final: true}))

	complete := s.completion(provider)
	s.EqualValues(len("hello there"), complete.Characters)
	s.InDelta(200.0, complete.AudioDurationMs, 1.0)
	s.Positive(complete.TimeToFirstByteMs)
	s.False(complete.Interrupted)

	posted := fake.posted()
	s.Require().Len(posted, 1, "a whole utterance is a single request")
	s.Equal("hello there", posted[0].Text)
	s.Equal("ref-1", posted[0].ReferenceID)
	s.Equal("pcm", posted[0].Format)
	s.Equal(DefaultSampleRate, posted[0].SampleRate)
	s.Equal("balanced", posted[0].Latency)

	headers := fake.sentHeaders()
	s.Equal("Bearer test-key", headers[0].Get("Authorization"))
	s.Equal(DefaultModel, headers[0].Get("model"), "the model is selected by header")
}

func (s *FishSuite) TestDeltasAreBufferedIntoOneRequest() {
	fake := newFakeFish()
	defer fake.server.Close()
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "hello "}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: "there"}))

	s.Empty(fake.posted(), "a partial utterance should not be sent")

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Final: true}))
	complete := s.completion(provider)

	s.Equal("u1", complete.SynthesisID)
	posted := fake.posted()
	s.Require().Len(posted, 1)
	s.Equal("hello there", posted[0].Text)
}

func (s *FishSuite) TestAudioIsChunkedAtTheConfiguredRate() {
	fake := newFakeFish()
	defer fake.server.Close()
	provider := s.start(fake, Options{SampleRate: 16_000})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Final: true}))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var chunks []tts.AudioChunk
	for _, event := range events {
		if chunk, ok := event.(tts.AudioChunk); ok {
			chunks = append(chunks, chunk)
		}
	}
	s.Require().NotEmpty(chunks)
	s.Equal(16_000, chunks[0].Audio.SampleRate)
	s.Equal(1, chunks[0].Audio.Channels)
	s.Equal(0, chunks[0].Index)
}

func (s *FishSuite) TestAFailedRequestIsReportedAndStillSettles() {
	fake := newFakeFish()
	defer fake.server.Close()
	fake.status, fake.body = http.StatusPaymentRequired, "out of credit"
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "hello", Final: true}))

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
	s.Require().True(sawFailure, "a rejected request should reach the caller")
	s.ErrorContains(failure.Err, "http 402")
	s.ErrorContains(failure.Err, "out of credit")

	complete := events[len(events)-1].(tts.SynthesisComplete)
	s.True(complete.Interrupted, "a failed synthesis produced no speech")
	s.Zero(complete.AudioDurationMs)
}

func (s *FishSuite) TestInterruptStopsTheAudioMidResponse() {
	fake := newFakeFish()
	defer fake.server.Close()
	// A long response delivered slowly, so the interrupt lands while it is streaming.
	fake.samples, fake.samplesPerWrite = 240_000, 2400
	fake.pause = 20 * time.Millisecond
	provider := s.start(fake, Options{})

	s.Require().NoError(provider.Synthesize(tts.Request{Text: "a very long sentence", Final: true}))

	// Wait for audio to be flowing before cutting it off.
	s.collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})
	s.Require().NoError(provider.Interrupt())

	complete := s.completion(provider)
	s.True(complete.Interrupted)
	s.Less(complete.AudioDurationMs, 10_000.0, "barge-in should not bill the whole utterance")
}

func (s *FishSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newTTS(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	for range provider.Events() {
	}
}

func (s *FishSuite) TestSynthesizeFailsAfterClose() {
	provider := s.newTTS(Options{})
	s.Require().NoError(provider.Close())

	err := provider.Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "session closed")
}

func (s *FishSuite) TestSatisfiesTTSInterface() {
	var _ tts.TTS = s.newTTS(Options{})
}
