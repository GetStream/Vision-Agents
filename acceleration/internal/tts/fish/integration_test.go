//go:build integration

package fish

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// sentence takes a couple of seconds to say, which is long enough for the duration and
// chunking assertions to mean something.
const sentence = "The quick brown fox jumps over the lazy dog, and then it does it again."

type FishIntegrationSuite struct {
	suite.Suite
}

func TestFishIntegrationSuite(t *testing.T) {
	suite.Run(t, new(FishIntegrationSuite))
}

func (s *FishIntegrationSuite) SetupSuite() {
	if os.Getenv("FISH_API_KEY") == "" {
		s.T().Skip("FISH_API_KEY not set")
	}
}

func (s *FishIntegrationSuite) start(options Options) *TTS {
	provider, err := New(options)
	s.Require().NoError(err)

	s.Require().NoError(provider.Start(context.Background()))
	return provider
}

// collect reads events until the predicate is satisfied. A provider error fails the test
// straight away, so a rejected request reports what went wrong instead of timing out.
func (s *FishIntegrationSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
	var events []tts.Event
	deadline := time.After(90 * time.Second)

	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
			if failure, failed := event.(tts.Error); failed {
				s.FailNowf("provider error", "%v", failure.Err)
			}
			if until(event) {
				return events
			}
		case <-deadline:
			s.FailNow("timed out waiting for audio")
			return events
		}
	}
}

func (s *FishIntegrationSuite) TestSynthesizesRealSpeech() {
	provider := s.start(Options{})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: sentence, Final: true}))

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

	s.Require().NotEmpty(chunks, "the model should have said something")
	s.Greater(len(chunks), 1, "audio should stream rather than arrive in one lump")
	s.Equal(DefaultSampleRate, chunks[0].Audio.SampleRate)
	s.Equal(1, chunks[0].Audio.Channels)

	s.EqualValues(len(sentence), complete.Characters)
	// Roughly 15 words at a conversational pace.
	s.Greater(complete.AudioDurationMs, 2_000.0)
	s.Less(complete.AudioDurationMs, 20_000.0)
	s.Positive(complete.TimeToFirstByteMs)
	s.False(complete.Interrupted)
}

func (s *FishIntegrationSuite) TestSynthesizesAtTheRequestedSampleRate() {
	provider := s.start(Options{SampleRate: 44_100})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	s.Require().NoError(provider.Synthesize(tts.Request{Text: sentence, Final: true}))

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

	s.Require().NotEmpty(chunks)
	s.Equal(44_100, chunks[0].Audio.SampleRate)
	// A wrong rate would show up here as speech at the wrong speed.
	s.Greater(complete.AudioDurationMs, 2_000.0)
	s.Less(complete.AudioDurationMs, 20_000.0)
}

func (s *FishIntegrationSuite) TestDeltasAreBufferedIntoOneUtterance() {
	provider := s.start(Options{})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	for _, word := range []string{"The quick ", "brown fox ", "jumps over ", "the lazy dog."} {
		s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: word}))
	}
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Final: true}))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete := events[len(events)-1].(tts.SynthesisComplete)

	s.EqualValues(len("The quick brown fox jumps over the lazy dog."), complete.Characters)
	s.Greater(complete.AudioDurationMs, 1_000.0)
}
