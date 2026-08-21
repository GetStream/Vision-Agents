//go:build integration

package cartesia

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

type CartesiaIntegrationSuite struct {
	suite.Suite
}

func TestCartesiaIntegrationSuite(t *testing.T) {
	suite.Run(t, new(CartesiaIntegrationSuite))
}

func (s *CartesiaIntegrationSuite) SetupSuite() {
	if os.Getenv("CARTESIA_API_KEY") == "" {
		s.T().Skip("CARTESIA_API_KEY not set")
	}
}

func (s *CartesiaIntegrationSuite) start(options Options) *TTS {
	provider, err := New(options)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	return provider
}

// collect reads events until the predicate is satisfied. A provider error fails the test
// straight away, so a rejected request reports what went wrong instead of timing out.
func (s *CartesiaIntegrationSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
	var events []tts.Event
	deadline := time.After(60 * time.Second)

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

func (s *CartesiaIntegrationSuite) TestSynthesizesRealSpeech() {
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
		case tts.Error:
			s.Failf("provider error", "%v", typed.Err)
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

func (s *CartesiaIntegrationSuite) TestStreamingDeltasStartAudioBeforeTheTurnEnds() {
	// This is how the agent speaks: a sentence at a time into one context, closed at the
	// end of the turn. It only stays fast because the server is told not to buffer.
	provider := s.start(Options{})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	for _, part := range []string{"The quick brown fox jumps. ", "Then it does it again."} {
		s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: part}))
	}
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Final: true}))

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	complete := events[len(events)-1].(tts.SynthesisComplete)

	s.EqualValues(len("The quick brown fox jumps. Then it does it again."), complete.Characters)
	s.Greater(complete.AudioDurationMs, 1_000.0)
	s.Less(complete.TimeToFirstByteMs, 2_000.0,
		"a buffering server would sit on the first sentence instead of saying it")
}

func (s *CartesiaIntegrationSuite) TestInterruptStopsALongUtterance() {
	provider := s.start(Options{})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	long := sentence + " " + sentence + " " + sentence + " " + sentence
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: long, Final: true}))

	// Cut in as soon as the first sound arrives, the way a user talking over the agent
	// would.
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

	s.True(complete.Interrupted)
	s.Less(complete.AudioDurationMs, 20_000.0, "barge-in should not bill the whole utterance")
}
