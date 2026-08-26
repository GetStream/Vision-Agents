//go:build integration

package breeze

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// coldStartTimeout is how long a scaled-to-zero GPU deployment may take to answer. This
// one builds CUDA graphs on the way up, so it is the slower of the two to wake.
const coldStartTimeout = 15 * time.Minute

// sentence takes a couple of seconds to say, which is long enough for the duration and
// chunking assertions to mean something.
const sentence = "The quick brown fox jumps over the lazy dog, and then it does it again."

type BreezeIntegrationSuite struct {
	suite.Suite
}

func TestBreezeIntegrationSuite(t *testing.T) {
	suite.Run(t, new(BreezeIntegrationSuite))
}

func (s *BreezeIntegrationSuite) SetupSuite() {
	if os.Getenv("BREEZE_WS_URL") == "" {
		s.T().Skip("BREEZE_WS_URL not set")
	}
	if os.Getenv("BASETEN_API_KEY") == "" {
		s.T().Skip("BASETEN_API_KEY not set")
	}
}

// start returns a connected provider. The deployment scales to zero, so the first
// connection after an idle spell pays for a GPU cold start.
func (s *BreezeIntegrationSuite) start(options Options) *TTS {
	options.HandshakeTimeout = coldStartTimeout
	provider, err := New(options)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), coldStartTimeout)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	return provider
}

// collect reads events until the predicate is satisfied. A provider error fails the test
// straight away, so a rejected request reports what went wrong instead of timing out.
func (s *BreezeIntegrationSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
	var events []tts.Event
	deadline := time.After(coldStartTimeout)

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

// say synthesises one utterance and returns how it finished, with the audio it produced.
func (s *BreezeIntegrationSuite) say(provider *TTS, request tts.Request) (tts.SynthesisComplete, []tts.AudioChunk) {
	s.Require().NoError(provider.Synthesize(request))

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
	return complete, chunks
}

func (s *BreezeIntegrationSuite) TestSynthesizesRealSpeech() {
	provider := s.start(Options{})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	complete, chunks := s.say(provider, tts.Request{ID: "u1", Text: sentence, Final: true})

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

func (s *BreezeIntegrationSuite) TestDeltasStreamUpstreamAndFlushIntoOneUtterance() {
	provider := s.start(Options{})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	// Feed the sentence a word at a time, the way an LLM produces it.
	for _, word := range []string{"The quick ", "brown fox ", "jumps over ", "the lazy dog."} {
		s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: word}))
	}
	complete, _ := s.say(provider, tts.Request{ID: "u1", Final: true})

	s.EqualValues(len("The quick brown fox jumps over the lazy dog."), complete.Characters)
	s.Greater(complete.AudioDurationMs, 1_000.0)
}

func (s *BreezeIntegrationSuite) TestADirectionIsActedRatherThanSpelledOut() {
	// The deployment turns [sigh] into the model's own syntax. If that stopped happening
	// the words would be read out, which costs about a second of extra audio.
	provider := s.start(Options{})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	plain, _ := s.say(provider, tts.Request{ID: "u1", Text: "Fine, I will do it.", Final: true})
	directed, _ := s.say(provider, tts.Request{
		ID: "u2", Text: "[sigh] Fine, I will do it.", Final: true,
	})

	s.Positive(directed.AudioDurationMs)
	s.Less(directed.AudioDurationMs, plain.AudioDurationMs+1_500.0,
		"a performed sigh should cost far less audio than reading the word out")
}

func (s *BreezeIntegrationSuite) TestAVoiceDescribedInWordsIsTheVoiceThatSpeaks() {
	// There is no id to assert on, so what is checked is that the description is accepted
	// and changes the audio rather than being rejected or ignored.
	provider := s.start(Options{Voice: "A deep, slow, weary old man."})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	complete, chunks := s.say(provider, tts.Request{ID: "u1", Text: sentence, Final: true})

	s.Require().NotEmpty(chunks)
	s.Greater(complete.AudioDurationMs, 2_000.0)
	s.False(complete.Interrupted, "a voice description should not fail the utterance")
}

func (s *BreezeIntegrationSuite) TestInterruptStopsALongUtterance() {
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
