package mock

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

func TestMockTTSSuite(t *testing.T) { suite.Run(t, new(MockTTSSuite)) }

// sentence takes a couple of seconds to say at the default pace.
const sentence = "The quick brown fox jumps over the lazy dog, and then it does it again."

type MockTTSSuite struct {
	suite.Suite

	provider *TTS
}

func (s *MockTTSSuite) SetupTest() {
	s.provider = s.started(Options{})
}

func (s *MockTTSSuite) started(options Options) *TTS {
	provider, err := New(options)
	s.Require().NoError(err)
	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { s.Require().NoError(provider.Close()) })
	return provider
}

// settled drains events until the utterance finishes, and returns it with its audio.
func (s *MockTTSSuite) settled(provider *TTS) (tts.SynthesisComplete, []tts.AudioChunk) {
	var chunks []tts.AudioChunk
	deadline := time.After(10 * time.Second)
	for {
		select {
		case event := <-provider.Events():
			switch typed := event.(type) {
			case tts.AudioChunk:
				chunks = append(chunks, typed)
			case tts.SynthesisComplete:
				return typed, chunks
			}
		case <-deadline:
			s.FailNow("timed out waiting for the utterance to finish")
			return tts.SynthesisComplete{}, nil
		}
	}
}

func (s *MockTTSSuite) TestASentenceBecomesSpeechOfTheRightLength() {
	s.Require().NoError(s.provider.Synthesize(tts.Request{ID: "u1", Text: sentence, Final: true}))

	complete, chunks := s.settled(s.provider)

	s.Greater(len(chunks), 1, "audio should stream rather than arrive in one lump")
	s.Equal(DefaultSampleRate, chunks[0].Audio.SampleRate)
	s.Equal(1, chunks[0].Audio.Channels)
	s.EqualValues(len(sentence), complete.Characters)
	s.InDelta(float64(len(sentence))/defaultCharsPerSecond*1000, complete.AudioDurationMs, 1)
	s.False(complete.Interrupted)
}

func (s *MockTTSSuite) TestTheAudioArrivesFasterThanItPlays() {
	started := time.Now()
	s.Require().NoError(s.provider.Synthesize(tts.Request{ID: "u1", Text: sentence, Final: true}))

	complete, _ := s.settled(s.provider)

	s.Less(time.Since(started).Seconds()*1000, complete.AudioDurationMs,
		"a voice that generated slower than it speaks would starve the speaker")
	s.Positive(complete.TimeToFirstByteMs)
}

func (s *MockTTSSuite) TestDeltasBecomeOneUtterance() {
	for _, delta := range strings.SplitAfter(sentence, " ") {
		s.Require().NoError(s.provider.Synthesize(tts.Request{ID: "u1", Text: delta}))
	}
	s.Require().NoError(s.provider.Synthesize(tts.Request{ID: "u1", Final: true}))

	complete, _ := s.settled(s.provider)

	s.EqualValues(len(sentence), complete.Characters)
}

func (s *MockTTSSuite) TestBargeInStopsTheRestOfTheUtterance() {
	long := strings.Repeat(sentence+" ", 4)
	s.Require().NoError(s.provider.Synthesize(tts.Request{ID: "u1", Text: long, Final: true}))

	// Cut in as soon as the first sound arrives, the way a caller talking over the agent
	// would.
	for event := range s.provider.Events() {
		if _, speaking := event.(tts.AudioChunk); speaking {
			break
		}
	}
	s.Require().NoError(s.provider.Interrupt())

	complete, _ := s.settled(s.provider)

	s.True(complete.Interrupted)
	s.Less(complete.AudioDurationMs, float64(len(long))/defaultCharsPerSecond*1000,
		"barge-in should not bill the whole utterance")
}

func (s *MockTTSSuite) TestTextBeforeTheSessionIsOpenIsRefused() {
	provider, err := New(Options{})
	s.Require().NoError(err)

	s.Error(provider.Synthesize(tts.Request{ID: "u1", Text: sentence, Final: true}))
}
