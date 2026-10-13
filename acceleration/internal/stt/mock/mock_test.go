package mock

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func TestMockSTTSuite(t *testing.T) { suite.Run(t, new(MockSTTSuite)) }

type MockSTTSuite struct {
	suite.Suite

	provider *STT
}

func (s *MockSTTSuite) SetupTest() {
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Require().NoError(provider.Start(context.Background()))
	s.provider = provider
	s.T().Cleanup(func() { s.Require().NoError(provider.Close()) })

	s.drain()
}

// speak feeds loud frames, and quiet the silence that ends a turn. Both are 100ms, the
// size a call delivers audio in.
func (s *MockSTTSuite) speak(ms int) { s.feed(ms, 0.3) }
func (s *MockSTTSuite) quiet(ms int) { s.feed(ms, 0) }

func (s *MockSTTSuite) feed(ms int, level float64) {
	for sent := 0; sent < ms; sent += 100 {
		s.Require().NoError(s.provider.ProcessAudio(frame(100, level), stt.Participant{ID: "caller"}))
	}
}

// transcripts is everything waiting on the channel, which is everything the frames so far
// produced: the provider emits on the goroutine that fed it.
func (s *MockSTTSuite) transcripts() []stt.Transcript {
	var heard []stt.Transcript
	for {
		select {
		case event := <-s.provider.Events():
			if transcript, ok := event.(stt.Transcript); ok {
				heard = append(heard, transcript)
			}
		default:
			return heard
		}
	}
}

func (s *MockSTTSuite) drain() { s.transcripts() }

func (s *MockSTTSuite) TestTheQuietAfterSpeechSettlesATurn() {
	s.speak(1000)
	s.quiet(500)
	s.Empty(final(s.transcripts()), "500ms of quiet is a pause, not the end of a turn")

	s.quiet(200)

	settled := final(s.transcripts())
	s.Require().Len(settled, 1)
	s.Equal(DefaultTranscript, settled[0].Text)
	s.Equal("caller", settled[0].Participant.ID)
	s.InDelta(1000, settled[0].AudioDurationMs, 1)
}

func (s *MockSTTSuite) TestEachTurnIsNumbered() {
	s.speak(1000)
	s.quiet(700)
	s.speak(1000)
	s.quiet(700)

	settled := final(s.transcripts())
	s.Require().Len(settled, 2)
	s.EqualValues(1, settled[0].Utterance)
	s.EqualValues(2, settled[1].Utterance)
}

func (s *MockSTTSuite) TestTheWordsArriveWhileTheCallerIsStillTalking() {
	s.speak(1000)

	heard := s.transcripts()
	s.NotEmpty(heard, "a caller a second into a turn should already have been written down")
	for _, transcript := range heard {
		s.Equal(stt.ModeReplacement, transcript.Mode)
		s.EqualValues(1, transcript.Utterance, "the words so far belong to the turn about to settle")
	}
}

func (s *MockSTTSuite) TestAShortNoiseIsNotATurn() {
	s.speak(100)
	s.quiet(700)

	s.Empty(final(s.transcripts()), "a cough is not somebody talking")
}

func (s *MockSTTSuite) TestQuietAloneSaysNothing() {
	s.quiet(3000)

	s.Empty(s.transcripts())
}

func (s *MockSTTSuite) TestAudioBeforeTheSessionIsOpenIsRefused() {
	provider, err := New(Options{})
	s.Require().NoError(err)

	s.Error(provider.ProcessAudio(frame(100, 0.3), stt.Participant{}))
}

func (s *MockSTTSuite) TestAudioAtTheWrongRateIsRefused() {
	wrong := frame(100, 0.3)
	wrong.SampleRate = 48_000

	s.ErrorContains(s.provider.ProcessAudio(wrong, stt.Participant{}), "sample rate")
}

func final(heard []stt.Transcript) []stt.Transcript {
	var settled []stt.Transcript
	for _, transcript := range heard {
		if transcript.Final() {
			settled = append(settled, transcript)
		}
	}
	return settled
}

// frame is a chunk of tone at the given fraction of full scale, which is what the
// provider measures to tell speech from the room.
func frame(ms int, level float64) stt.PcmData {
	samples := make([]int16, stt.SampleRate*ms/1000)
	for i := range samples {
		samples[i] = int16(math.Sin(float64(i)*0.1) * level * math.MaxInt16)
	}
	return stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
}
