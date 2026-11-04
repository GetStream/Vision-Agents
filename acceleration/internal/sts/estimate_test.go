package sts

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// spoken is a reply that streamed ms of audio at 24 kHz.
func spoken(ms int) *Turn {
	turn := NewTurn("", 1, time.Time{})
	turn.Chunk(audio.FromBytes(make([]byte, 24*ms*2), 24_000, 1))
	return turn
}

func (s *STSSuite) TestAVendorWithNoSplitHasItsEstimateInTheTotals() {
	estimator := NewEstimator(20, false)

	used := estimator.Estimate(spoken(1_000))

	s.Equal(Usage{OutputTokens: 20, Estimated: true}, used)
}

func (s *STSSuite) TestAnEstimateIsNeverLearnedFrom() {
	estimator := NewEstimator(20, true)
	estimator.Reported(spoken(1_000), Usage{InputTokens: 9, OutputTokens: 500, OutputAudioTokens: 500, Estimated: true})

	used := estimator.Estimate(spoken(1_000))

	s.EqualValues(20, used.OutputAudioTokens, "the rate is still the vendor's")
	s.Zero(used.InputTokens)
}

func (s *STSSuite) TestAReplyThatStreamedNothingBeforeAnyCountIsNotEstimated() {
	estimator := NewEstimator(20, true)

	s.Equal(Usage{}, estimator.Estimate(NewTurn("", 1, time.Time{})))
}
