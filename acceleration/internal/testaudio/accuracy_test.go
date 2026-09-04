package testaudio

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type AccuracySuite struct {
	suite.Suite
}

func TestAccuracySuite(t *testing.T) {
	suite.Run(t, new(AccuracySuite))
}

func (s *AccuracySuite) TestTheSameSentenceScoresPerfectly() {
	said := "In a quiet village where the sky brushes the fields in hues of gold"

	s.Equal(1.0, Accuracy(said, said))
}

func (s *AccuracySuite) TestCasingAndPunctuationAreNotHeldAgainstAProvider() {
	s.Equal(1.0, Accuracy(
		"In a quiet village, young Mia discovered a map.",
		"in a quiet village young mia discovered a map",
	))
}

func (s *AccuracySuite) TestOneWrongWordInTenCostsATenth() {
	s.InDelta(0.9, Accuracy(
		"one two three four five six seven eight nine ten",
		"one two three four five six seven eight nine eleven",
	), 0.001)
}

func (s *AccuracySuite) TestADroppedWordCostsAsMuchAsAWrongOne() {
	s.InDelta(0.9, Accuracy(
		"one two three four five six seven eight nine ten",
		"one two three four five six seven eight nine",
	), 0.001)
}

func (s *AccuracySuite) TestWordsNobodySaidCountAgainstTheTranscript() {
	s.InDelta(0.8, Accuracy(
		"one two three four five six seven eight nine ten",
		"one two three four five six seven eight nine ten eleven twelve",
	), 0.001)
}

func (s *AccuracySuite) TestATranscriptOfNothingScoresNothing() {
	s.Zero(Accuracy("in a quiet village", ""))
}

func (s *AccuracySuite) TestATranscriptWithNothingInCommonScoresNothing() {
	s.Zero(Accuracy("in a quiet village", "entirely different words spoken here"))
}

func (s *AccuracySuite) TestAlignSplitsSubstitutionsFromInsertions() {
	got := Align(
		"one two three four five",
		"one two three four five six",
	)
	s.Equal(0, got.Substitutions)
	s.Equal(1, got.Insertions)
	s.Equal(0, got.Deletions)
	s.InDelta(0.2, got.WER(), 0.001)
}
