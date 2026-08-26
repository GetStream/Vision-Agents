package tts

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type DirectionsSuite struct {
	suite.Suite
}

func TestDirectionsSuite(t *testing.T) {
	suite.Run(t, new(DirectionsSuite))
}

// stripped is the reply as a reader would see it, written one delta at a time.
func (s *DirectionsSuite) stripped(deltas ...string) string {
	var directions Directions
	var kept strings.Builder
	for _, delta := range deltas {
		kept.WriteString(directions.Add(delta))
	}
	kept.WriteString(directions.Flush())
	return kept.String()
}

func (s *DirectionsSuite) TestADirectionIsTakenOutOfTheLine() {
	s.Equal("That is a good one.", s.stripped("[laughs] That is a good one."))
}

func (s *DirectionsSuite) TestADirectionLeavesNoGapWhereItWas() {
	// Two spaces where a word was lifted out is audible: a voice reading it pauses.
	s.Equal("Well, fine.", s.stripped("Well, [sighs] fine."))
}

func (s *DirectionsSuite) TestADirectionSplitAcrossDeltasIsStillWhole() {
	s.Equal("Well fine.", s.stripped("Well ", "[sig", "hs]", " fine."))
}

func (s *DirectionsSuite) TestTextBeforeADirectionIsReleasedWithoutWaiting() {
	// Whoever is reading is waiting for it, so only what might be a direction is held.
	var directions Directions

	s.Equal("Hello there. ", directions.Add("Hello there. [wh"))
}

func (s *DirectionsSuite) TestABracketThatNeverClosesWasOnlyEverText() {
	s.Equal("The total is [1", s.stripped("The total is [1"))
}

func (s *DirectionsSuite) TestABracketTooLongToBeADirectionStopsBeingTreatedAsOne() {
	// Holding text back for a closing bracket that is never coming would leave the reader
	// watching nothing while the model talks.
	rambling := "[" + strings.Repeat("a", directionLimit+10)

	s.Equal(rambling, s.stripped(rambling))
}

func (s *DirectionsSuite) TestSeveralDirectionsInOneLineAllGo() {
	s.Equal("Fine. Really.", s.stripped("[sighs] Fine. [pause] Really."))
}

func (s *DirectionsSuite) TestALineWithNoDirectionsIsUntouched() {
	s.Equal("Hello there. How are you?", s.stripped("Hello there. ", "How are you?"))
}

func (s *DirectionsSuite) TestAReplyThatWasAbandonedIsForgotten() {
	var directions Directions
	directions.Add("Hello [wh")

	directions.Reset()

	s.Equal("Goodbye.", directions.Add("Goodbye."))
}
