package agent

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type ChunkerSuite struct {
	suite.Suite
}

func TestChunkerSuite(t *testing.T) {
	suite.Run(t, new(ChunkerSuite))
}

// stream feeds text one delta at a time, which is how a model actually arrives.
func (s *ChunkerSuite) stream(deltas ...string) []string {
	var chunker chunker
	var chunks []string
	for _, delta := range deltas {
		chunks = append(chunks, chunker.Add(delta)...)
	}
	if remainder := chunker.Flush(); remainder != "" {
		chunks = append(chunks, remainder)
	}
	return chunks
}

func (s *ChunkerSuite) TestASentenceIsReturnedAsSoonAsItEnds() {
	// The point of chunking is that the first sentence goes to the voice before the model
	// has finished the reply.
	var chunker chunker

	s.Empty(chunker.Add("The weather today "))
	s.Equal([]string{"The weather today is fine."}, chunker.Add("is fine."))
}

func (s *ChunkerSuite) TestSentencesArriveInOrder() {
	s.Equal(
		[]string{"Hello there.", "How are you?"},
		s.stream("Hello", " there. ", "How are ", "you?"),
	)
}

func (s *ChunkerSuite) TestOneDeltaMayFinishSeveralSentences() {
	s.Equal(
		[]string{"That is one thing.", "This is another!"},
		s.stream("That is one thing. This is another!"),
	)
}

func (s *ChunkerSuite) TestTextWithNoPunctuationIsStillSaid() {
	s.Equal([]string{"a reply that just stops"}, s.stream("a reply that just stops"))
}

func (s *ChunkerSuite) TestAnAbbreviationDoesNotEndASentence() {
	// Splitting on the dot in "Dr." would put a pause in the middle of a name.
	s.Equal([]string{"Dr. Watson was there."}, s.stream("Dr. Watson was there."))
}

func (s *ChunkerSuite) TestANewlineEndsASentence() {
	s.Equal([]string{"the first line", "the second line"}, s.stream("the first line\nthe second line"))
}

func (s *ChunkerSuite) TestFullWidthPunctuationEndsASentence() {
	// The models are multilingual, and these languages do not use the ASCII marks.
	s.Equal([]string{"今天天气很好，我们去散步吧。"}, s.stream("今天天气很好，我们去散步吧。"))
}

func (s *ChunkerSuite) TestWhitespaceIsNotWorthSaying() {
	s.Empty(s.stream("   \n  "))
}

func (s *ChunkerSuite) TestFlushingTwiceReturnsNothingTheSecondTime() {
	var chunker chunker
	chunker.Add("a reply that just stops")

	s.NotEmpty(chunker.Flush())
	s.Empty(chunker.Flush())
}

func (s *ChunkerSuite) TestResettingThrowsAwayAnInterruptedReply() {
	var chunker chunker
	chunker.Add("half a sentence")

	chunker.Reset()

	s.Empty(chunker.Flush(), "an interrupted reply is not finished off after the fact")
}
