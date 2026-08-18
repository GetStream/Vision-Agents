package session

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type ReviewSuite struct {
	suite.Suite
}

func TestReviewSuite(t *testing.T) {
	suite.Run(t, new(ReviewSuite))
}

func (s *ReviewSuite) TestTheCallIsQuotedWithBothSidesNamed() {
	written := conversation([]spoken{
		{text: "are you open on Sundays?"},
		{agent: true, text: "we are, from ten."},
	})

	s.Contains(written, "Caller: are you open on Sundays?")
	s.Contains(written, "Agent: we are, from ten.")
}

func (s *ReviewSuite) TestALongCallIsReviewedFromItsBeginning() {
	// What a call was about is said at the start of it, so the part that is dropped is
	// the end rather than the beginning.
	var said []spoken
	said = append(said, spoken{text: "I want to change my flight"})
	for range reviewLimit + 50 {
		said = append(said, spoken{agent: true, text: "one moment."})
	}

	written := conversation(said)

	s.Contains(written, "I want to change my flight")
	s.Equal(reviewLimit, strings.Count(written, "\n")-2, "more of the call was sent than fits")
}

func (s *ReviewSuite) TestAJudgementIsReadOutOfWhatTheModelWrote() {
	verdict, err := parseJudgement(
		"```json\n{\"summary\":\"they wanted a refund\",\"score\":4,\"notes\":\"knew the policy\"}\n```")

	s.Require().NoError(err)
	s.Equal("they wanted a refund", verdict.Summary)
	s.Equal(4, verdict.Score)
	s.Equal("knew the policy", verdict.Notes)
}

func (s *ReviewSuite) TestAReviewThatSaidNothingIsNotOne() {
	// An empty summary written onto the row would read as a call nobody could describe
	// rather than as a review that failed.
	_, err := parseJudgement(`{"summary":"","score":5}`)

	s.Require().Error(err)
}

func (s *ReviewSuite) TestAnAnswerThatIsNotJsonIsRefused() {
	_, err := parseJudgement("I think the call went quite well.")

	s.Require().Error(err)
}
