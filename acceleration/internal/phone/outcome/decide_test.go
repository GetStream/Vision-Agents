package outcome

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type DecideSuite struct {
	suite.Suite
	base time.Time
	call Call
}

func TestDecideSuite(t *testing.T) { suite.Run(t, new(DecideSuite)) }

func (s *DecideSuite) SetupTest() {
	s.base = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC)
	s.call = Call{Vendor: "telnyx", Reports: true, PlacedAt: s.base}
}

// at is the base time and n seconds.
func (s *DecideSuite) at(n int) time.Time { return s.base.Add(time.Duration(n) * time.Second) }

func (s *DecideSuite) report(call Call, reported Outcome, word string, at time.Time) Call {
	return decide(call, event{kind: reportedEvent, reported: reported, word: word, at: at})
}

func (s *DecideSuite) answer(call Call, at time.Time) Call {
	return decide(call, event{kind: answeredEvent, at: at})
}

func (s *DecideSuite) end(call Call, at time.Time) Call {
	return decide(call, event{kind: endedEvent, at: at})
}

func (s *DecideSuite) TestTheFirstFinalOutcomeIsKept() {
	call := s.report(s.call, Busy, "busy", s.at(1))
	call = s.report(call, Completed, "completed", s.at(2))

	s.Equal(Busy, call.Outcome)
	s.Equal("busy", call.VendorStatus)
	s.Equal(s.at(1), *call.DecidedAt)
}

func (s *DecideSuite) TestAWordThatDecidesNothingIsKeptForDebugging() {
	call := s.report(s.call, "", "ringing", s.at(1))

	s.Empty(call.Outcome)
	s.Nil(call.DecidedAt)
	s.Equal("ringing", call.VendorStatus)
}

func (s *DecideSuite) TestAReportAfterThePhoneJoinedIsCompleted() {
	call := s.answer(s.call, s.at(1))
	call = s.report(call, NoAnswer, "no-answer", s.at(2))

	s.Equal(Completed, call.Outcome)
	s.Equal(s.at(2), *call.DecidedAt)
}

func (s *DecideSuite) TestAJoinAfterAVendorsOutcomeMakesItCompleted() {
	call := s.report(s.call, Busy, "busy", s.at(1))
	call = s.answer(call, s.at(2))

	s.Equal(Completed, call.Outcome)
	s.Equal(s.at(2), *call.DecidedAt)
}

func (s *DecideSuite) TestAJoinAloneDecidesNothing() {
	call := s.answer(s.call, s.at(1))

	s.Empty(call.Outcome)
	s.Nil(call.DecidedAt)
	s.Equal(s.at(1), *call.AnsweredAt)
}

func (s *DecideSuite) TestACallEndingAfterItsPhoneJoinedIsCompleted() {
	call := s.report(s.call, Busy, "busy", s.at(1))
	call = s.answer(call, s.at(2))
	call = s.end(call, s.at(3))

	s.Equal(Completed, call.Outcome)
	s.Equal(s.at(3), *call.EndedAt)
}

func (s *DecideSuite) TestACallEndingNeverJoinedIsNoAnswerWhenTheVendorReports() {
	call := s.end(s.call, s.at(1))

	s.Equal(NoAnswer, call.Outcome)
	s.Equal(s.at(1), *call.DecidedAt)
}

func (s *DecideSuite) TestACallEndingNeverJoinedIsUnknownWhenTheVendorDoesNot() {
	s.call.Reports = false

	call := s.end(s.call, s.at(1))

	s.Equal(Unknown, call.Outcome)
}

func (s *DecideSuite) TestACallEndingDoesNotOverwriteAVendorsOutcome() {
	call := s.report(s.call, Busy, "busy", s.at(1))
	call = s.end(call, s.at(2))

	s.Equal(Busy, call.Outcome)
	s.Equal(s.at(1), *call.DecidedAt)
}

func (s *DecideSuite) TestASecondJoinOrEndKeepsTheFirstTime() {
	call := s.answer(s.call, s.at(1))
	call = s.answer(call, s.at(2))
	call = s.end(call, s.at(3))
	call = s.end(call, s.at(4))

	s.Equal(s.at(1), *call.AnsweredAt)
	s.Equal(s.at(3), *call.EndedAt)
}

func (s *DecideSuite) TestOnlyBusyNoAnswerFailedAndCanceledAreUnanswered() {
	s.True(Busy.Unanswered())
	s.True(NoAnswer.Unanswered())
	s.True(Failed.Unanswered())
	s.True(Canceled.Unanswered())
	s.False(Completed.Unanswered())
	s.False(Unknown.Unanswered())
	s.False(Outcome("").Unanswered())
}

func (s *DecideSuite) TestABusyCallWhosePhoneJoinedWasAnswered() {
	busy := s.report(s.call, Busy, "busy", s.at(1))
	s.True(busy.NobodyAnswered())

	busy.AnsweredAt = new(s.at(2))

	s.False(busy.NobodyAnswered())
}

func (s *DecideSuite) TestADurationRunsFromTheJoinToTheEnd() {
	call := s.answer(s.call, s.at(1))
	_, known := call.Duration()
	s.False(known, "not until the call ends")

	call = s.end(call, s.at(31))

	duration, known := call.Duration()
	s.True(known)
	s.Equal(30*time.Second, duration)
}
