package campaign

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type TransitionSuite struct {
	suite.Suite
	chicago *time.Location
	hours   CallingHours
}

func TestTransitionSuite(t *testing.T) { suite.Run(t, new(TransitionSuite)) }

func (s *TransitionSuite) SetupTest() {
	loc, err := time.LoadLocation("America/Chicago")
	s.Require().NoError(err)
	s.chicago = loc
	s.hours = DefaultCallingHours()
}

func (s *TransitionSuite) at(day, hour, minute int) time.Time {
	return time.Date(2026, time.November, day, hour, minute, 0, 0, s.chicago)
}

// 2026-11-02 is a Monday.
func (s *TransitionSuite) sequence(double bool) Sequence {
	seq := DefaultSequence("cfg")
	seq.DoubleDial = double
	return seq
}

func (s *TransitionSuite) contact(attempt int) store.Contact {
	return store.Contact{Attempt: attempt}
}

func (s *TransitionSuite) TestNobodyAnsweringTheFirstDialDialsAgainAtOnce() {
	_, again := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(0), 1, Dialed{Result: Unanswered}, s.at(2, 10, 0), s.at(2, 10, 1))
	s.True(again)
}

func (s *TransitionSuite) TestNobodyAnsweringTheDoubleDialSchedulesTheNextAttemptAfterItsDelay() {
	settle, again := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(1), 2, Dialed{Result: Unanswered, CallID: "c2", VendorCallID: "v2"}, s.at(2, 10, 0), s.at(2, 10, 3))
	s.False(again)
	s.Equal(store.Enrolled, settle.Stage)
	s.Equal(2, settle.Attempt)
	s.Equal("step_1", settle.StepID)
	s.Require().NotNil(settle.NextAttemptAt)
	s.True(s.at(5, 10, 0).Equal(*settle.NextAttemptAt), settle.NextAttemptAt.String())
	s.Equal("c2", settle.CallID)
	s.Equal("v2", settle.VendorCallID)
}

func (s *TransitionSuite) TestWithoutDoubleDialTheFirstUnansweredDialSchedulesTheNextAttempt() {
	settle, again := afterDial(s.sequence(false), s.hours, s.chicago, s.contact(0), 1, Dialed{Result: Unanswered}, s.at(2, 10, 0), s.at(2, 10, 1))
	s.False(again)
	s.Equal(store.Enrolled, settle.Stage)
	s.Equal(2, settle.Attempt)
	s.True(s.at(5, 10, 0).Equal(*settle.NextAttemptAt))
}

func (s *TransitionSuite) TestTheNextAttemptLandsInsideCallingHours() {
	started := s.at(2, 17, 0)
	seq := s.sequence(false)
	seq.Steps[0].Attempts = []Attempt{{0}, {180}}
	settle, _ := afterDial(seq, s.hours, s.chicago, s.contact(1), 1, Dialed{Result: Unanswered}, started, started)
	s.Require().NotNil(settle.NextAttemptAt)
	s.True(s.at(3, 9, 0).Equal(*settle.NextAttemptAt), settle.NextAttemptAt.String())
}

func (s *TransitionSuite) TestTheLastUnansweredAttemptEndsWithNoResponse() {
	settle, again := afterDial(s.sequence(false), s.hours, s.chicago, s.contact(4), 1, Dialed{Result: Unanswered}, s.at(2, 10, 0), s.at(2, 10, 1))
	s.False(again)
	s.Equal(store.ContactCompleted, settle.Stage)
	s.Equal(OutcomeNoResponse, settle.Outcome)
	s.False(settle.Converted)
	s.Nil(settle.NextAttemptAt)
	s.Equal(4, settle.Attempt)
}

func (s *TransitionSuite) TestAnAnsweredCallWaitsForItsOutcome() {
	settle, again := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(2), 2, Dialed{Result: Answered, CallID: "c"}, s.at(2, 10, 0), s.at(2, 10, 1))
	s.False(again)
	s.Equal(store.AwaitingOutcome, settle.Stage)
	s.Equal(2, settle.Attempt)
	s.Nil(settle.NextAttemptAt)
	s.Equal("c", settle.CallID)
}

func (s *TransitionSuite) TestSilenceIsUnansweredWithoutADoubleDial() {
	settle := afterSilence(s.sequence(true), s.hours, s.chicago, s.contact(1), s.at(2, 10, 0), s.at(2, 10, 1))
	s.Equal(store.Enrolled, settle.Stage)
	s.Equal(2, settle.Attempt)
	s.True(s.at(5, 10, 0).Equal(*settle.NextAttemptAt))

	last := afterSilence(s.sequence(true), s.hours, s.chicago, s.contact(4), s.at(2, 10, 0), s.at(2, 10, 1))
	s.Equal(store.ContactCompleted, last.Stage)
	s.Equal(OutcomeNoResponse, last.Outcome)
}

func (s *TransitionSuite) TestDoNotCallEndsUnsuccessfully() {
	settle := afterOutcome(s.sequence(true).First(), OutcomeDoNotCall)
	s.Equal(store.ContactCompleted, settle.Stage)
	s.Equal(OutcomeDoNotCall, settle.Outcome)
	s.False(settle.Converted)
}

func (s *TransitionSuite) TestAnOutcomeEndsAsItsRouteSays() {
	step := s.sequence(true).First()
	won := afterOutcome(step, OutcomeSuccess)
	s.Equal(store.ContactCompleted, won.Stage)
	s.Equal(OutcomeSuccess, won.Outcome)
	s.True(won.Converted)
	lost := afterOutcome(step, OutcomeNotInterested)
	s.False(lost.Converted)
}

func (s *TransitionSuite) TestAnOutcomeNobodyDefinedFollowsOther() {
	step := s.sequence(true).First()
	step.Other = Route{Next: Next{End: EndSuccessful}}
	settle := afterOutcome(step, "made_up_by_the_model")
	s.Equal(store.ContactCompleted, settle.Stage)
	s.Equal(OutcomeOther, settle.Outcome)
	s.True(settle.Converted)
}

func (s *TransitionSuite) TestAConcurrentLimitDefersWithoutUsingTheAttempt() {
	now := s.at(2, 10, 0)
	err := dlc.NewRefusal("concurrent")
	dialed := resultOf(err)
	s.Equal(Deferred, dialed.Result)
	s.Equal("concurrent", dialed.Reason)
	settle, again := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(2), 1, dialed, now, now)
	s.False(again)
	s.Equal(store.Enrolled, settle.Stage)
	s.Equal(2, settle.Attempt)
	s.True(now.Add(5 * time.Minute).Equal(*settle.NextAttemptAt))

	late := s.at(2, 17, 58)
	settle, _ = afterDial(s.sequence(true), s.hours, s.chicago, s.contact(2), 1, dialed, late, late)
	s.True(s.at(3, 9, 0).Equal(*settle.NextAttemptAt), "five minutes later is after closing")
}

func (s *TransitionSuite) TestADailyLimitDefersToTomorrowsFirstOpening() {
	now := s.at(2, 14, 0)
	dialed := resultOf(dlc.NewRefusal("per_day"))
	s.Equal(Deferred, dialed.Result)
	settle, _ := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(1), 1, dialed, now, now)
	s.Equal(store.Enrolled, settle.Stage)
	s.Equal(1, settle.Attempt)
	s.True(s.at(3, 9, 0).Equal(*settle.NextAttemptAt), settle.NextAttemptAt.String())

	fri := s.at(6, 14, 0)
	settle, _ = afterDial(s.sequence(true), s.hours, s.chicago, s.contact(1), 1, dialed, fri, fri)
	s.True(s.at(9, 9, 0).Equal(*settle.NextAttemptAt), "tomorrow is Saturday, so Monday")
}

func (s *TransitionSuite) TestARecipientTheSandboxRefusesFails() {
	dialed := resultOf(dlc.NewRefusal("sandbox_recipient"))
	s.Equal(Refused, dialed.Result)
	settle, again := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(1), 1, dialed, s.at(2, 10, 0), s.at(2, 10, 0))
	s.False(again)
	s.Equal(store.Failed, settle.Stage)
	s.Equal("sandbox_recipient", settle.Reason)
}

func (s *TransitionSuite) TestAnOptedOutNumberFailsAsOptedOut() {
	dialed := resultOf(dlc.NewRefusal("opted_out"))
	settle, _ := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(1), 1, dialed, s.at(2, 10, 0), s.at(2, 10, 0))
	s.Equal(store.Failed, settle.Stage)
	s.Equal("opted_out", settle.Reason)
}

func (s *TransitionSuite) TestAnInvalidNumberFails() {
	dialed := resultOf(phone.Dialable("12345"))
	s.Equal(InvalidNumber, dialed.Result)
	settle, again := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(1), 1, dialed, s.at(2, 10, 0), s.at(2, 10, 0))
	s.False(again)
	s.Equal(store.Failed, settle.Stage)
	s.Equal("invalid_number", settle.Reason)
}

func (s *TransitionSuite) TestARefusalOfMoneyPausesAndKeepsTheAttempt() {
	now := s.at(2, 10, 0)
	for err, code := range map[error]string{
		phone.ErrSuspended: "telephony_suspended",
		dlc.ErrUnpaid:      "insufficient_credits",
	} {
		dialed := resultOf(stack.Wrap(err))
		s.Equal(Paused, dialed.Result)
		s.Equal(code, dialed.Reason)
		settle, again := afterDial(s.sequence(true), s.hours, s.chicago, s.contact(3), 1, dialed, now, now)
		s.False(again)
		s.Equal(store.Enrolled, settle.Stage)
		s.Equal(3, settle.Attempt)
		s.True(now.Equal(*settle.NextAttemptAt))
	}
}

func (s *TransitionSuite) TestAWrappedErrorIsRecognisedAsItsKind() {
	wrapped := fmt.Errorf("dial: %w", stack.Wrap(dlc.NewRefusal("country")))
	s.Equal(Refused, resultOf(wrapped).Result)
	s.Equal("country", resultOf(wrapped).Reason)
	s.Equal(InvalidNumber, resultOf(fmt.Errorf("x: %w", stack.Wrap(phone.Dialable("+19005550123")))).Result)
}

func (s *TransitionSuite) TestAnyOtherDialErrorIsUnanswered() {
	s.Equal(Unanswered, resultOf(errors.New("vendor hung up")).Result)
}

func (s *TransitionSuite) TestAContactWithNoOpeningFailsInsteadOfGettingNoTime() {
	hours := s.hours
	hours.Windows = map[string][]Window{"monday": {{Start: "09:00", End: "18:00"}}}
	hours.ExcludedDates = nil
	for d := 0; d < 420; d++ {
		hours.ExcludedDates = append(hours.ExcludedDates, s.at(2, 0, 0).AddDate(0, 0, d).Format("2006-01-02"))
	}
	settle, _ := afterDial(s.sequence(false), hours, s.chicago, s.contact(1), 1, Dialed{Result: Unanswered}, s.at(2, 10, 0), s.at(2, 10, 0))
	s.Equal(store.Failed, settle.Stage)
	s.Equal("no_calling_hours", settle.Reason)
	s.Nil(settle.NextAttemptAt)
}
