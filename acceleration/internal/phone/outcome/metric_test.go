package outcome

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DroppedSuite is the rule that tells a dropped call from a completed one: the session of an
// answered call, as it was when the call's leg ended.
type DroppedSuite struct {
	suite.Suite
	ended time.Time
}

func TestDroppedSuite(t *testing.T) { suite.Run(t, new(DroppedSuite)) }

func (s *DroppedSuite) SetupTest() { s.ended = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC) }

// closed is a session that closed d after the leg ended (before it, when d is negative).
func (s *DroppedSuite) closed(d time.Duration, reason string) store.AgentSession {
	return store.AgentSession{ClosedAt: new(s.ended.Add(d)), EndReason: reason}
}

func (s *DroppedSuite) TestASessionStillOpenWhenThePhoneLeftDidNotDropTheCall() {
	s.False(leftOnTheLine(store.AgentSession{}, s.ended), "the caller or the network hung up first")
}

func (s *DroppedSuite) TestASessionStoppedBeforeThePhoneLeftDroppedTheCall() {
	s.True(leftOnTheLine(s.closed(-time.Second, ""), s.ended))
}

func (s *DroppedSuite) TestASessionStoppedAsThePhoneLeftDroppedTheCall() {
	s.True(leftOnTheLine(s.closed(0, ""), s.ended))
}

func (s *DroppedSuite) TestASessionThatClosedAfterThePhoneLeftDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(time.Millisecond, ""), s.ended))
}

func (s *DroppedSuite) TestAnAgentThatHungUpDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "agent_hangup"), s.ended))
}

func (s *DroppedSuite) TestAnAgentThatTransferredDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "transferred"), s.ended))
}

func (s *DroppedSuite) TestAnAgentThatLeftAVoicemailDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "voicemail"), s.ended))
}

func (s *DroppedSuite) TestASessionThatFollowedTheCallersHangUpDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "hangup"), s.ended))
}

func (s *DroppedSuite) TestASessionThatGaveUpWaitingBeforeThePhoneJoinedDroppedTheCall() {
	s.True(leftOnTheLine(s.closed(-time.Second, "start"), s.ended), "the caller answered an agent already gone")
	s.True(leftOnTheLine(s.closed(-time.Second, "unanswered"), s.ended))
}

// CarrierSuite is how a vendor's report of a call's carrier is labelled and when it counts.
type CarrierSuite struct {
	suite.Suite
	ended time.Time
}

func TestCarrierSuite(t *testing.T) { suite.Run(t, new(CarrierSuite)) }

func (s *CarrierSuite) SetupTest() { s.ended = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC) }

func (s *CarrierSuite) TestAResponseCodeOfRFC3261IsItsOwnLabel() {
	for code, label := range map[int]string{200: "200", 404: "404", 486: "486", 487: "487", 503: "503", 603: "603"} {
		got, ok := sipCodeLabel(code)
		s.True(ok, code)
		s.Equal(label, got)
	}
}

func (s *CarrierSuite) TestAnyOtherFinalCodeIsCountedAsItsClass() {
	for code, label := range map[int]string{299: "2xx", 399: "3xx", 429: "4xx", 599: "5xx", 607: "6xx", 699: "6xx"} {
		got, ok := sipCodeLabel(code)
		s.True(ok, code)
		s.Equal(label, got)
	}
}

func (s *CarrierSuite) TestACodeThatIsNoFinalResponseIsNotCounted() {
	for _, code := range []int{0, -486, 100, 180, 199, 700, 4860} {
		_, ok := sipCodeLabel(code)
		s.False(ok, code)
	}
}

func (s *CarrierSuite) TestACallAnsweredForUnderSixSecondsIsShort() {
	s.True(isShort(0))
	s.True(isShort(5999 * time.Millisecond))
	s.False(isShort(6 * time.Second))
	s.False(isShort(-time.Second), "a join dated after the end")
}

func (s *CarrierSuite) TestAnUndecidedCallHasNoFinalReport() {
	s.False(finalReportKept(Call{}))
}

func (s *CarrierSuite) TestAnOutcomeDecidedWhileTheLegIsUpCameFromAReport() {
	s.True(finalReportKept(Call{Outcome: Busy, DecidedAt: new(s.ended)}))
}

func (s *CarrierSuite) TestAnOutcomeDecidedBeforeTheEndCameFromAReport() {
	s.True(finalReportKept(Call{Outcome: Busy, DecidedAt: new(s.ended.Add(-time.Second)), EndedAt: new(s.ended)}))
}

func (s *CarrierSuite) TestAnOutcomeDecidedAtTheEndCameFromTheEnd() {
	s.False(finalReportKept(Call{Outcome: NoAnswer, DecidedAt: new(s.ended), EndedAt: new(s.ended)}))
	s.False(finalReportKept(Call{Outcome: Completed, DecidedAt: new(s.ended.Add(time.Second)), EndedAt: new(s.ended)}),
		"a join after the end")
}
