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
