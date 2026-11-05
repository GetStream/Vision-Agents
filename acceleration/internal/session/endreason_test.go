package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
)

// EndReasonSuite holds the outcome package's own copy of the end reasons that end a call to
// these: a reason renamed here and not there would count every such call as dropped.
type EndReasonSuite struct{ suite.Suite }

func TestEndReasonSuite(t *testing.T) { suite.Run(t, new(EndReasonSuite)) }

func (s *EndReasonSuite) TestTheReasonsThatEndACallAreTheOnesTheOutcomeCounterKnows() {
	for _, reason := range []EndReason{EndAgentHangup, EndTransferred, EndVoicemail, TimeoutHangup} {
		s.True(outcome.EndsTheCall(string(reason)), reason)
	}
	for _, reason := range []EndReason{TimeoutStart, TimeoutUnanswered, ""} {
		s.False(outcome.EndsTheCall(string(reason)), reason)
	}
}
