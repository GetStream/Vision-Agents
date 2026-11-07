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
	// Whether each reason ends its call, for every reason in EndReasons: a reason added there
	// fails here until it is decided, rather than counting its calls as dropped unseen.
	endsTheCall := map[EndReason]bool{
		EndAgentHangup: true, EndTransferred: true, EndVoicemail: true, TimeoutHangup: true,
		TimeoutStart: false, TimeoutUnanswered: false, EndClosed: false, EndShutdown: false,
		EndCreateFailed: false, EndCreateVendorFailed: false, EndCreateConfigRefused: false,
		EndCreateNoCredit: false, EndCreateNoVideo: false,
		EndAgentFailed: false, EndModelFailed: false, EndModelConfigRefused: false, EndModelNoCredit: false,
	}
	for _, info := range EndReasons {
		ends, decided := endsTheCall[info.Reason]
		s.True(decided, "%s is not decided here", info.Reason)
		s.Equal(ends, outcome.EndsTheCall(string(info.Reason)), info.Reason)
	}
	s.Len(endsTheCall, len(EndReasons), "a reason decided here that EndReasons does not have")
	s.False(outcome.EndsTheCall(""), "a session that ended before its reason was kept")
}
