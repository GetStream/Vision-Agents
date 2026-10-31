package phone

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
)

type ServiceSuite struct {
	suite.Suite
}

func TestService(t *testing.T) { suite.Run(t, new(ServiceSuite)) }

func (s *ServiceSuite) TestEachCallResultIsAnOutcome() {
	for result, want := range map[CallResult]outcome.Outcome{
		CallCompleted: outcome.Completed,
		CallBusy:      outcome.Busy,
		CallNoAnswer:  outcome.NoAnswer,
		CallFailed:    outcome.Failed,
		CallCanceled:  outcome.Canceled,
	} {
		got, err := outcomeOf(result)
		s.Require().NoError(err)
		s.Equal(want, got, string(result))
	}
}

func (s *ServiceSuite) TestANonFinalResultIsNoOutcome() {
	got, err := outcomeOf("")
	s.Require().NoError(err)

	s.Empty(got)
}

func (s *ServiceSuite) TestAnUnknownResultIsRefused() {
	_, err := outcomeOf(CallResult("voicemail"))

	s.ErrorContains(err, "voicemail")
}
