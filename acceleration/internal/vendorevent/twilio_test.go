package vendorevent

import (
	"crypto/rand"
	"testing"

	"github.com/stretchr/testify/suite"
)

// TwilioSuite is the name and level each Twilio Monitor alert and usage trigger is counted
// under.
type TwilioSuite struct {
	suite.Suite
}

func TestTwilioSuite(t *testing.T) {
	suite.Run(t, new(TwilioSuite))
}

// An inactive or suspended account sends nothing, so each pages.
func (s *TwilioSuite) TestAnInactiveOrSuspendedAccountPages() {
	for _, code := range []string{"10001", "30002"} {
		event, level := TwilioAlert(code)
		s.Equal("error."+code, event)
		s.Equal(LevelPage, level, code)
	}
}

// Refused credentials and a blocked caller ID lose some requests and calls, not all of them.
func (s *TwilioSuite) TestRefusedCredentialsAndABlockedCallerIDWarn() {
	for _, code := range []string{"20003", "32017"} {
		event, level := TwilioAlert(code)
		s.Equal("error."+code, event)
		s.Equal(LevelWarn, level, code)
	}
}

// A code the map does not name is one label value, whatever Twilio sent, so a new code or a
// near miss of a listed one adds no series and pages nobody.
func (s *TwilioSuite) TestACodeTheMapDoesNotNameIsCountedAsErrorOther() {
	for _, code := range []string{"", "11200", "1000", "100010", " 10001", "10001 ", "010001", "10001.0",
		"other", rand.Text()} {
		event, level := TwilioAlert(code)
		s.Equal("error.other", event, code)
		s.Equal(LevelInfo, level, code)
	}
}

func (s *TwilioSuite) TestATotalPriceTriggerPages() {
	event, level := TwilioUsageTrigger("totalprice")
	s.Equal("usage_trigger.totalprice", event)
	s.Equal(LevelPage, level)
}

func (s *TwilioSuite) TestACallsTriggerWarns() {
	event, level := TwilioUsageTrigger("calls")
	s.Equal("usage_trigger.calls", event)
	s.Equal(LevelWarn, level)
}

// Somebody set every trigger to be told about it, so one the map does not name still warns,
// under one label value. Twilio's categories are lower case, so another case is another
// category.
func (s *TwilioSuite) TestACategoryTheMapDoesNotNameIsCountedAsUsageTriggerOtherAndWarns() {
	for _, category := range []string{"", "sms", "TotalPrice", "totalprice ", "calls-sip", "other", rand.Text()} {
		event, level := TwilioUsageTrigger(category)
		s.Equal("usage_trigger.other", event, category)
		s.Equal(LevelWarn, level, category)
	}
}
