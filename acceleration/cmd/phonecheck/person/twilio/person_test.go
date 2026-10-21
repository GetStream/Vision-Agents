package twilio

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

const (
	agentNumber  = "+15550100001"
	personNumber = "+15550100002"
)

type TwilioSuite struct {
	suite.Suite
	fake   *fakeTwilio
	person person.Person
	logs   *bytes.Buffer
}

func TestTwilioSuite(t *testing.T) { suite.Run(t, new(TwilioSuite)) }

func (s *TwilioSuite) SetupTest() {
	s.fake = newFakeTwilio()
	s.T().Cleanup(s.fake.server.Close)
	s.T().Setenv("TWILIO_ACCOUNT_SID", fakeSID)
	s.T().Setenv("TWILIO_AUTH_TOKEN", fakeToken)

	oldBase, oldEvery, oldMargin := apiBase, pollEvery, waitMargin
	apiBase, pollEvery, waitMargin = s.fake.server.URL, time.Millisecond, time.Second
	s.T().Cleanup(func() { apiBase, pollEvery, waitMargin = oldBase, oldEvery, oldMargin })

	s.person = New()
	s.logs = &bytes.Buffer{}
}

func (s *TwilioSuite) call(dir person.Direction) person.Call {
	logger := slog.New(slog.NewTextHandler(s.logs, nil))
	return person.Call{
		ID: "pc-test", Direction: dir, PersonNumber: personNumber, AgentNumber: agentNumber,
		Talk: 20 * time.Second, RegionLabel: "ashburn", Started: time.Now().Add(-time.Minute), Logger: logger,
	}
}

func (s *TwilioSuite) statusLines() int { return strings.Count(s.logs.String(), "status=") }

func (s *TwilioSuite) TestDialingPlacesTheCallWithTheScriptInline() {
	err := s.person.Dial(context.Background(), s.call(person.Inbound))

	s.Require().NoError(err)
	s.Require().Len(s.fake.placed, 1)
	form := s.fake.placed[0]
	s.Equal(personNumber, form.Get("From"))
	s.Equal(agentNumber, form.Get("To"))
	twiml := form.Get("Twiml")
	s.True(strings.HasPrefix(twiml, "<Response><Pause"), twiml)
	s.Contains(twiml, "This is a phonecheck test call.")
	s.Contains(twiml, `<Pause length="20"/>`)
	s.Contains(twiml, "<Hangup/>")
	s.NotContains(twiml, "<Dial")
}

func (s *TwilioSuite) TestWaitLogsTheCallUntilItEnds() {
	s.fake.placedStatuses = []string{"ringing", "in-progress", "completed"}
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))

	err := s.person.Wait(context.Background(), call)

	s.Require().NoError(err)
	s.Equal(3, s.statusLines(), s.logs.String())
	for _, want := range []string{"status=ringing", "status=in-progress", "status=completed", "region_label=ashburn"} {
		s.Contains(s.logs.String(), want)
	}
}

func (s *TwilioSuite) TestAnOutboundWaitFindsTheCallToThePersonNumber() {
	call := s.call(person.Outbound)
	s.fake.add("CAolder", call.Started.Add(-10*time.Second), "completed")
	s.fake.add("CAnewer", call.Started.Add(5*time.Second), "in-progress", "completed")
	s.fake.list("CAolder", "CAnewer")

	err := s.person.Wait(context.Background(), call)

	s.Require().NoError(err)
	s.Contains(s.fake.paths(), "GET /2010-04-01/Accounts/"+fakeSID+"/Calls/CAnewer.json")
	s.NotContains(s.fake.paths(), "GET /2010-04-01/Accounts/"+fakeSID+"/Calls/CAolder.json")
	s.Equal(personNumber, s.fake.queries[0].Get("To"))
	s.NotEmpty(s.fake.queries[0].Get("StartTime>="))
}

func (s *TwilioSuite) TestAnOutboundCallThatNeverShowsUpIsNotObserved() {
	waitMargin = 50 * time.Millisecond
	call := s.call(person.Outbound)
	call.Talk = 0

	err := s.person.Wait(context.Background(), call)

	s.Require().NoError(err)
	s.Equal(1, strings.Count(s.logs.String(), "person call not observed"), s.logs.String())
}

func (s *TwilioSuite) TestACallThatNeverEndsStopsBeingWaitedOn() {
	waitMargin = 50 * time.Millisecond
	s.fake.placedStatuses = []string{"in-progress"}
	call := s.call(person.Inbound)
	call.Talk = 0
	s.Require().NoError(s.person.Dial(context.Background(), call))

	err := s.person.Wait(context.Background(), call)

	s.ErrorContains(err, "did not end")
}

func (s *TwilioSuite) TestHangupCompletesEveryCallStillLive() {
	s.fake.placedStatuses = []string{"in-progress"}
	s.Require().NoError(s.person.Dial(context.Background(), s.call(person.Inbound)))

	s.Require().NoError(s.person.Hangup(context.Background()))

	s.Equal([]string{"CAplaced"}, s.fake.completedSIDs())
}

func (s *TwilioSuite) TestHangupLeavesAnEndedCallAlone() {
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))
	s.Require().NoError(s.person.Wait(context.Background(), call))

	s.Require().NoError(s.person.Hangup(context.Background()))

	s.Empty(s.fake.completedSIDs())
}

func (s *TwilioSuite) TestHangupAlsoEndsALiveCallItFound() {
	call := s.call(person.Outbound)
	s.fake.add("CAfound", call.Started.Add(time.Second), "in-progress")
	s.fake.add("CAdone", call.Started.Add(time.Second), "completed")
	s.fake.list("CAfound", "CAdone")
	waitMargin = 50 * time.Millisecond
	call.Talk = 0
	_ = s.person.Wait(context.Background(), call) // follows the found call until the margin runs out

	s.Require().NoError(s.person.Hangup(context.Background()))

	s.Equal([]string{"CAfound"}, s.fake.completedSIDs())
}

func (s *TwilioSuite) TestNothingButCallsIsSent() {
	s.fake.placedStatuses = []string{"in-progress", "completed"}
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))
	s.Require().NoError(s.person.Wait(context.Background(), call))
	s.Require().NoError(s.person.Hangup(context.Background()))

	for _, request := range s.fake.paths() {
		s.Contains(request, "/Accounts/"+fakeSID+"/Calls", request)
		s.NotContains(request, "IncomingPhoneNumbers")
	}
}

func (s *TwilioSuite) TestNothingLoggedCarriesTheAuthToken() {
	s.fake.placedStatuses = []string{"ringing", "completed"}
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))
	s.Require().NoError(s.person.Wait(context.Background(), call))
	s.fake.failWith = 500
	errs := []error{
		s.person.Dial(context.Background(), call),
		s.person.Wait(context.Background(), call),
		s.person.Hangup(context.Background()),
	}

	s.NotContains(s.logs.String(), fakeToken)
	for _, err := range errs {
		s.Require().Error(err)
		s.NotContains(err.Error(), fakeToken)
	}
}

func (s *TwilioSuite) TestMissingCredentialsAreNamed() {
	s.T().Setenv("TWILIO_AUTH_TOKEN", "")

	err := s.person.Dial(context.Background(), s.call(person.Inbound))

	s.ErrorContains(err, "TWILIO_AUTH_TOKEN")
}
