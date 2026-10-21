package telnyx

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

type TelnyxSuite struct {
	suite.Suite
	fake   *fakeTelnyx
	person person.Person
	logs   *bytes.Buffer
}

func TestTelnyxSuite(t *testing.T) { suite.Run(t, new(TelnyxSuite)) }

func (s *TelnyxSuite) SetupTest() {
	s.fake = newFakeTelnyx()
	s.T().Cleanup(s.fake.server.Close)
	s.T().Setenv("TELNYX_API_KEY", fakeKey)
	s.T().Setenv("TELNYX_ACCOUNT_SID", fakeAccount)
	s.T().Setenv("TELNYX_PERSON_CONNECTION_ID", fakeAppID)

	oldBase, oldEvery, oldWait, oldObserve := apiBase, pollEvery, waitMargin, observeMargin
	apiBase, pollEvery, waitMargin = s.fake.server.URL, time.Millisecond, time.Second
	s.T().Cleanup(func() { apiBase, pollEvery, waitMargin, observeMargin = oldBase, oldEvery, oldWait, oldObserve })

	s.person = New()
	s.logs = &bytes.Buffer{}
}

func (s *TelnyxSuite) call(dir person.Direction) person.Call {
	logger := slog.New(slog.NewTextHandler(s.logs, nil))
	return person.Call{
		ID: "pc-test", Direction: dir, PersonNumber: personNumber, AgentNumber: agentNumber,
		Talk: 20 * time.Second, RegionLabel: "ashburn", Started: time.Now().Add(-time.Minute), Logger: logger,
	}
}

func (s *TelnyxSuite) TestDialingPlacesTheCallThroughThePersonApp() {
	err := s.person.Dial(context.Background(), s.call(person.Inbound))

	s.Require().NoError(err)
	s.Equal([]string{"POST " + s.fake.callsPath}, s.fake.paths())
	s.Require().Len(s.fake.placed, 1)
	body := s.fake.placed[0]
	s.Equal(fakeAppID, body["ApplicationSid"])
	s.Equal(personNumber, body["From"])
	s.Equal(agentNumber, body["To"])
	s.Contains(body["Texml"], "This is a phonecheck test call.")
	s.Contains(body["Texml"], `<Pause length="20"/>`)
	s.Contains(body["Texml"], "<Hangup/>")
	s.NotContains(body["Texml"], "<Dial")
}

func (s *TelnyxSuite) TestDialingNeedsThePersonConnectionID() {
	s.T().Setenv("TELNYX_PERSON_CONNECTION_ID", "")

	err := s.person.Dial(context.Background(), s.call(person.Inbound))

	s.ErrorContains(err, "TELNYX_PERSON_CONNECTION_ID")
	s.Empty(s.fake.paths())
}

func (s *TelnyxSuite) TestMissingCredentialsAreNamed() {
	for _, name := range []string{"TELNYX_API_KEY", "TELNYX_ACCOUNT_SID"} {
		s.Run(name, func() {
			s.T().Setenv(name, "")

			err := New().Dial(context.Background(), s.call(person.Inbound))

			s.ErrorContains(err, name)
		})
	}
}

func (s *TelnyxSuite) TestWaitLogsTheInboundCallUntilItEnds() {
	s.fake.setStatuses("ringing", "in-progress", "completed")
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))

	err := s.person.Wait(context.Background(), call)

	s.Require().NoError(err)
	s.Equal(3, strings.Count(s.logs.String(), "status="), s.logs.String())
	for _, want := range []string{"status=ringing", "status=completed", "region_label=ashburn",
		"start=2026-10-21T09:00:00Z", "end=2026-10-21T09:00:25Z"} {
		s.Contains(s.logs.String(), want)
	}
}

func (s *TelnyxSuite) TestACallThatNeverEndsStopsBeingWaitedOn() {
	waitMargin = 50 * time.Millisecond
	s.fake.setStatuses("in-progress")
	call := s.call(person.Inbound)
	call.Talk = 0
	s.Require().NoError(s.person.Dial(context.Background(), call))

	err := s.person.Wait(context.Background(), call)

	s.ErrorContains(err, "did not end")
}

func (s *TelnyxSuite) TestAnOutboundPersonCallIsNotObserved() {
	observeMargin = 30 * time.Millisecond
	call := s.call(person.Outbound)
	call.Talk = 10 * time.Millisecond

	start := time.Now()
	err := s.person.Wait(context.Background(), call)

	s.Require().NoError(err)
	s.GreaterOrEqual(time.Since(start), 40*time.Millisecond)
	s.Equal(1, strings.Count(s.logs.String(), "person call not observed"), s.logs.String())
	s.Require().NoError(s.person.Hangup(context.Background(), call))
	s.Empty(s.fake.paths())
}

func (s *TelnyxSuite) TestAnOutboundWaitStopsWhenTheRunIsCancelled() {
	call := s.call(person.Outbound)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	err := s.person.Wait(ctx, call)

	s.ErrorIs(err, context.DeadlineExceeded)
}

func (s *TelnyxSuite) TestHangupEndsTheLiveCall() {
	s.fake.setStatuses("in-progress")
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))

	s.Require().NoError(s.person.Hangup(context.Background(), call))

	s.Require().Len(s.fake.hangups(), 1)
	s.Equal("completed", s.fake.hangups()[0].Get("Status"))
	s.Contains(s.fake.paths(), "POST "+s.fake.callsPath+"/"+fakeSID)
}

func (s *TelnyxSuite) TestHangupLeavesAnEndedCallAlone() {
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))
	s.Require().NoError(s.person.Wait(context.Background(), call))

	s.Require().NoError(s.person.Hangup(context.Background(), call))

	s.Empty(s.fake.hangups())
}

func (s *TelnyxSuite) TestNothingButCallsIsSent() {
	s.fake.setStatuses("in-progress", "completed")
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))
	s.Require().NoError(s.person.Wait(context.Background(), call))
	s.Require().NoError(s.person.Hangup(context.Background(), call))

	s.NotEmpty(s.fake.paths())
	for _, request := range s.fake.paths() {
		s.Contains(request, "/v2/texml/Accounts/"+fakeAccount+"/Calls", request)
		s.NotContains(request, "phone_numbers")
		s.NotContains(request, "texml_applications")
	}
}

func (s *TelnyxSuite) TestNothingLoggedCarriesTheAPIKey() {
	s.fake.setStatuses("ringing", "completed")
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))
	s.Require().NoError(s.person.Wait(context.Background(), call))
	s.fake.failWith = 500
	errs := []error{
		s.person.Dial(context.Background(), call),
		s.person.Wait(context.Background(), call),
	}

	s.NotContains(s.logs.String(), fakeKey)
	for _, err := range errs {
		s.Require().Error(err)
		s.NotContains(err.Error(), fakeKey)
	}
}

func (s *TelnyxSuite) TestWaitKeepsPollingWhileAJustPlacedCallIsNotYetReadable() {
	s.fake.notFound = 3
	s.fake.setStatuses("in-progress", "completed")
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))

	err := s.person.Wait(context.Background(), call)

	s.Require().NoError(err)
	s.Contains(s.logs.String(), "status=in-progress")
	s.Contains(s.logs.String(), "status=completed")
}

func (s *TelnyxSuite) TestACallThatNeverBecomesReadableStopsBeingWaitedOn() {
	waitMargin = 50 * time.Millisecond
	s.fake.notFound = 1 << 20
	call := s.call(person.Inbound)
	call.Talk = 0
	s.Require().NoError(s.person.Dial(context.Background(), call))

	err := s.person.Wait(context.Background(), call)

	s.ErrorContains(err, "did not end")
}

func (s *TelnyxSuite) TestAPlacedCallWithNoSIDIsAnError() {
	s.fake.noSID = true

	err := s.person.Dial(context.Background(), s.call(person.Inbound))

	s.ErrorContains(err, "no call sid")
}

func (s *TelnyxSuite) TestAnInboundWaitWithNoPlacedCallIsAnError() {
	err := s.person.Wait(context.Background(), s.call(person.Inbound))

	s.ErrorContains(err, "no call was placed")
	s.Empty(s.fake.paths())
}

func (s *TelnyxSuite) TestATimeThatCannotBeReadIsLeftEmptyAndWarnedAbout() {
	s.fake.badTimes = true
	call := s.call(person.Inbound)
	s.Require().NoError(s.person.Dial(context.Background(), call))

	s.Require().NoError(s.person.Wait(context.Background(), call))

	s.Contains(s.logs.String(), "could not read a time of the person's call")
	s.Contains(s.logs.String(), "value=yesterday")
	s.Contains(s.logs.String(), "status=completed start=\"\"")
}

func (s *TelnyxSuite) TestReadyNamesWhatInboundIsMissing() {
	s.Require().NoError(s.person.Ready(person.Inbound))
	for _, name := range []string{"TELNYX_API_KEY", "TELNYX_ACCOUNT_SID", "TELNYX_PERSON_CONNECTION_ID"} {
		s.Run(name, func() {
			s.T().Setenv(name, "")

			s.ErrorContains(New().Ready(person.Inbound), name)
		})
	}
	s.Empty(s.fake.paths())
}

func (s *TelnyxSuite) TestReadyNeedsNothingForOutbound() {
	s.T().Setenv("TELNYX_API_KEY", "")

	s.NoError(s.person.Ready(person.Outbound))
}
