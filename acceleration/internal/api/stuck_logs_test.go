//go:build integration

package api

import (
	"context"
	"net/http"
	"regexp"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A log line where a customer's caller or writer gets stuck names the customer, so the line
// in the log store says whose number it was.

// loggedOneLine is the pattern of one log line holding level and msg, then customer.
func loggedOneLine(level, msg, customer string) *regexp.Regexp {
	return regexp.MustCompile(`level=` + level + ` msg="` + regexp.QuoteMeta(msg) + `"[^\n]* customer=` +
		regexp.QuoteMeta(customer) + `( |\n)`)
}

// Defect: the router rejects a call to a number of an app whose telephony is suspended and
// logs it at Info with only the number, so nobody can tell whose number was called.
func (s *TelnyxHookSuite) TestARejectedCallToASuspendedAppWarnsNamingTheApp() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@example.test",
	}))

	status, _, body := s.deliver(s.call(s.e164), true)
	s.Require().Equal(http.StatusOK, status)
	s.Require().Contains(body, "<Reject")

	s.Regexp(loggedOneLine("WARN", "rejected a telnyx call to a number of an app whose telephony is suspended",
		s.customerID()), s.thisTest())
}

// Defect: a call to a number of an organization that may not pay is rejected and logged at Info
// without the app.
func (s *AdmissionSuite) TestATelnyxCallRefusedForMoneyWarnsNamingTheApp() {
	e164 := s.utils.number()
	s.line.attach(e164, "sip:"+e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	status, _, body := s.line.deliver(s.line.call(e164), true)
	s.Require().Equal(http.StatusOK, status)
	s.Require().Contains(body, "<Reject")

	s.Regexp(loggedOneLine("WARN", "rejected a telnyx call to a number of an organization that may not pay for it",
		s.customerID()), s.logged.String())
}

// Defect: a vendor fetching a placed call's plan that cannot be served logs the failure
// without the app whose call it is.
func (s *CallStatusSuite) TestAPlanThatCannotBeServedIsLoggedNamingTheCallsApp() {
	s.useApp(s.data.createApp())
	token := s.utils.uuid()
	s.Require().NoError(s.store.ParkBridge(context.Background(), &store.CallBridge{
		Token: token, CustomerID: s.customerID(), Vendor: "twilio", TrunkURI: "sip:trunk@trunk.sip.example.test",
		CallID: s.utils.uuid(), ExpiresAt: time.Now().Add(time.Minute),
	}))

	// Twilio takes its plan on the request that places the call, so it has none to serve.
	s.Require().Equal(http.StatusNotFound, s.unauthenticatedClient.do(http.MethodGet, "/v1/phone/answer/"+token, nil, nil))

	s.Regexp(loggedOneLine("ERROR", "could not answer a placed call", s.customerID()), s.logged.String())
}

// Defect: a message on a line no agent answers on is logged with only the line, so the app
// that should name it in an agent's config is not known.
func (s *ChannelsSuite) TestAMessageOnALineNoAgentAnswersOnWarnsNamingTheApp() {
	line := s.connect(s.number())

	s.Require().Equal(http.StatusAccepted, s.deliver(line, s.writer(), "msg-"+s.utils.uuid(), "what is open?"))

	pattern := loggedOneLine("WARN", "a message arrived on a line no agent answers on", s.customerID())
	s.Require().Eventually(func() bool { return pattern.MatchString(s.logged.String()) },
		settleFor, 10*time.Millisecond, "the router never warned naming the app")
}

// Defect: a call that rang an app's number, which no attached number reaches, logged at Debug,
// which production does not keep.
func (s *CallHooksSuite) TestACallThatRangANumberNoNumberReachesIsWarnedAbout() {
	s.Require().Equal(http.StatusOK, s.arriveRingingNumber())

	s.True(s.loggedLine("level=WARN", `msg="no number reaches an arriving call"`, "stream_app="))
}

// Defect: every video call in the app, which arrives here too and no number is meant to reach,
// warned about as a caller stuck.
func (s *CallHooksSuite) TestAVideoCallNoNumberReachesIsNotWarnedAbout() {
	callID := "video-" + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+callID))

	s.True(s.loggedLine("level=DEBUG", `msg="no number reaches an arriving call"`, callID))
	s.False(s.loggedLine("level=WARN", `msg="no number reaches an arriving call"`))
}

// Defect: a call named for a number, which no number reaches, logged at Debug as a video call is.
func (s *CallHooksSuite) TestACallNamedForANumberNoNumberReachesIsWarnedAbout() {
	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	s.True(s.loggedLine("level=WARN", `msg="no number reaches an arriving call"`, s.loggedAs(s.callID)))
}
