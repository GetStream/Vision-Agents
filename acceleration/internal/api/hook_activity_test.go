//go:build integration

package api

import (
	"context"
	"net/http"
	"net/url"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A hook carries no credential, so it names no client: each is counted as the API's.

// Defect: a hook Stream signed, which no credential names, left out of the activity record, or
// counted with no organization, so the app's organization shows no hooks in its digest.
func (s *AppHooksSuite) TestAHookTheGatewayVouchedForIsCountedUnderItsAppAndOrganization() {
	s.Require().NoError(s.store.JoinOrganization(context.Background(), s.customerID(), s.app.organization.ID))

	s.Require().Equal(http.StatusOK, s.fromApp(s.appPath(chat.MessageHookPath), s.message(s.config(s.customerID()))))

	row := s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
	s.Equal(int64(0), row.refused)
	s.Equal(s.app.organization.ID, row.org)
}

// Defect: a hook counted before its signature was checked, so anyone posting to the path
// writes rows under whichever app it names. The good hook after it is the latest a bad one's
// row could have been written beside.
func (s *AppHooksSuite) TestAHookWithABadSignatureIsNotCounted() {
	s.Require().Equal(http.StatusUnauthorized, s.hook(s.appPath(chat.MessageHookPath),
		s.message(s.config(s.customerID())), "not the app's secret", s.apiKey))
	s.Require().Equal(http.StatusOK, s.fromApp(s.appPath(chat.MessageHookPath), s.message(s.config(s.customerID()))))

	s.Equal(int64(0), s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI).refused)
}

// Defect: a hook carried to the node with the worker counted by both nodes, each having
// checked its signature.
func (s *AppHooksSuite) TestAHookCarriedToAnotherNodeIsCountedOnce() {
	other := s.otherNodeDispatching(dispatch.NewPool())
	worker := s.serverClient.on(other).opens("/v1/dispatch")
	s.await(worker, "ready")

	s.Require().Equal(http.StatusOK, s.fromApp(s.appPath(chat.MessageHookPath), s.message(s.config(s.customerID()))))
	s.await(worker, "message")

	s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
	s.Never(func() bool { return s.counted(activity.AreaHooksIn, store.AuditSourceAPI).requests > 1 },
		10*suiteActivityInterval, suiteActivityInterval, "the carried hook was counted again")
}

// Defect: a call event from the customer's own app, which may name no number, left uncounted.
func (s *AppHooksSuite) TestACallHookTheGatewayVouchedForIsCounted() {
	s.Require().Equal(http.StatusOK, s.fromApp(s.appPath(phone.CallHookPath), `{"type": "call.created"}`))

	s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
}

// Defect: a Stream Chat delivery to a customer's own app left uncounted.
func (s *ChatInboundAppSuite) TestADeliveryIsCountedUnderItsApp() {
	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(s.appID(), s.config(s.customerID()), ""), s.current))

	s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
}

// Defect: in deployment mode, where every hook is the deployment's own app's, a call to a
// customer's number counted under nobody.
func (s *CallHooksSuite) TestACallToAHeldNumberIsCountedUnderTheNumbersApp() {
	s.hold("default", s.callID)

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
}

// Defect: a call Telnyx signed counted as a hook in general rather than as the app's phone
// work, or an unsigned one counted at all.
func (s *TelnyxHookSuite) TestASignedCallIsCountedAsItsNumbersPhoneWork() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")

	unsigned, _, _ := s.deliver(s.call(s.e164), false)
	s.Require().Equal(http.StatusUnauthorized, unsigned)
	signed, _, body := s.deliver(s.call(s.e164), true)
	s.Require().Equal(http.StatusOK, signed, body)

	s.Equal(int64(0), s.countedOnce(activity.AreaPhoneSMS, store.AuditSourceAPI).refused)
	s.Equal(int64(0), s.counted(activity.AreaHooksIn, store.AuditSourceAPI).requests)
}

// Defect: a message a channel's provider delivered, signed with the line's secret, left out of
// the app's phone and SMS work, or one signed with another secret counted.
func (s *ChannelsSuite) TestAMessageOnALineIsCountedAsItsAppsPhoneWork() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, "")
	body := s.delivery(s.writer(), "msg-"+s.utils.uuid(), "what is open?")
	forged := http.Header{}
	forged.Set("X-Hub-Signature-256", "sha256=00")

	unsigned, _ := s.post(line.WebhookUrl, forged, body)
	s.Require().Equal(http.StatusUnauthorized, unsigned)
	s.Require().Equal(http.StatusAccepted, s.deliver(line, s.writer(), "msg-"+s.utils.uuid(), "what is open?"))

	// Connecting the line is the app's phone work from the API too.
	s.Require().Eventually(func() bool { return s.counted(activity.AreaPhoneSMS, store.AuditSourceAPI).requests >= 2 },
		settleFor, 10*time.Millisecond, "the delivery was not counted")
	s.Never(func() bool { return s.counted(activity.AreaPhoneSMS, store.AuditSourceAPI).requests > 2 },
		10*suiteActivityInterval, suiteActivityInterval, "the forged delivery was counted")
	s.Equal(int64(0), s.counted(activity.AreaPhoneSMS, store.AuditSourceAPI).refused)
}

// Defect: a vendor's status post for a placed call, which carries only the call's token, left
// out of the app's phone work.
func (s *CallStatusSuite) TestAStatusPostIsCountedAsItsCallsPhoneWork() {
	s.useApp(s.data.createApp())
	token, _ := s.placed("twilio")

	s.Require().Equal(http.StatusNoContent, s.postForm(token, url.Values{"CallStatus": {"completed"}}))

	s.countedOnce(activity.AreaPhoneSMS, store.AuditSourceAPI)
}

// Defect: a vendor fetching a placed call's plan, which carries only the call's token, left out
// of the app's phone work.
func (s *CallStatusSuite) TestAPlanFetchIsCountedAsItsCallsPhoneWork() {
	s.useApp(s.data.createApp())
	token := s.utils.uuid()
	s.Require().NoError(s.store.ParkBridge(context.Background(), &store.CallBridge{
		Token: token, CustomerID: s.customerID(), Vendor: "twilio", TrunkURI: "sip:trunk@trunk.sip.example.test",
		CallID: s.utils.uuid(), ExpiresAt: time.Now().Add(time.Minute),
	}))

	// Twilio takes its plan on the request that places the call, so it has none to fetch: the
	// fetch is still the app's, refused.
	s.Require().Equal(http.StatusNotFound, s.unauthenticatedClient.do(http.MethodGet, "/v1/phone/answer/"+token, nil, nil))

	s.Equal(int64(1), s.countedOnce(activity.AreaPhoneSMS, store.AuditSourceAPI).refused)
}

// Defect: an event the customer's own provider app delivered, checked with that app's signing
// secret, left uncounted, or one signed with another secret counted.
func (s *EventForwardingSuite) TestAProviderAppsEventIsCountedUnderItsCustomer() {
	body := s.interaction(`{"type":"block_actions","team":{"id":"` + s.workspace + `"},"user":{"id":"U0000ALICE"},` +
		`"trigger_id":"synthetic.trigger","actions":[]}`)

	forged, _, _ := s.slack.DeliverInteraction(s.eventsURL(), "not the app's secret", body)
	s.Require().Equal(http.StatusUnauthorized, forged)
	signed, _, _ := s.slack.DeliverInteraction(s.eventsURL(), s.secret, body)
	s.Require().Equal(http.StatusOK, signed)

	s.Equal(int64(0), s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI).refused)
}

// Defect: a call Telnyx signed to a number of an app whose telephony is suspended, refused in a
// 200 plan, counted as answered or with no code.
func (s *TelnyxHookSuite) TestACallToASuspendedAppIsCountedRefused() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@example.test",
	}))

	status, _, body := s.deliver(s.call(s.e164), true)
	s.Require().Equal(http.StatusOK, status)
	s.Require().Contains(body, "<Reject")

	row := s.countedOnce(activity.AreaPhoneSMS, store.AuditSourceAPI)
	s.Equal(int64(1), row.refused)
	s.Equal(codeTelephonySuspended, row.code)
}

// Defect: a call Stream signed to a number of an app whose telephony is suspended, refused and
// ended but answered 200, counted as answered or with no code.
func (s *CallHooksSuite) TestACallToASuspendedAppIsCountedRefused() {
	s.hold("default", s.callID)
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@example.test",
	}))

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	row := s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
	s.Equal(int64(1), row.refused)
	s.Equal(codeTelephonySuspended, row.code)
}

// Defect: a call no worker could take, refused and ended but answered 200, counted as answered
// or with no code.
func (s *CallHooksSuite) TestACallNoWorkerCanTakeIsCountedRefused() {
	s.hold("default", s.callID)

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	row := s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
	s.Equal(int64(1), row.refused)
	s.Equal(unavailable("").Code, row.code)
}

// Defect: a call to a number with an agent on a router that runs no sessions, refused and
// ended but answered 200, counted as answered or with no code.
func (s *CallHooksSuite) TestACallToANumberWithAnAgentOnARouterWithNoSessionsIsCountedRefused() {
	s.answerable()
	s.hold("agent", s.callID)
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, s.config("echo/local-model")))
	sessions := s.router.sessions
	s.router.sessions = nil
	defer func() { s.router.sessions = sessions }()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("agent"))

	row := s.countedOnce(activity.AreaHooksIn, store.AuditSourceAPI)
	s.Equal(int64(1), row.refused)
	s.Equal(errNoSessions.Code, row.code)
}
