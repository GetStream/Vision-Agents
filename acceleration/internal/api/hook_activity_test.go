//go:build integration

package api

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
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

// Defect: a call Telnyx signed that the router hangs up on for its own failure past finding the
// number's app, a trunk password it cannot open, counted as answered.
func (s *TelnyxHookSuite) TestACallWhoseTrunkPasswordCannotBeOpenedIsCountedRefused() {
	ctx := context.Background()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: s.e164, Vendor: "telnyx", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(ctx, s.customerID(), s.e164, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), CallType: "default",
		BridgeURI: "sip:" + s.e164 + "@trunk.sip.example.test", BridgeUsername: "trunk-user",
		// Bytes no key sealed, so no version of the suite's key opens them.
		BridgePasswordSealed: []byte("not sealed by any key"), BridgePasswordKEKVersion: 1,
	}))

	status, _, body := s.deliver(s.call(s.e164), true)
	s.Require().Equal(http.StatusOK, status)
	s.Require().Contains(body, "<Hangup/>")
	s.Require().True(s.loggedLine("could not find its trunk"))

	row := s.countedOnce(activity.AreaPhoneSMS, store.AuditSourceAPI)
	s.Equal(int64(1), row.refused)
	s.Equal(internalError().Code, row.code)
}

// Defect: a call Telnyx signed that the router hangs up on because the number's stored trunk is
// not one TeXML can dial, counted as answered.
func (s *TelnyxHookSuite) TestACallWhoseTrunkCannotBeRenderedIsCountedRefused() {
	s.attach(s.e164, "trunk.sip.example.test", "trunk-user", "trunk-password")

	status, _, body := s.deliver(s.call(s.e164), true)
	s.Require().Equal(http.StatusOK, status)
	s.Require().Contains(body, "<Hangup/>")
	s.Require().True(s.loggedLine("could not render its trunk"))

	row := s.countedOnce(activity.AreaPhoneSMS, store.AuditSourceAPI)
	s.Equal(int64(1), row.refused)
	s.Equal(internalError().Code, row.code)
}

// activityWritten waits until what the suite's activity record holds now is written: a row
// recorded after it is written in the same write, one transaction, or a later one.
func (s *RouterSuite) activityWritten() {
	marker := "written-" + rand.Text()
	s.activity.Record(activity.Request{AppID: s.customerID(), Area: marker, Client: store.AuditSourceAPI, At: time.Now()})
	s.Require().Eventually(func() bool { return s.counted(marker, store.AuditSourceAPI).requests == 1 },
		settleFor, 10*time.Millisecond, "the activity record was not written")
}

// hooksAfter delivers, and returns the app's hooks_in row before and after it.
func (s *RouterSuite) hooksAfter(deliver func()) (before, after counted) {
	s.activityWritten()
	before = s.counted(activity.AreaHooksIn, store.AuditSourceAPI)
	deliver()
	s.activityWritten()
	return before, s.counted(activity.AreaHooksIn, store.AuditSourceAPI)
}

// Defect: an event the app's MCP server signed with the subscription's secret left out of the
// app's hooks, or one signed with another subscription's secret counted under the app.
func (s *ConnectionEventsSuite) TestASignedMCPEventIsCountedAsTheAppsHookAndAForgedOneIsNot() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	binding["events"] = append(binding["events"].([]map[string]any), map[string]any{"event": "issue.closed"})
	s.subscribed(binding)
	subs := s.atTheFake(connection)
	s.Require().Len(subs, 2)

	before, after := s.hooksAfter(func() {
		s.Require().Equal(http.StatusUnauthorized, s.provider.DeliverEvent(subs[0].URL, subs[1].Secret, subs[0].ID,
			"evt_"+s.utils.uuid(), subs[0].Name, map[string]any{}))
		s.Require().Equal(http.StatusAccepted, s.provider.DeliverEvent(subs[0].URL, subs[0].Secret, subs[0].ID,
			"evt_"+s.utils.uuid(), subs[0].Name, map[string]any{}))
	})

	s.Equal(before.requests+1, after.requests)
	s.Equal(before.refused, after.refused)
}

// Defect: a signed delivery the router then turned down left out of the app's hooks, so the
// digest shows none of the app's failing MCP events.
func (s *ConnectionEventsSuite) TestASignedMCPEventTheRouterTurnsDownIsCountedRefused() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	sub := s.atTheFake(connection)[0]

	before, after := s.hooksAfter(func() {
		s.Require().Equal(http.StatusBadRequest, s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID,
			"evt_"+s.utils.uuid(), "issue.closed", map[string]any{}))
	})

	s.Equal(before.requests+1, after.requests)
	s.Equal(before.refused+1, after.refused)
}

// Defect: a report the registrar signed about an app's campaign left out of the app's phone
// work, or one signed with another key counted.
func (s *DLCSuite) TestASignedReportOnACampaignIsCountedAsItsAppsPhoneWork() {
	_, campaign := s.pendingWithTheVendor()
	_, stranger, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.activityWritten()
	before := s.counted(activity.AreaPhoneSMS, store.AuditSourceAPI)

	s.Require().Equal(http.StatusUnauthorized, s.report(`{"campaignId":"`+campaign+`"}`, stranger))
	s.Require().Equal(http.StatusNoContent, s.report(`{"campaignId":"`+campaign+`"}`, s.signing))
	s.activityWritten()

	after := s.counted(activity.AreaPhoneSMS, store.AuditSourceAPI)
	s.Equal(before.requests+1, after.requests)
	s.Equal(before.refused, after.refused)
	s.Equal(int64(0), s.counted(activity.AreaHooksIn, store.AuditSourceAPI).requests, "a report is phone work")
}
