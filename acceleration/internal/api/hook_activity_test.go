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
