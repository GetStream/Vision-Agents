//go:build integration

package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/twilio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PhoneSuite covers the telephony surface of a deployment that declares every vendor and
// holds credentials for none: which vendor can be asked for what, and what a request that
// could never reach a carrier is answered with.
type PhoneSuite struct {
	RouterSuite
}

func TestPhoneSuite(t *testing.T) {
	runSuite(t, new(PhoneSuite))
}

func (s *PhoneSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *PhoneSuite) TestEveryVendorSaysWhichOperationsItSupports() {
	var vendors []PhoneVendor
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/phone/vendors", nil, &vendors))

	s.Require().NotEmpty(vendors)
	for _, vendor := range vendors {
		if !vendor.Implemented {
			s.Nil(vendor.Operations, vendor.Vendor+" claims operations it cannot do")
			continue
		}
		s.Require().NotNil(vendor.Operations, vendor.Vendor+" does not say what it can do")
		s.Contains(*vendor.Operations, PhoneOperationBuy)
	}
}

func (s *PhoneSuite) TestSearchingAtAVendorNobodyHasWrittenYetIsNotFound() {
	status, failure := s.serverClient.failure(http.MethodGet,
		"/v1/phone/numbers/available?vendor=bics&country=US", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, "not implemented")
	s.Contains(failure, "bics")
}

func (s *PhoneSuite) TestSearchingAtAVendorThatIsNotOneIsRefused() {
	status, failure := s.serverClient.failure(http.MethodGet,
		"/v1/phone/numbers/available?vendor=carrier-pigeon&country=US", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not a known vendor")
}

func (s *PhoneSuite) TestSearchingEveryVendorSaysNoneCanBeAsked() {
	// Leaving the vendor out fans the search out, and no vendor here has credentials,
	// which is a different answer from having nothing for sale.
	status, failure := s.serverClient.failure(http.MethodGet,
		"/v1/phone/numbers/available?country=US&administrative_area=CO", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "no vendor has the credentials")
}

func (s *PhoneSuite) TestAnAppThatHasBoughtNothingHoldsNoNumbers() {
	var numbers []PhoneNumber
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/phone/numbers", nil, &numbers))

	s.Empty(numbers)
}

func (s *PhoneSuite) TestANumberCannotBeBoughtWithLabelsTheRollupsCannotCarry() {
	// A key longer than the rollups index is refused before the vendor is called, so a
	// number is never bought under a label nothing can be reported on.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/numbers", BuyNumberRequest{
		Vendor: "twilio", E164: "+15125551234",
		Tags: &map[string]string{strings.Repeat("x", 100): "v"},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "tag")
}

func (s *PhoneSuite) TestBuyingWithoutSayingWhatToBuyIsRefused() {
	status, _ := s.serverClient.call(http.MethodPost, "/v1/phone/numbers", BuyNumberRequest{})

	s.Equal(http.StatusBadRequest, status)
}

func (s *PhoneSuite) TestACallCannotRingForLessThanNoTime() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "+15550001111", RingTimeoutSeconds: pointerTo(-5),
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "less than no time")
}

func (s *PhoneSuite) TestACallCannotNameTheNumberItRangUnder() {
	// The call hook dispatches the agent of whoever holds the number a call names, so a
	// customer naming somebody else's would hand their call to that customer's agent.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "+15550001111", Custom: &map[string]string{"phone_number": "+15125550000"},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "phone_number")
}

func (s *PhoneSuite) TestACallToAShortCodeIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "911",
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "E.164")
}

func (s *PhoneSuite) TestACallOrATransferToAPremiumRateNumberIsForbidden() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "+19005550100",
	})
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "premium-rate")

	status, failure = s.serverClient.failure(http.MethodPost, "/v1/phone/calls/transfer", TransferCallRequest{
		From: "+15125551234", To: "+19765550100", SessionId: s.data.storedSession(),
	})
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "premium-rate")
}

// A gate that cannot read whether the callee opted out is neither a refusal nor a vendor
// failure: the client is told what failed, not that the call is forbidden.
func (s *PhoneSuite) TestACallWhoseGateCannotReadIsNotForbidden() {
	s.gateOnClosedStore()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "+15550001111",
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "database is closed")
}

func (s *PhoneSuite) TestATransferWhoseGateCannotReadIsNotForbidden() {
	session := s.data.storedSession()
	s.gateOnClosedStore()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls/transfer", TransferCallRequest{
		From: "+15125551234", To: "+15550001111", SessionId: session,
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "database is closed")
}

// The session tests dial a premium-rate number, refused only once the session is let
// through, so a 403 is a session that passed.

func (s *PhoneSuite) TestACallMayBeRoutedIntoTheCallersOwnSession() {
	own := s.serverClient.createSession(textSession(nil)).Id

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "+19005550100", SessionId: &own,
	})

	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "premium-rate")
}

func (s *PhoneSuite) TestACallMayNameTheSessionItIsAboutToStart() {
	fresh := "session-" + s.utils.uuid()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "+19005550100", SessionId: &fresh,
	})

	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "premium-rate")
}

func (s *PhoneSuite) TestACallCannotBeRoutedIntoAnotherCustomersSession() {
	theirs := s.data.backendOfAnotherApp().createSession(textSession(nil)).Id

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{
		From: "+15125551234", To: "+19005550100", SessionId: &theirs,
	})

	s.Equal(http.StatusConflict, status)
}

func (s *PhoneSuite) TestAHumanCannotBeTransferredIntoAnotherCustomersSession() {
	theirs := s.data.backendOfAnotherApp().createSession(textSession(nil)).Id

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/phone/calls/transfer", TransferCallRequest{
		From: "+15125551234", To: "+19765550100", SessionId: theirs,
	})

	s.Equal(http.StatusNotFound, status)
}

func (s *PhoneSuite) TestAHumanCannotBeTransferredIntoASessionThatNeverWas() {
	status, _ := s.serverClient.failure(http.MethodPost, "/v1/phone/calls/transfer", TransferCallRequest{
		From: "+15125551234", To: "+19765550100", SessionId: "session-" + s.utils.uuid(),
	})

	s.Equal(http.StatusNotFound, status)
}

func (s *PhoneSuite) TestTheAnswerPathIsReachedByAVendorThatHasNoCustomerToName() {
	// A telephony vendor fetching a call plan holds no credential of ours, so this path
	// has to be outside the middleware that requires one. Being unauthorized here would
	// mean a vendor bridging a live call to nowhere.
	status, failure := s.unauthenticatedClient.failure(
		http.MethodGet, "/v1/phone/answer/some-token", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, "not waiting to be answered")
}

func (s *PhoneSuite) TestOnlyTheAppsOwnBackendMayBuyNumbers() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodGet, "/v1/phone/numbers", nil)
		return status
	})
}

func (s *PhoneSuite) TestANumberThatOnlyPlacedCallsIsNotListedAsAttached() {
	s.useApp(s.data.createApp())
	ctx := context.Background()
	number := s.utils.number()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: number, Vendor: "telnyx", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	group := store.TrunkGroup{CustomerID: s.customerID(), Vendor: "telnyx"}
	s.Require().NoError(s.store.InTrunkGroup(ctx, group, func(_ context.Context, tx *store.TrunkGroupTx) error {
		return tx.SetLine(number, "trunk-"+s.utils.uuid(), "route-"+s.utils.uuid(), store.NumberBridge{})
	}))

	var numbers []PhoneNumber
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/numbers", nil, &numbers))

	s.Require().Len(numbers, 1)
	s.Nil(numbers[0].StreamTrunkId, "its callers reach no agent")
}

func (s *PhoneSuite) attachedNumber(customerID string) string {
	ctx := context.Background()
	number := s.utils.number()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: number, Vendor: "bics", Country: "US", CustomerID: customerID, PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(ctx, customerID, number, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), RouteID: "route-" + s.utils.uuid(), CallType: "default", CallID: "call-" + s.utils.uuid(),
	}))
	return number
}

func (s *PhoneSuite) TestADetachedNumberIsKeptButReachesNoAgent() {
	s.useApp(s.data.createApp())
	number := s.attachedNumber(s.customerID())

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/numbers/"+number+"/attach", nil, nil))

	var numbers []PhoneNumber
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/numbers", nil, &numbers))
	s.Require().Len(numbers, 1)
	s.Equal(number, numbers[0].E164)
	s.Nil(numbers[0].StreamTrunkId)
}

func (s *PhoneSuite) TestDetachingANumberTwiceChangesNothing() {
	s.useApp(s.data.createApp())
	number := s.attachedNumber(s.customerID())

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/numbers/"+number+"/attach", nil, nil))
	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/numbers/"+number+"/attach", nil, nil))
}

func (s *PhoneSuite) TestAnotherCustomersNumberCannotBeDetached() {
	s.useApp(s.data.createApp())
	theirs := s.attachedNumber(s.data.createApp().app.ID)

	status, _ := s.serverClient.failure(http.MethodDelete, "/v1/phone/numbers/"+theirs+"/attach", nil)

	s.Equal(http.StatusNotFound, status)
}

// CallStatusSuite is the path a vendor posts a placed call's status to. Twilio and Telnyx are
// the real providers, opened with credentials no request is ever sent with: reading a status
// reaches no vendor.
type CallStatusSuite struct {
	RouterSuite
}

func TestCallStatusSuite(t *testing.T) {
	runSuite(t, new(CallStatusSuite))
}

func (s *CallStatusSuite) SetupSuite() {
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)
	// The suites share one process, so the credentials are not set in it: the two vendors
	// are declared as needing none, and handed some of their own.
	for i, vendor := range config.Vendors {
		if vendor.Vendor == "twilio" || vendor.Vendor == "telnyx" {
			config.Vendors[i].Credentials = nil
		}
	}
	registry := vendors.Registry(config)
	registry.Register("twilio", func() (phone.Provider, error) {
		return twilio.New(twilio.Options{AccountSID: "AC1", AuthToken: "t"})
	})
	registry.Register("telnyx", func() (phone.Provider, error) {
		return telnyx.New(telnyx.Options{APIKey: "k"})
	})
	s.phoneVendors = registry
	s.sessionTelephony = true
	s.RouterSuite.SetupSuite()
}

func (s *CallStatusSuite) SetupTest() {
	s.useFixture("standard")
}

// placed keeps a call placed at vendor and returns its status token and call id.
func (s *CallStatusSuite) placed(vendor string) (string, string) {
	callID := s.utils.uuid()
	token, err := s.router.phone.Outcomes().Placed(context.Background(), outcome.Call{
		Key: outcome.Key{
			CustomerID: s.customerID(), CallType: "default", CallID: callID,
			Local: s.utils.number(), Remote: s.utils.number(),
		},
		Vendor:  vendor,
		Reports: true,
	})
	s.Require().NoError(err)
	return token, callID
}

// post posts body to token's status path as a vendor does, with no credential of ours.
func (s *CallStatusSuite) post(token, contentType, body string) (int, string) {
	request, err := http.NewRequest(http.MethodPost, s.server.URL+"/v1/phone/status/"+token, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", contentType)
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answered, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(answered)
}

// postForm posts form as a vendor's status.
func (s *CallStatusSuite) postForm(token string, form url.Values) int {
	status, _ := s.post(token, "application/x-www-form-urlencoded", form.Encode())
	return status
}

func (s *CallStatusSuite) read(callID string) outcome.Call {
	call, err := s.router.phone.Outcomes().Read(context.Background(), s.customerID(), callID)
	s.Require().NoError(err)
	return call
}

func (s *CallStatusSuite) TestATwilioStatusIsReachedWithoutACustomer() {
	busy, busyCall := s.placed("twilio")
	noAnswer, noAnswerCall := s.placed("twilio")
	failed, failedCall := s.placed("twilio")
	completed, completedCall := s.placed("twilio")

	s.Equal(http.StatusNoContent, s.postForm(busy, url.Values{"CallStatus": {"busy"}}))
	s.Equal(http.StatusNoContent, s.postForm(noAnswer, url.Values{"CallStatus": {"no-answer"}}))
	s.Equal(http.StatusNoContent, s.postForm(failed, url.Values{"CallStatus": {"failed"}}))
	s.Equal(http.StatusNoContent, s.postForm(completed, url.Values{"CallStatus": {"completed"}}))

	s.Equal(outcome.Busy, s.read(busyCall).Outcome)
	s.Equal(outcome.NoAnswer, s.read(noAnswerCall).Outcome)
	s.Equal(outcome.Failed, s.read(failedCall).Outcome)
	s.Equal(outcome.Completed, s.read(completedCall).Outcome)
}

func (s *CallStatusSuite) TestATelnyxStatusIsRecorded() {
	token, callID := s.placed("telnyx")

	s.Equal(http.StatusNoContent, s.postForm(token, url.Values{"CallStatus": {"busy"}}))

	call := s.read(callID)
	s.Equal(outcome.Busy, call.Outcome)
	s.Equal("busy", call.VendorStatus)
}

func (s *CallStatusSuite) TestAnUnknownStatusTokenIs404() {
	status, failure := s.post(s.utils.uuid(), "application/x-www-form-urlencoded", "CallStatus=busy")

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, "that call is not waiting for its status")
}

func (s *CallStatusSuite) TestAStatusThatCannotBeReadIs400() {
	token, callID := s.placed("twilio")

	empty, _ := s.post(token, "application/x-www-form-urlencoded", "")
	notAForm, _ := s.post(token, "application/json", `{"CallStatus":"busy"}`)

	s.Equal(http.StatusBadRequest, empty)
	s.Equal(http.StatusBadRequest, notAForm)
	call := s.read(callID)
	s.Empty(call.Outcome)
	s.Empty(call.VendorStatus)
}

func (s *CallStatusSuite) TestAStatusOver64KiBIs413() {
	token, _ := s.placed("twilio")

	status, _ := s.post(token, "application/x-www-form-urlencoded", "CallStatus=busy&pad="+strings.Repeat("a", 64<<10))

	s.Equal(http.StatusRequestEntityTooLarge, status)
}

// placedFor keeps a call placed at Twilio for session and returns its status token and key.
func (s *CallStatusSuite) placedFor(session string) (string, outcome.Key) {
	key := outcome.Key{
		CustomerID: s.customerID(), CallType: "default", CallID: session,
		Local: s.utils.number(), Remote: s.utils.number(),
	}
	token, err := s.router.phone.Outcomes().Placed(context.Background(), outcome.Call{Key: key, Vendor: "twilio", Reports: true})
	s.Require().NoError(err)
	return token, key
}

// busy records a busy call placed for session through the keeper, not the status path: the
// path also ends the session, and one ended a moment ago is not found until its row is written.
func (s *CallStatusSuite) busy(session string) {
	token, _ := s.placedFor(session)
	_, err := s.router.phone.Outcomes().Reported(context.Background(), token, outcome.Busy, "busy")
	s.Require().NoError(err)
}

func (s *CallStatusSuite) TestASessionShowsHowItsPlacedCallEnded() {
	opened := s.serverClient.createSession(textSession(nil))
	s.busy(opened.Id)

	read := s.serverClient.getSession(opened.Id)

	s.Require().NotNil(read.CallOutcome)
	s.Equal(CallOutcomeBusy, value(read.CallOutcome.Outcome))
	s.Nil(read.CallOutcome.AnsweredAt)
	s.Nil(read.CallOutcome.DurationSeconds)
}

func (s *CallStatusSuite) TestASessionShowsWhenTheCallWasAnsweredAndForHowLong() {
	opened := s.serverClient.createSession(textSession(nil))
	_, key := s.placedFor(opened.Id)
	answered := time.Now().UTC().Truncate(time.Second)
	s.Require().NoError(s.router.phone.Outcomes().Answered(context.Background(), key, answered))
	s.Require().NoError(s.router.phone.Outcomes().Ended(context.Background(), key, answered.Add(75*time.Second)))

	read := s.serverClient.getSession(opened.Id)

	s.Require().NotNil(read.CallOutcome)
	s.Equal(CallOutcomeCompleted, value(read.CallOutcome.Outcome))
	s.Require().NotNil(read.CallOutcome.AnsweredAt)
	s.True(answered.Equal(*read.CallOutcome.AnsweredAt))
	s.Equal(75, value(read.CallOutcome.DurationSeconds))
}

func (s *CallStatusSuite) TestASessionWithoutAPlacedCallHasNoCallOutcome() {
	opened := s.serverClient.createSession(textSession(nil))

	s.Nil(s.serverClient.getSession(opened.Id).CallOutcome)
}

func (s *CallStatusSuite) TestADeviceIsNotShownTheCallOutcome() {
	backend := s.serverClient.actingFor(s.client)
	opened := backend.createSession(textSession(nil))
	s.busy(opened.Id)

	s.Nil(s.client.getSession(opened.Id).CallOutcome)
	s.NotNil(backend.getSession(opened.Id).CallOutcome, "the backend is shown the outcome of the same session")
}

func (s *CallStatusSuite) TestABusyStatusEndsItsSessionBeforeTheStartTimeout() {
	// The suite's vendors have no credentials to place a call with, so the session is opened
	// as the phone service opens one, by the manager.
	opened, err := s.manager.Create(context.Background(), session.Spec{CustomerID: s.customerID(),
		Phone: &session.PhoneSpec{VendorCallID: "CA1"}, Attribution: routing.Attribution{APIKey: "key-phone"}})
	s.Require().NoError(err)
	token, _ := s.placedFor(opened.ID())

	s.Require().Equal(http.StatusNoContent, s.postForm(token, url.Values{"CallStatus": {"busy"}}))

	s.Require().Eventually(func() bool { return !s.manager.Running(opened.ID()) }, 10*time.Second, 20*time.Millisecond,
		"the session waits for its start timeout")
}

// openPhoneSession opens the session id for a placed call as an SDK does, after placing it.
func (s *CallStatusSuite) openPhoneSession(id string) (int, ErrorResponse) {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sessions", CreateSessionRequest{
		Id: &id, StartVoice: pointerTo(true), Phone: &SessionPhone{Number: s.utils.number()},
	})
	var answered ErrorResponse
	if status >= http.StatusBadRequest {
		s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	}
	return status, answered
}

func (s *CallStatusSuite) TestOpeningASessionAfterItsCallWasBusyIsRefused() {
	id := s.utils.uuid()
	token, _ := s.placedFor(id)
	s.Require().Equal(http.StatusNoContent, s.postForm(token, url.Values{"CallStatus": {"busy"}}))

	status, answered := s.openPhoneSession(id)

	s.Equal(http.StatusConflict, status)
	s.Equal("call_unanswered", answered.Error.Code)
	s.Contains(answered.Error.Message, "busy")
	s.False(s.manager.Running(id))
	s.Equal(outcome.Busy, s.read(id).Outcome)
}

func (s *CallStatusSuite) TestASessionOpensAfterItsCallCompleted() {
	id := s.utils.uuid()
	token, _ := s.placedFor(id)
	s.Require().Equal(http.StatusNoContent, s.postForm(token, url.Values{"CallStatus": {"completed"}}))

	status, answered := s.openPhoneSession(id)
	defer s.serverClient.stopSession(id)

	s.Equal(http.StatusCreated, status, answered.Error.Message)
	s.True(s.manager.Running(id))
}

func (s *CallStatusSuite) TestASessionOpensWhileItsCallRings() {
	id := s.utils.uuid()
	s.placedFor(id)

	status, answered := s.openPhoneSession(id)
	defer s.serverClient.stopSession(id)

	s.Equal(http.StatusCreated, status, answered.Error.Message)
	s.True(s.manager.Running(id))
}

// attachedForSessions is a number attached the way attaching does today: every caller gets a
// session of their own, so no one call is recorded.
func (s *PhoneSuite) attachedForSessions(customerID string) string {
	ctx := context.Background()
	number := s.utils.number()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: number, Vendor: "bics", Country: "US", CustomerID: customerID, PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(ctx, customerID, number, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), RouteID: "route-" + s.utils.uuid(), CallType: "agent",
	}))
	return number
}

func (s *PhoneSuite) listedNumber() PhoneNumber {
	var numbers []PhoneNumber
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/numbers", nil, &numbers))
	s.Require().Len(numbers, 1)
	return numbers[0]
}

func (s *PhoneSuite) TestANumberCanBeGivenAnAgent() {
	s.useApp(s.data.createApp())
	number := s.attachedForSessions(s.customerID())
	config := s.config("echo/local-model")

	var answered PhoneNumber
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config}, &answered))

	s.Require().NotNil(answered.AgentConfigId)
	s.Equal(config, *answered.AgentConfigId)
	listed := s.listedNumber()
	s.Require().NotNil(listed.AgentConfigId)
	s.Equal(config, *listed.AgentConfigId)
}

func (s *PhoneSuite) TestANumberCanBeGivenAnAgentByName() {
	s.useApp(s.data.createApp())
	number := s.attachedForSessions(s.customerID())
	name := "agent-" + s.utils.uuid()
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: name, Llm: pointerTo("echo/local-model")}, &created))

	var answered PhoneNumber
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{Agent: &name}, &answered))

	s.Require().NotNil(answered.AgentConfigId)
	s.Equal(created.Id, *answered.AgentConfigId)
}

func (s *PhoneSuite) TestGivingANumberAnotherAgentReplacesIt() {
	s.useApp(s.data.createApp())
	number := s.attachedForSessions(s.customerID())
	first, second := s.config("echo/local-model"), s.config("echo/local-model")

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &first}, nil))
	var answered PhoneNumber
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &second}, &answered))

	s.Require().NotNil(answered.AgentConfigId)
	s.Equal(second, *answered.AgentConfigId)
}

func (s *PhoneSuite) TestANumbersAgentCanBeRemoved() {
	s.useApp(s.data.createApp())
	number := s.attachedForSessions(s.customerID())
	config := s.config("echo/local-model")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config}, nil))

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/numbers/"+number+"/agent", nil, nil))

	s.Nil(s.listedNumber().AgentConfigId)
	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/numbers/"+number+"/agent", nil, nil))
}

func (s *PhoneSuite) TestANumberThatIsNotAttachedCannotBeGivenAnAgent() {
	s.useApp(s.data.createApp())
	number := s.utils.number()
	s.Require().NoError(s.store.RecordNumber(context.Background(), &store.PhoneNumber{
		E164: number, Vendor: "bics", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	config := s.config("echo/local-model")

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config})

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "attach the number first")
}

func (s *PhoneSuite) TestANumberAttachedToACallOfItsOwnCannotBeGivenAnAgent() {
	s.useApp(s.data.createApp())
	number := s.attachedNumber(s.customerID())
	config := s.config("echo/local-model")

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config})

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "attach the number again")
}

func (s *PhoneSuite) TestAnAgentMustBeOneOfTheCustomersConfigs() {
	s.useApp(s.data.createApp())
	number := s.attachedForSessions(s.customerID())
	unknown := "config-" + s.utils.uuid()

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &unknown})
	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownConfig.Message)

	status, _ = s.serverClient.failure(http.MethodPut, "/v1/phone/numbers/"+number+"/agent", SetNumberAgentRequest{})
	s.Equal(http.StatusBadRequest, status)
}

func (s *PhoneSuite) TestAnotherCustomersNumberCannotBeGivenAnAgent() {
	s.useApp(s.data.createApp())
	theirs := s.attachedForSessions(s.data.createApp().app.ID)
	config := s.config("echo/local-model")

	status, _ := s.serverClient.failure(http.MethodPut, "/v1/phone/numbers/"+theirs+"/agent",
		SetNumberAgentRequest{ConfigId: &config})

	s.Equal(http.StatusNotFound, status)
}

func (s *PhoneSuite) TestDetachingANumberRemovesItsAgent() {
	s.useApp(s.data.createApp())
	number := s.attachedForSessions(s.customerID())
	config := s.config("echo/local-model")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config}, nil))

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/numbers/"+number+"/attach", nil, nil))

	answering, err := s.store.NumbersAnsweredBy(context.Background(), s.customerID(), config)
	s.Require().NoError(err)
	s.Empty(answering)
	s.Nil(s.listedNumber().AgentConfigId)
}

func (s *PhoneSuite) TestReleasingANumberRemovesItsAgent() {
	s.useApp(s.data.createApp())
	ctx := context.Background()
	number := s.attachedForSessions(s.customerID())
	config := s.config("echo/local-model")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config}, nil))

	s.Require().NoError(s.store.ReleaseNumber(ctx, s.customerID(), number, time.Time{}))

	answering, err := s.store.NumbersAnsweredBy(ctx, s.customerID(), config)
	s.Require().NoError(err)
	s.Empty(answering)
	held, err := s.store.CustomerNumbers(ctx, s.customerID(), true)
	s.Require().NoError(err)
	s.Require().Len(held, 1)
	s.Empty(held[0].ConfigID, "a released row keeps no agent")
}

func (s *PhoneSuite) TestReattachingANumberKeepsItsAgent() {
	s.useApp(s.data.createApp())
	number := s.attachedForSessions(s.customerID())
	config := s.config("echo/local-model")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config}, nil))

	s.Require().NoError(s.store.AttachNumber(context.Background(), s.customerID(), number, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), RouteID: "route-" + s.utils.uuid(), CallType: "agent",
	}))

	listed := s.listedNumber()
	s.Require().NotNil(listed.AgentConfigId)
	s.Equal(config, *listed.AgentConfigId)
}

func (s *PhoneSuite) TestClearingANumbersLineRemovesItsAgent() {
	s.useApp(s.data.createApp())
	ctx := context.Background()
	number := s.attachedForSessions(s.customerID())
	config := s.config("echo/local-model")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/numbers/"+number+"/agent",
		SetNumberAgentRequest{ConfigId: &config}, nil))

	group := store.TrunkGroup{CustomerID: s.customerID(), Vendor: "bics"}
	s.Require().NoError(s.store.InTrunkGroup(ctx, group, func(_ context.Context, tx *store.TrunkGroupTx) error {
		return tx.ClearLine(number)
	}))

	answering, err := s.store.NumbersAnsweredBy(ctx, s.customerID(), config)
	s.Require().NoError(err)
	s.Empty(answering)
	s.Nil(s.listedNumber().AgentConfigId)
}
