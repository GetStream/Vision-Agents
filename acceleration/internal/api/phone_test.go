//go:build integration

package api

import (
	"context"
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
