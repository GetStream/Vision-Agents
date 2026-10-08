//go:build integration

package api

import (
	"net/http"
	"strings"
	"testing"
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
