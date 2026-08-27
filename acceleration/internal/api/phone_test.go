package api

import (
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// The suite's server has no telephony, so these cover the two things the HTTP contract
// owes a caller without one: authentication, and saying plainly that the deployment
// cannot do this rather than pretending a number was bought.

func (s *ServerSuite) TestPhoneVendorsRequireTheCustomerHeader() {
	recorder := s.get("/v1/phone/vendors", "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestWithoutTelephonyThereAreNoVendorsRatherThanAnError() {
	recorder := s.get("/v1/phone/vendors", "acme")

	s.Equal(http.StatusOK, recorder.Code)

	var vendors []PhoneVendor
	s.decode(recorder, &vendors)
	s.Empty(vendors)
}

func (s *ServerSuite) TestWithoutTelephonyBuyingANumberSaysWhatIsMissing() {
	recorder := s.post("/v1/phone/numbers", "acme", `{"vendor":"twilio","e164":"+15125551234"}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no telephony configured")
}

func (s *ServerSuite) TestBuyingANumberNeedsABody() {
	recorder := s.post("/v1/phone/numbers", "acme", "")

	s.Equal(http.StatusBadRequest, recorder.Code)
}

func (s *ServerSuite) TestPlacingACallRequiresTheCustomerHeader() {
	recorder := s.post("/v1/phone/calls", "", `{"from":"+15125551234","to":"+15550001111"}`)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

// With telephony configured but no vendor credentials, the paths reach the vendor layer
// and report what it said rather than a missing dependency.

func (s *ServerSuite) TestSearchingAtAnUnimplementedVendorIsNotFound() {
	s.withTelephony()

	recorder := s.get("/v1/phone/numbers/available?vendor=bics&country=US", "acme")

	s.Equal(http.StatusNotFound, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "not implemented")
	s.Contains(failure.Error, "bics")
}

func (s *ServerSuite) TestSearchingAtAnUnknownVendorIsRejected() {
	s.withTelephony()

	recorder := s.get("/v1/phone/numbers/available?vendor=carrier-pigeon&country=US", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "not a known vendor")
}

func (s *ServerSuite) TestSearchingWithoutAVendorSaysNoVendorCanBeAsked() {
	// Leaving the vendor out fans the search out, and this deployment has no vendor with
	// credentials, which is a different answer from having nothing for sale.
	s.withTelephony()

	recorder := s.get("/v1/phone/numbers/available?country=US&administrative_area=CO", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no vendor has the credentials")
}

func (s *ServerSuite) TestEveryVendorSaysWhichOperationsItSupports() {
	s.withTelephony()

	recorder := s.get("/v1/phone/vendors", "acme")
	s.Equal(http.StatusOK, recorder.Code)

	var vendors []PhoneVendor
	s.decode(recorder, &vendors)
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

func (s *ServerSuite) TestListingNumbersWithoutADatabaseSaysSo() {
	s.withTelephony()

	recorder := s.get("/v1/phone/numbers", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "needs a database")
}

func (s *ServerSuite) TestANumberCannotBeBoughtWithLabelsTheRollupsCannotCarry() {
	s.withTelephony()

	// A key longer than the rollups index is rejected before the vendor is called, so a
	// number is never bought under a label that cannot be reported on.
	body := `{"vendor":"twilio","e164":"+15125551234","tags":{"` +
		strings.Repeat("x", 100) + `":"v"}}`

	recorder := s.post("/v1/phone/numbers", "acme", body)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "tag")
}

func (s *ServerSuite) TestACallCannotRingForLessThanNoTime() {
	s.withTelephony()

	recorder := s.post("/v1/phone/calls", "acme",
		`{"from":"+15125551234","to":"+15550001111","ring_timeout_seconds":-5}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "less than no time")
}

func (s *ServerSuite) TestTheAnswerPathIsReachedByAVendorThatHasNoCustomerToName() {
	// A telephony vendor fetching a call plan sends no customer header, so this path has
	// to be outside the middleware that requires one. Being unauthorized here would mean a
	// vendor bridging a live call to nowhere.
	s.withTelephony()

	recorder := s.get("/v1/phone/answer/some-token", "")

	s.Equal(http.StatusNotFound, recorder.Code)
	s.Contains(recorder.Body.String(), "not waiting to be answered")
}

// withTelephony rebuilds the handler with a phone service that has vendors declared but
// no store, no Stream and no vendor credentials.
func (s *ServerSuite) withTelephony() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	phoneConfig, err := phone.DefaultConfig()
	s.Require().NoError(err)

	telephony, err := phone.NewService(phone.ServiceOptions{
		Registry: phone.NewRegistry(phoneConfig),
	})
	s.Require().NoError(err)

	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: speech},
		Phone:   telephony,
	})
	s.Require().NoError(err)
	s.handler = server.Handler()
}

// post issues a request with a JSON body, optionally with the customer header.
func (s *ServerSuite) post(path, customerID, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	if customerID != "" {
		request.Header.Set(CustomerHeader, customerID)
	}
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)
	return recorder
}
