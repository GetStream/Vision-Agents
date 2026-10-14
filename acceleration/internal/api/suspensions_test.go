//go:build integration

package api

import (
	"net/http"
	"testing"
)

// SuspensionSuite is Stream staff stopping an app's telephony for abuse, and lifting it.
type SuspensionSuite struct {
	RouterSuite
	// staff is Stream's own tool, holding the ops key and no app's credentials.
	staff *testClient
}

func TestSuspensionSuite(t *testing.T) {
	runSuite(t, new(SuspensionSuite))
}

func (s *SuspensionSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.staff = &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
}

func (s *SuspensionSuite) path() string {
	return "/v1/ops/customers/" + s.customerID() + "/telephony-suspension"
}

func (s *SuspensionSuite) suspend() {
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPut, s.path(),
		SuspendTelephonyRequest{Reason: "calls to premium-rate numbers", SuspendedBy: "ops@getstream.io"}, nil))
}

func (s *SuspensionSuite) TestASuspendedAppCanNeitherBuyNorCallNorTransfer() {
	s.suspend()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/numbers",
		BuyNumberRequest{Vendor: "twilio", E164: "+15125551234"})
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "suspended")

	status, failure = s.serverClient.failure(http.MethodPost, "/v1/phone/calls",
		PlaceCallRequest{From: "+15125551234", To: "+15550001111"})
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "suspended")

	status, failure = s.serverClient.failure(http.MethodPost, "/v1/phone/calls/transfer",
		TransferCallRequest{From: "+15125551234", To: "+15550001111", SessionId: "session-" + s.utils.uuid()})
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "suspended")
	s.NotContains(failure, "premium-rate", "why an app is suspended is Stream's to know")
}

func (s *SuspensionSuite) TestStaffSeeWhyAnAppIsSuspendedUntilTheyLiftIt() {
	s.suspend()

	var suspension TelephonySuspension
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, s.path(), nil, &suspension))
	s.Equal(s.customerID(), suspension.CustomerId)
	s.Equal("calls to premium-rate numbers", suspension.Reason)
	s.Equal("ops@getstream.io", suspension.SuspendedBy)
	s.False(suspension.SuspendedAt.IsZero())

	s.Require().Equal(http.StatusNoContent, s.staff.do(http.MethodDelete, s.path(), nil, nil))
	s.Equal(http.StatusNotFound, s.staff.do(http.MethodGet, s.path(), nil, nil))
	status, _ := s.serverClient.failure(http.MethodPost, "/v1/phone/calls",
		PlaceCallRequest{From: "+15125551234", To: "+15550001111"})
	s.Equal(http.StatusBadRequest, status, "a lifted app gets past the suspension, then fails for want of Stream")
}

func (s *SuspensionSuite) TestLiftingAnAppThatIsNotSuspendedIsNotFound() {
	s.Equal(http.StatusNotFound, s.staff.do(http.MethodDelete, s.path(), nil, nil))
}

func (s *SuspensionSuite) TestASuspensionSaysWhyAndWhoMadeIt() {
	s.Equal(http.StatusBadRequest, s.staff.do(http.MethodPut, s.path(),
		SuspendTelephonyRequest{Reason: " ", SuspendedBy: "ops@getstream.io"}, nil))
	s.Equal(http.StatusNotFound, s.staff.do(http.MethodGet, s.path(), nil, nil))
}

func (s *SuspensionSuite) TestAnAppCannotLiftItsOwnSuspension() {
	s.suspend()

	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodDelete, s.path(), nil, nil))
	s.Equal(http.StatusOK, s.staff.do(http.MethodGet, s.path(), nil, nil))
}
