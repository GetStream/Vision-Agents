//go:build integration

package api

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/humatest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SIPTrunksSuite covers customers' own SIP trunks over HTTP. Every test makes its own app,
// so nothing one test adds is in another's way.
type SIPTrunksSuite struct {
	RouterSuite
}

func TestSIPTrunksSuite(t *testing.T) {
	runSuite(t, new(SIPTrunksSuite))
}

func (s *SIPTrunksSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// e164 is a fictional number of this run's own, since nothing is cleaned up between runs.
func e164() string {
	return fmt.Sprintf("+1555%07d", rand.IntN(10_000_000))
}

func (s *SIPTrunksSuite) createTrunk() SipTrunk {
	var created SipTrunk
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/phone/trunks",
		map[string]any{
			"name": "main", "host": "trunk.example.com", "username": "agent",
			"password": "s3cret", "late_offer": true,
		}, &created))
	return created
}

func (s *SIPTrunksSuite) TestATrunkIsCreatedWithDefaultsAndNoPasswordInTheAnswer() {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/phone/trunks", map[string]any{
		"name": "main", "host": "trunk.example.com", "username": "agent", "password": "s3cret",
	})
	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.NotContains(string(payload), "s3cret")
	s.NotContains(string(payload), "password_sealed")

	created := s.createTrunk()
	s.Equal(5060, created.Port)
	s.Equal("tcp", created.Transport)
	s.Equal([]string{"PCMU", "PCMA"}, created.Codecs)
	s.True(created.HasPassword)
}

func (s *SIPTrunksSuite) TestATrunkNeedsAPassword() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/trunks", map[string]any{
		"name": "main", "host": "trunk.example.com", "username": "agent",
	})
	s.Equal(http.StatusBadRequest, status)
	s.Equal("sip_trunk: invalid: password is required", failure)
}

func (s *SIPTrunksSuite) TestEveryProblemWithATrunkIsNamedAtOnce() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/trunks", map[string]any{
		"name": "main", "host": "sip:trunk.example.com", "username": "agent", "password": "s3cret",
		"transport": "sctp", "codecs": []string{"OPUS"},
	})
	s.Equal(http.StatusBadRequest, status)
	s.Equal(`sip_trunk: invalid: host "sip:trunk.example.com" must be a bare hostname, without sip: or a port; `+
		`transport "sctp" must be udp, tcp or tls; codec "OPUS" is not supported, use PCMU, PCMA or G722`, failure)
}

func (s *SIPTrunksSuite) TestABadNumberIsRefusedWithItsReason() {
	created := s.createTrunk()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/trunks/"+created.Id+"/numbers",
		map[string]any{"e164": "5550000", "country": "USA"})
	s.Equal(http.StatusBadRequest, status)
	s.Equal(`sip_trunk: invalid: e164 "5550000" is not an E.164 number, e.g. +15551234567; `+
		`country "USA" must be a two-letter code, e.g. US`, failure)
}

func (s *SIPTrunksSuite) TestUpdatingWithoutAPasswordKeepsIt() {
	created := s.createTrunk()

	var updated SipTrunk
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/phone/trunks/"+created.Id,
		map[string]any{"name": "renamed", "port": 5080}, &updated))
	s.Equal("renamed", updated.Name)
	s.Equal(5080, updated.Port)
	s.Equal("trunk.example.com", updated.Host)
	s.True(updated.HasPassword)
}

func (s *SIPTrunksSuite) TestAPasswordSealedAtNoKeyIsNotAPassword() {
	created := s.createTrunk()
	// What a data move leaves on a trunk whose password was set again before it.
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE sip_trunks SET password_kek_version = 0 WHERE id = ?", created.Id)
	s.Require().NoError(err)

	var got SipTrunk
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/trunks/"+created.Id, nil, &got))
	s.False(got.HasPassword)
}

func (s *SIPTrunksSuite) TestAnEmptyPasswordIsRefused() {
	created := s.createTrunk()

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/phone/trunks/"+created.Id,
		map[string]any{"password": ""})
	s.Equal(http.StatusBadRequest, status)
	s.Equal("sip_trunk: invalid: password cannot be empty", failure)
}

func (s *SIPTrunksSuite) TestAnotherCustomersTrunkIsNotFound() {
	created := s.createTrunk()
	s.useApp(s.data.createApp())

	for _, request := range []struct {
		method string
		path   string
		body   any
	}{
		{http.MethodGet, "/v1/phone/trunks/" + created.Id, nil},
		{http.MethodPatch, "/v1/phone/trunks/" + created.Id, map[string]any{"name": "mine"}},
		{http.MethodDelete, "/v1/phone/trunks/" + created.Id, nil},
		{http.MethodPost, "/v1/phone/trunks/" + created.Id + "/numbers", map[string]any{"e164": e164(), "country": "US"}},
	} {
		status, failure := s.serverClient.failure(request.method, request.path, request.body)
		s.Equal(http.StatusNotFound, status, request.method+" "+request.path)
		s.Equal("store: no such sip trunk", failure, request.method+" "+request.path)

		// The same answer as for an id nobody made, so a trunk id says nothing about whose it is.
		neverStatus, neverFailure := s.serverClient.failure(request.method,
			strings.Replace(request.path, created.Id, "never-made", 1), request.body)
		s.Equal(neverStatus, status, request.method+" "+request.path)
		s.Equal(neverFailure, failure, request.method+" "+request.path)
	}

	var listed []SipTrunk
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/trunks", nil, &listed))
	s.Empty(listed)
}

func (s *SIPTrunksSuite) TestANumberOnATrunkIsListedWithItsTrunk() {
	created := s.createTrunk()
	number := e164()

	var added PhoneNumber
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost,
		"/v1/phone/trunks/"+created.Id+"/numbers", map[string]any{"e164": number, "country": "us"}, &added))
	s.Equal("sip_trunk", added.Vendor)
	s.Equal("US", added.Country)
	s.Equal(int64(0), added.MonthlyCostMicros)
	s.Require().NotNil(added.SipTrunkId)
	s.Equal(created.Id, *added.SipTrunkId)

	var numbers []PhoneNumber
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/numbers", nil, &numbers))
	s.Require().Len(numbers, 1)
	// Postgres keeps microseconds, the answer to the POST had nanoseconds.
	s.WithinDuration(added.PurchasedAt, numbers[0].PurchasedAt, time.Millisecond)
	numbers[0].PurchasedAt = added.PurchasedAt
	s.Equal(added, numbers[0])
}

func (s *SIPTrunksSuite) TestATrunkWithANumberCannotBeDeletedUntilItIsReleased() {
	created := s.createTrunk()
	number := e164()
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost,
		"/v1/phone/trunks/"+created.Id+"/numbers", map[string]any{"e164": number, "country": "US"}, nil))

	status, failure := s.serverClient.failure(http.MethodDelete, "/v1/phone/trunks/"+created.Id, nil)
	s.Equal(http.StatusConflict, status)
	s.Equal("store: the sip trunk still has numbers on it", failure)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/numbers/"+number, nil, nil))
	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/trunks/"+created.Id, nil, nil))
}

func (s *SIPTrunksSuite) TestTheSameNumberTwiceIsAConflict() {
	created := s.createTrunk()
	number := e164()
	path := "/v1/phone/trunks/" + created.Id + "/numbers"
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, path,
		map[string]any{"e164": number, "country": "US"}, nil))

	status, failure := s.serverClient.failure(http.MethodPost, path, map[string]any{"e164": number, "country": "US"})
	s.Equal(http.StatusConflict, status)
	s.Equal("store: the customer already holds this number", failure)
}

func (s *SIPTrunksSuite) TestSIPTrunkIsNotAVendorOnOffer() {
	var vendors []PhoneVendor
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/vendors", nil, &vendors))
	s.NotEmpty(vendors)
	for _, vendor := range vendors {
		s.NotEqual("sip_trunk", vendor.Vendor)
	}
}

func (s *SIPTrunksSuite) TestATrunkNumberCannotBeAttached() {
	created := s.createTrunk()
	number := e164()
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost,
		"/v1/phone/trunks/"+created.Id+"/numbers", map[string]any{"e164": number, "country": "US"}, nil))

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/numbers/"+number+"/attach", nil)
	s.Equal(http.StatusBadRequest, status)
	s.Equal("phone: inbound calls to a number on the customer's own sip trunk are not supported", failure)
}

func (s *SIPTrunksSuite) TestSIPTrunkFailuresMapToTheirStatus() {
	for err, want := range map[error]int{
		phone.ErrSIPTrunksDisabled: http.StatusBadRequest,
		store.ErrNoSIPTrunk:        http.StatusNotFound,
		store.ErrSIPTrunkInUse:     http.StatusConflict,
		store.ErrNumberHeld:        http.StatusConflict,
	} {
		var status huma.StatusError
		s.Require().ErrorAs(sipTrunkFailure(fmt.Errorf("wrapped: %w", err)), &status)
		s.Equal(want, status.GetStatus(), err.Error())
	}

	invalid := fmt.Errorf("%w: port 0 must be 1..65535", phone.ErrInvalidSIPTrunk)
	var status huma.StatusError
	s.Require().ErrorAs(sipTrunkFailure(invalid), &status)
	s.Equal(http.StatusBadRequest, status.GetStatus())
	s.Equal("sip_trunk: invalid: port 0 must be 1..65535", status.Error())

	// Anything else goes back unchanged, which Huma answers 500 like every other operation's
	// own failure.
	dbDown := errors.New("db down")
	s.Require().ErrorIs(sipTrunkFailure(dbDown), dbDown)
	s.False(errors.As(sipTrunkFailure(dbDown), &status))

	_, api := humatest.New(s.T())
	huma.Get(api, "/failing", func(context.Context, *struct{}) (*struct{}, error) {
		return nil, sipTrunkFailure(dbDown)
	})
	answer := api.Get("/failing")
	s.Equal(http.StatusInternalServerError, answer.Code)
	s.JSONEq(`{"error":"unexpected error occurred: db down"}`, answer.Body.String())
}

func (s *SIPTrunksSuite) TestNoTrunkAnswerCarriesThePassword() {
	created := s.createTrunk()

	for _, request := range []struct {
		method string
		path   string
		body   any
	}{
		{http.MethodGet, "/v1/phone/trunks", nil},
		{http.MethodGet, "/v1/phone/trunks/" + created.Id, nil},
		{http.MethodPatch, "/v1/phone/trunks/" + created.Id, map[string]any{"password": "n3w-s3cret"}},
	} {
		status, payload := s.serverClient.call(request.method, request.path, request.body)
		s.Require().Equal(http.StatusOK, status, string(payload))
		for _, leak := range []string{"s3cret", "password_sealed", "PasswordSealed", "password_kek_version", "PasswordKEKVersion"} {
			s.NotContains(string(payload), leak, request.method+" "+request.path)
		}
		s.Contains(string(payload), `"has_password":true`, request.method+" "+request.path)
	}
}

func (s *SIPTrunksSuite) TestATrunkThatNeverExistedIsNotFound() {
	for _, request := range []struct {
		method string
		path   string
		body   any
	}{
		{http.MethodGet, "/v1/phone/trunks/never-made", nil},
		{http.MethodPatch, "/v1/phone/trunks/never-made", map[string]any{"name": "mine"}},
		{http.MethodDelete, "/v1/phone/trunks/never-made", nil},
		{http.MethodPost, "/v1/phone/trunks/never-made/numbers", map[string]any{"e164": e164(), "country": "US"}},
	} {
		status, failure := s.serverClient.failure(request.method, request.path, request.body)
		s.Equal(http.StatusNotFound, status, request.method+" "+request.path)
		s.Equal("store: no such sip trunk", failure, request.method+" "+request.path)
	}
}

// An empty trunk id never reaches the phone service: with nothing after it the path is
// unknown, and an empty segment before /numbers is refused as malformed.
func (s *SIPTrunksSuite) TestAnEmptyTrunkIdIsRefusedBeforeTheService() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/trunks//numbers",
		map[string]any{"e164": e164(), "country": "US"})
	s.Equal(http.StatusBadRequest, status)
	s.Equal("validation failed: required path parameter is missing (path.id: )", failure)

	for _, method := range []string{http.MethodPatch, http.MethodDelete} {
		status, failure := s.serverClient.failure(method, "/v1/phone/trunks/", map[string]any{"name": "mine"})
		s.Equal(http.StatusNotFound, status, method)
		s.Equal("404 page not found\n", failure, method)
	}
}
