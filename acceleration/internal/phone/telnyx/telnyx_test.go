package telnyx

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

type TelnyxSuite struct {
	suite.Suite
	ctx      context.Context
	server   *httptest.Server
	provider *Provider

	// seen is what the last request asked for.
	seen request
	// respond answers the next request.
	respond func(w http.ResponseWriter, r *http.Request)
}

type request struct {
	method string
	path   string
	query  url.Values
	body   map[string]any
	auth   string
}

func TestTelnyx(t *testing.T) { suite.Run(t, new(TelnyxSuite)) }

func (s *TelnyxSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.answer(`{"data":{}}`)

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &body)
		}
		s.seen = request{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.Query(),
			body:   body,
			auth:   r.Header.Get("Authorization"),
		}
		w.Header().Set("Content-Type", "application/json")
		s.respond(w, r)
	}))

	provider, err := New(Options{
		APIKey:       "KEY123",
		ConnectionID: "conn-1",
		BaseURL:      s.server.URL,
	})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *TelnyxSuite) TearDownTest() { s.server.Close() }

func (s *TelnyxSuite) TestAKeyIsRequired() {
	s.T().Setenv("TELNYX_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "TELNYX_API_KEY")
}

func (s *TelnyxSuite) TestSearchingFiltersByCountryAndReturnsWhatIsOffered() {
	s.answer(`{"data":[{
		"phone_number":"+15125551234","country_code":"US",
		"features":[{"name":"voice"},{"name":"sms"}],
		"region_information":[{"region_type":"state","region_name":"TX"},
		                      {"region_type":"rate_center","region_name":"AUSTIN"}],
		"cost_information":{"monthly_cost":"1.00"}}]}`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:      "us",
		AreaCode:     "512",
		Limit:        3,
		Capabilities: []phone.Capability{phone.Voice},
	})
	s.Require().NoError(err)

	s.Equal("/v2/available_phone_numbers", s.seen.path)
	s.Equal("US", s.seen.query.Get("filter[country_code]"))
	s.Equal("512", s.seen.query.Get("filter[national_destination_code]"))
	s.Equal("3", s.seen.query.Get("filter[limit]"))
	s.Equal("Bearer KEY123", s.seen.auth)

	s.Require().Len(offered, 1)
	s.Equal("+15125551234", offered[0].E164)
	s.Equal("TX", offered[0].Region)
	s.Equal("AUSTIN", offered[0].Locality)
	s.Equal([]phone.Capability{phone.Voice, phone.SMS}, offered[0].Capabilities)
	s.Equal(int64(1_000_000), offered[0].MonthlyCostMicros)
}

func (s *TelnyxSuite) TestSearchingWithoutACountryIsRejectedBeforeAnyCall() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{})

	s.ErrorContains(err, "country is required")
	s.Empty(s.seen.path)
}

func (s *TelnyxSuite) TestBuyingANumberOrdersItOnTheConnectionThatReachesTheBridge() {
	s.answer(`{"data":{"id":"order-9","phone_numbers":[
		{"phone_number":"+15125551234","country_code":"US","features":[{"name":"voice"}]}]}}`)

	bought, err := s.provider.BuyNumber(s.ctx, "+15125551234")
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v2/number_orders", s.seen.path)
	s.Equal("conn-1", s.seen.body["connection_id"])

	s.Equal("order-9", bought.VendorID)
	s.Equal("telnyx", bought.Vendor)
	s.Equal("US", bought.Country)
	s.Equal([]phone.Capability{phone.Voice}, bought.Capabilities)
}

func (s *TelnyxSuite) TestReleasingANumberLooksUpItsIdFirst() {
	var paths []string
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		_, _ = w.Write([]byte(`{"data":[{"id":"num-3","phone_number":"+15125551234"}]}`))
	}

	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+15125551234"))

	s.Equal([]string{"GET /v2/phone_numbers", "DELETE /v2/phone_numbers/num-3"}, paths)
}

func (s *TelnyxSuite) TestANumberThisAccountDoesNotOwnCannotBeReleased() {
	s.answer(`{"data":[]}`)

	err := s.provider.ReleaseNumber(s.ctx, "+15125551234")

	s.ErrorContains(err, "not one of this account's numbers")
}

func (s *TelnyxSuite) TestConfiguringInboundAssignsTheNumberToTheConnection() {
	s.answer(`{"data":[{"id":"num-3","phone_number":"+15125551234"}]}`)

	err := s.provider.ConfigureInbound(s.ctx, phone.Inbound{
		E164:   "+15125551234",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})
	s.Require().NoError(err)

	s.Equal(http.MethodPatch, s.seen.method)
	s.Equal("/v2/phone_numbers/num-3", s.seen.path)
	s.Equal("conn-1", s.seen.body["connection_id"])
}

func (s *TelnyxSuite) TestWithoutAConnectionThereIsNowhereToRouteACallTo() {
	provider, err := New(Options{APIKey: "KEY123", BaseURL: s.server.URL})
	s.Require().NoError(err)

	err = provider.ConfigureInbound(s.ctx, phone.Inbound{
		E164:   "+15125551234",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})

	s.ErrorContains(err, "connection id is required")
}

func (s *TelnyxSuite) TestDiallingOutLinksTheAnsweredCallToTheTrunk() {
	s.answer(`{"data":{"call_control_id":"cc-1","is_alive":true}}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From: "+15125551234",
		To:   "+15550001111",
		Bridge: phone.Bridge{
			URI:      "sip:trunk-7@sip.stream-io-api.com",
			Username: "agent",
			Password: "hunter2",
		},
	})
	s.Require().NoError(err)

	s.Equal("/v2/calls", s.seen.path)
	s.Equal("+15550001111", s.seen.body["to"])
	s.Equal("sip:trunk-7@sip.stream-io-api.com", s.seen.body["link_to"])
	s.Equal("agent", s.seen.body["sip_auth_username"])
	s.Equal("cc-1", placed.VendorCallID)
	s.Equal("dialing", placed.Status)
}

func (s *TelnyxSuite) TestPressingDigitsActsOnTheLegWithoutEndingIt() {
	// Telnyx sends the tones as an action on the live call, so the agent stays bridged
	// while the menu is answered.
	s.answer(`{"data":{"result":"ok"}}`)

	s.Require().NoError(s.provider.SendDigits(s.ctx, "cc-1", "4123"))

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v2/calls/cc-1/actions/send_dtmf", s.seen.path)
	s.Equal("4123", s.seen.body["digits"])
}

func (s *TelnyxSuite) TestPressingWithoutACallOrDigitsIsRejected() {
	s.ErrorContains(s.provider.SendDigits(s.ctx, "", "1"), "call to press them on")
	s.ErrorContains(s.provider.SendDigits(s.ctx, "cc-1", ""), "digits to press")
}

func (s *TelnyxSuite) TestACallWithoutBothEndsIsRejected() {
	_, err := s.provider.Dial(s.ctx, phone.Outbound{To: "+15550001111"})

	s.ErrorContains(err, "from and a to")
}

func (s *TelnyxSuite) TestAFailureFromTelnyxSaysWhatTelnyxSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"errors":[{"detail":"no such number"}]}`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	s.ErrorContains(err, "422")
	s.ErrorContains(err, "no such number")
}

func (s *TelnyxSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}
