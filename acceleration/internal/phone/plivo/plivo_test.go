package plivo

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

type PlivoSuite struct {
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
	auth   string
	body   map[string]any
}

func TestPlivo(t *testing.T) { suite.Run(t, new(PlivoSuite)) }

func (s *PlivoSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.answer(`{}`)

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, _, _ := r.BasicAuth()
		s.seen = request{method: r.Method, path: r.URL.Path, query: r.URL.Query(), auth: user}
		if payload, err := io.ReadAll(r.Body); err == nil && len(payload) > 0 {
			_ = json.Unmarshal(payload, &s.seen.body)
		}
		w.Header().Set("Content-Type", "application/json")
		s.respond(w, r)
	}))

	provider, err := New(Options{AuthID: "MA123", AuthToken: "secret", BaseURL: s.server.URL})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *PlivoSuite) TearDownTest() { s.server.Close() }

func (s *PlivoSuite) TestCredentialsAreRequired() {
	s.T().Setenv("PLIVO_AUTH_ID", "")
	s.T().Setenv("PLIVO_AUTH_TOKEN", "")

	_, err := New(Options{})

	s.ErrorContains(err, "PLIVO_AUTH_ID")
}

func (s *PlivoSuite) TestSearchingReturnsNumbersInE164WithTheirPrice() {
	// Plivo quotes a number without a plus, which is not what anything else here uses.
	s.answer(`{"objects":[{
		"number":"17195551234","country_iso":"us","region":"Colorado","city":"Colorado Springs",
		"type":"local","monthly_rental_rate":"0.80",
		"voice_enabled":true,"sms_enabled":true,"mms_enabled":false}]}`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:      "us",
		Prefix:       "719",
		Locality:     "Colorado Springs",
		Type:         phone.Local,
		Capabilities: []phone.Capability{phone.Voice},
		Limit:        5,
	})
	s.Require().NoError(err)

	s.Equal("/v1/Account/MA123/PhoneNumber/", s.seen.path)
	s.Equal("MA123", s.seen.auth)
	s.Equal("US", s.seen.query.Get("country_iso"))
	s.Equal("719", s.seen.query.Get("pattern"))
	s.Equal("Colorado Springs", s.seen.query.Get("city"))
	s.Equal("local", s.seen.query.Get("type"))
	s.Equal("voice", s.seen.query.Get("services"))
	s.Equal("5", s.seen.query.Get("limit"))

	s.Require().Len(offered, 1)
	s.Equal("+17195551234", offered[0].E164)
	s.Equal("plivo", offered[0].Vendor)
	s.Equal("US", offered[0].Country)
	s.Equal("Colorado", offered[0].Region)
	s.Equal(phone.Local, offered[0].Type)
	s.Equal([]phone.Capability{phone.Voice, phone.SMS}, offered[0].Capabilities)
	s.Equal(int64(800_000), offered[0].MonthlyCostMicros)
}

func (s *PlivoSuite) TestAnAreaCodeIsSearchedAsAPrefix() {
	// Plivo's pattern is anchored after the dial code, so an area code is a prefix to it.
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", AreaCode: "512"})
	s.Require().NoError(err)

	s.Equal("512", s.seen.query.Get("pattern"))
}

func (s *PlivoSuite) TestSearchingWithoutACountryIsRejectedBeforeAnyCall() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{})

	s.ErrorContains(err, "country is required")
	s.Empty(s.seen.path)
}

func (s *PlivoSuite) TestBuyingRentsTheNumberAtItsOwnPath() {
	s.answer(`{"status":"fulfilled","message":"created"}`)

	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234", Country: "US"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v1/Account/MA123/PhoneNumber/17195551234/", s.seen.path)

	s.Equal("+17195551234", bought.E164)
	s.Equal("plivo", bought.Vendor)
	s.Equal("US", bought.Country)
}

func (s *PlivoSuite) TestReleasingUsesTheRentedNumberPathRatherThanTheInventoryOne() {
	// A number for sale is a PhoneNumber and a number held is a Number, which are
	// different resources at Plivo.
	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+17195551234"))

	s.Equal(http.MethodDelete, s.seen.method)
	s.Equal("/v1/Account/MA123/Number/17195551234/", s.seen.path)
}

func (s *PlivoSuite) TestPointingANumberAtATrunkSaysItIsNotHere() {
	s.ErrorIs(s.provider.ConfigureInbound(s.ctx, phone.Inbound{}), phone.ErrNotImplemented)
	s.ErrorIs(s.provider.SendDigits(s.ctx, "call-1", "1"), phone.ErrNotImplemented)
}

func (s *PlivoSuite) TestDiallingOutPointsPlivoAtThePlanRatherThanCarryingIt() {
	s.answer(`{"request_uuid":"req-1","message":"call fired"}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:        "+17195551234",
		To:          "+13035559876",
		Bridge:      phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"},
		RingTimeout: 20 * time.Second,
		AnswerURL:   "https://router.example.com/v1/phone/answer/tok-1",
	})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v1/Account/MA123/Call/", s.seen.path)
	s.Equal("MA123", s.seen.auth)
	// Plivo quotes and takes numbers without the plus.
	s.Equal("17195551234", s.seen.body["from"])
	s.Equal("13035559876", s.seen.body["to"])
	s.Equal("https://router.example.com/v1/phone/answer/tok-1", s.seen.body["answer_url"])
	s.Equal(http.MethodGet, s.seen.body["answer_method"])
	s.Equal(float64(20), s.seen.body["ring_timeout"])

	s.Equal("req-1", placed.VendorCallID)
}

func (s *PlivoSuite) TestDiallingWithNowhereToFetchThePlanFromIsRefusedBeforeItIsAsked() {
	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+17195551234",
		To:     "+13035559876",
		Bridge: phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"},
	})

	s.ErrorContains(err, "somewhere to fetch it from")
	s.Empty(s.seen.path)
}

func (s *PlivoSuite) TestTheAnswerPlanCarriesTheTrunkCredentials() {
	plan, err := s.provider.Answer(phone.Bridge{
		URI:      "sip:trunk@sip.stream-io-api.com",
		Username: "trunk-user",
		Password: "trunk-pass",
	}, "")
	s.Require().NoError(err)

	s.Equal("application/xml", plan.ContentType)
	rendered := string(plan.Body)
	s.Contains(rendered, "<Response>")
	s.Contains(rendered, `sipAuthUsername="trunk-user"`)
	s.Contains(rendered, `sipAuthPassword="trunk-pass"`)
	s.Contains(rendered, "sip:trunk@sip.stream-io-api.com")
}

func (s *PlivoSuite) TestAnAnswerPlanForNoTrunkIsRefused() {
	_, err := s.provider.Answer(phone.Bridge{}, "")

	s.ErrorContains(err, "bridge uri is required")
}

func (s *PlivoSuite) TestAStateSearchIsNotClaimed() {
	// Plivo's region filter wants a state's full name, so answering "CO" with an empty
	// list would look like no inventory rather than a filter it cannot express.
	s.False(s.provider.Supports(phone.FilterAdministrativeArea))
	s.False(s.provider.Supports(phone.FilterContains))
	s.True(s.provider.Supports(phone.FilterLocality))
}

func (s *PlivoSuite) TestAFailureFromPlivoSaysWhatPlivoSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not found"}`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	s.ErrorContains(err, "404")
	s.ErrorContains(err, "not found")
}

func (s *PlivoSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}
