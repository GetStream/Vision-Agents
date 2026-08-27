package sinch

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

type SinchSuite struct {
	suite.Suite
	ctx      context.Context
	server   *httptest.Server
	provider *Provider

	// seen is what the last numbers request asked for.
	seen request
	// tokens counts how many times the key was exchanged for a bearer token.
	tokens int
	// respond answers the next numbers request.
	respond func(w http.ResponseWriter, r *http.Request)
	// grant answers the next token request.
	grant string
}

type request struct {
	method string
	path   string
	query  url.Values
	auth   string
	body   string
	// user is the basic-auth name, which the calling API uses where the numbers API uses
	// a bearer token.
	user string
}

func TestSinch(t *testing.T) { suite.Run(t, new(SinchSuite)) }

func (s *SinchSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.tokens = 0
	s.grant = `{"access_token":"tok-1","expires_in":3600}`
	s.answer(`{"availableNumbers":[]}`)

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/oauth2/token" {
			s.tokens++
			_, _ = w.Write([]byte(s.grant))
			return
		}
		raw, _ := io.ReadAll(r.Body)
		user, _, _ := r.BasicAuth()
		s.seen = request{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.Query(),
			auth:   r.Header.Get("Authorization"),
			body:   string(raw),
			user:   user,
		}
		s.respond(w, r)
	}))

	provider, err := New(Options{
		ProjectID:         "proj-1",
		KeyID:             "key-1",
		KeySecret:         "secret",
		ApplicationKey:    "app-key",
		ApplicationSecret: "app-secret",
		BaseURL:           s.server.URL,
		AuthURL:           s.server.URL + "/oauth2/token",
		CallingBaseURL:    s.server.URL,
	})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *SinchSuite) TearDownTest() { s.server.Close() }

func (s *SinchSuite) TestCredentialsAreRequired() {
	s.T().Setenv("SINCH_PROJECT_ID", "")
	s.T().Setenv("SINCH_KEY_ID", "")
	s.T().Setenv("SINCH_KEY_SECRET", "")

	_, err := New(Options{})

	s.ErrorContains(err, "SINCH_PROJECT_ID")
}

func (s *SinchSuite) TestTheKeyIsExchangedOnceAndTheTokenReused() {
	// Exchanging the key per call would double every request.
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})
	s.Require().NoError(err)
	_, err = s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})
	s.Require().NoError(err)

	s.Equal(1, s.tokens)
	s.Equal("Bearer tok-1", s.seen.auth)
}

func (s *SinchSuite) TestATokenThatExpiresImmediatelyIsReplaced() {
	s.grant = `{"access_token":"tok-brief","expires_in":1}`

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})
	s.Require().NoError(err)
	_, err = s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})
	s.Require().NoError(err)

	s.Equal(2, s.tokens, "a token that dies in flight is worse than asking for another")
}

func (s *SinchSuite) TestATokenRequestThatFailsSaysSo() {
	s.server.Close()

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	s.ErrorContains(err, "token")
}

func (s *SinchSuite) TestSearchingAsksForLocalNumbersWhenNoTypeIsNamed() {
	// Sinch requires a type, and an agent wants a geographic number.
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})
	s.Require().NoError(err)

	s.Equal("/v1/projects/proj-1/availableNumbers", s.seen.path)
	s.Equal("LOCAL", s.seen.query.Get("type"))
}

func (s *SinchSuite) TestSearchingReturnsWhatIsOfferedWithItsPrice() {
	s.answer(`{"availableNumbers":[{
		"phoneNumber":"+17195551234","regionCode":"US","type":"LOCAL",
		"capability":["SMS","VOICE"],"monthlyPrice":{"currencyCode":"USD","amount":"1.20"}}]}`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:      "us",
		Prefix:       "719",
		Type:         phone.TollFree,
		Capabilities: []phone.Capability{phone.Voice},
		Limit:        3,
	})
	s.Require().NoError(err)

	s.Equal("US", s.seen.query.Get("regionCode"))
	s.Equal("TOLL_FREE", s.seen.query.Get("type"))
	s.Equal("719", s.seen.query.Get("numberPattern.pattern"))
	s.Equal("START", s.seen.query.Get("numberPattern.searchPattern"))
	s.Equal("VOICE", s.seen.query.Get("capabilities"))
	s.Equal("3", s.seen.query.Get("size"))

	s.Require().Len(offered, 1)
	s.Equal("+17195551234", offered[0].E164)
	s.Equal("sinch", offered[0].Vendor)
	s.Equal(phone.Local, offered[0].Type)
	s.Equal([]phone.Capability{phone.SMS, phone.Voice}, offered[0].Capabilities)
	s.Equal(int64(1_200_000), offered[0].MonthlyCostMicros)
}

func (s *SinchSuite) TestSearchingAnywhereInTheNumberUsesTheOtherStrategy() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", Contains: "555"})
	s.Require().NoError(err)

	s.Equal("555", s.seen.query.Get("numberPattern.pattern"))
	s.Equal("CONTAINS", s.seen.query.Get("numberPattern.searchPattern"))
}

func (s *SinchSuite) TestAPrefixAndASubstringCannotBothBeAsked() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:  "US",
		Prefix:   "719",
		Contains: "555",
	})

	s.ErrorContains(err, "not both")
	s.Empty(s.seen.path)
}

func (s *SinchSuite) TestRentingIsAnActionOnTheNumberBeingBought() {
	s.answer(`{"phoneNumber":"+17195551234","regionCode":"US","type":"LOCAL",
		"capability":["VOICE"],"money":{"currencyCode":"USD","amount":"1.20"}}`)

	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234", Country: "US"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v1/projects/proj-1/availableNumbers/+17195551234:rent", s.seen.path)

	s.Equal("+17195551234", bought.E164)
	s.Equal("US", bought.Country)
	s.Equal([]phone.Capability{phone.Voice}, bought.Capabilities)
	s.Equal(int64(1_200_000), bought.MonthlyCostMicros)
}

func (s *SinchSuite) TestReleasingActsOnTheActiveNumberRatherThanTheAvailableOne() {
	// A number for sale and a number held are different resources at Sinch.
	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+17195551234"))

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v1/projects/proj-1/activeNumbers/+17195551234:release", s.seen.path)
}

func (s *SinchSuite) TestPointingANumberAtATrunkSaysItIsNotHere() {
	s.ErrorIs(s.provider.ConfigureInbound(s.ctx, phone.Inbound{}), phone.ErrNotImplemented)
	s.ErrorIs(s.provider.SendDigits(s.ctx, "call-1", "1"), phone.ErrNotImplemented)
}

func (s *SinchSuite) TestDiallingOutCarriesTheAnswerPlanInlineOnTheCallout() {
	// Sinch's callbacks belong to the application, not to a call, so a plan for one call
	// has to travel on the request.
	s.answer(`{"callId":"c-1"}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:          "+17195551234",
		To:            "+13035559876",
		Bridge:        phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"},
		InitialDigits: "1234#",
	})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/calling/v1/callouts", s.seen.path)
	// The calling API is the application's, not the project's, so no bearer token here.
	s.Equal("app-key", s.seen.user)

	var sent struct {
		Method string `json:"method"`
		Custom struct {
			CLI         string `json:"cli"`
			Destination struct {
				Type     string `json:"type"`
				Endpoint string `json:"endpoint"`
			} `json:"destination"`
			ACE  string `json:"ace"`
			DTMF string `json:"dtmf"`
		} `json:"customCallout"`
	}
	s.Require().NoError(json.Unmarshal([]byte(s.seen.body), &sent))
	s.Equal("customCallout", sent.Method)
	s.Equal("+17195551234", sent.Custom.CLI)
	s.Equal("number", sent.Custom.Destination.Type)
	s.Equal("+13035559876", sent.Custom.Destination.Endpoint)
	s.Equal("1234#", sent.Custom.DTMF)

	// The plan is a string of SVAML rather than an object, so it is read back as one.
	var plan struct {
		Action struct {
			Name        string `json:"name"`
			Destination struct {
				Endpoint string `json:"endpoint"`
			} `json:"destination"`
		} `json:"action"`
	}
	s.Require().NoError(json.Unmarshal([]byte(sent.Custom.ACE), &plan))
	s.Equal("connectSip", plan.Action.Name)
	// Sinch takes a SIP endpoint without the scheme on the front.
	s.Equal("trunk@sip.stream-io-api.com", plan.Action.Destination.Endpoint)

	s.Equal("c-1", placed.VendorCallID)
}

func (s *SinchSuite) TestACalloutSinchAcceptsButPlacesNothingForIsAnError() {
	s.answer(`{}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+17195551234",
		To:     "+13035559876",
		Bridge: phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"},
	})

	s.ErrorContains(err, "without a call id")
}

func (s *SinchSuite) TestDiallingWithoutTheApplicationCredentialsSaysWhichAreMissing() {
	s.T().Setenv(applicationKeyEnvVar, "")
	s.T().Setenv(applicationSecretEnvVar, "")
	provider, err := New(Options{
		ProjectID:      "proj-1",
		KeyID:          "key-1",
		KeySecret:      "secret",
		BaseURL:        s.server.URL,
		AuthURL:        s.server.URL + "/oauth2/token",
		CallingBaseURL: s.server.URL,
	})
	s.Require().NoError(err)

	_, err = provider.Dial(s.ctx, phone.Outbound{
		From:   "+17195551234",
		To:     "+13035559876",
		Bridge: phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"},
	})

	s.ErrorContains(err, applicationKeyEnvVar)
	s.Empty(s.seen.path)
}

func (s *SinchSuite) TestPlaceSearchesAreNotClaimed() {
	s.False(s.provider.Supports(phone.FilterAdministrativeArea))
	s.False(s.provider.Supports(phone.FilterLocality))
	s.True(s.provider.Supports(phone.FilterContains))
}

func (s *SinchSuite) TestAFailureFromSinchSaysWhatSinchSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"error":{"message":"region not supported"}}`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "ZZ"})

	s.ErrorContains(err, "422")
	s.ErrorContains(err, "region not supported")
}

func (s *SinchSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}
