package bandwidth

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

type BandwidthSuite struct {
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
	// body is the XML as sent, since that is what Bandwidth reads.
	body string
	auth string
}

func TestBandwidth(t *testing.T) { suite.Run(t, new(BandwidthSuite)) }

func (s *BandwidthSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.answer(`<SearchResult><ResultCount>0</ResultCount></SearchResult>`)

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		user, _, _ := r.BasicAuth()
		s.seen = request{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.Query(),
			body:   string(raw),
			auth:   user,
		}
		w.Header().Set("Content-Type", "application/xml")
		s.respond(w, r)
	}))

	provider, err := New(Options{
		AccountID:     "acc-1",
		Username:      "user",
		Password:      "pass",
		SiteID:        "site-7",
		ApplicationID: "app-9",
		BaseURL:       s.server.URL,
		VoiceBaseURL:  s.server.URL,
	})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *BandwidthSuite) TearDownTest() { s.server.Close() }

func (s *BandwidthSuite) TestCredentialsAreRequired() {
	s.T().Setenv("BANDWIDTH_ACCOUNT_ID", "")
	s.T().Setenv("BANDWIDTH_USERNAME", "")
	s.T().Setenv("BANDWIDTH_PASSWORD", "")

	_, err := New(Options{})

	s.ErrorContains(err, "BANDWIDTH_ACCOUNT_ID")
}

func (s *BandwidthSuite) TestSearchingAStateReturnsNumbersInE164WithWhereTheyAre() {
	// Bandwidth quotes ten digits with no country code, which is not what anything else
	// here uses.
	s.answer(`<SearchResult>
		<ResultCount>1</ResultCount>
		<TelephoneNumberDetailList>
			<TelephoneNumberDetail>
				<City>COLORADO SPGS</City>
				<State>CO</State>
				<RateCenter>CLRDOSPRGS</RateCenter>
				<FullNumber>7195551234</FullNumber>
			</TelephoneNumberDetail>
		</TelephoneNumberDetailList>
	</SearchResult>`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:            "US",
		AdministrativeArea: "co",
		Locality:           "Colorado Springs",
		Limit:              5,
	})
	s.Require().NoError(err)

	s.Equal("/api/accounts/acc-1/availableNumbers", s.seen.path)
	s.Equal("user", s.seen.auth)
	s.Equal("CO", s.seen.query.Get("state"))
	s.Equal("Colorado Springs", s.seen.query.Get("city"))
	s.Equal("5", s.seen.query.Get("quantity"))
	s.Equal("true", s.seen.query.Get("enableTNDetail"),
		"without detail a search by place cannot say where the numbers are")

	s.Require().Len(offered, 1)
	s.Equal("+17195551234", offered[0].E164)
	s.Equal("bandwidth", offered[0].Vendor)
	s.Equal("CO", offered[0].Region)
	s.Equal("COLORADO SPGS", offered[0].Locality)
	s.Equal(phone.Local, offered[0].Type)
}

func (s *BandwidthSuite) TestABareListOfNumbersIsStillRead() {
	// An account that does not return number detail should still answer a search rather
	// than look like it had no inventory.
	s.answer(`<SearchResult>
		<ResultCount>2</ResultCount>
		<TelephoneNumberList>
			<TelephoneNumber>7195551234</TelephoneNumber>
			<TelephoneNumber>7195555678</TelephoneNumber>
		</TelephoneNumberList>
	</SearchResult>`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", AreaCode: "719"})
	s.Require().NoError(err)

	s.Equal("719", s.seen.query.Get("areaCode"))
	s.Require().Len(offered, 2)
	s.Equal("+17195551234", offered[0].E164)
	s.Equal("+17195555678", offered[1].E164)
}

func (s *BandwidthSuite) TestSearchingOutsideNorthAmericaSaysWhichInventoryIsWrapped() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "GB", AreaCode: "20"})

	s.ErrorContains(err, "north american inventory")
	s.Empty(s.seen.path)
}

func (s *BandwidthSuite) TestSearchingNothingInParticularIsRejected() {
	// Bandwidth's search needs somewhere to look, so a bare country would return whatever
	// the account's default happens to be.
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	s.ErrorContains(err, "area code, a city or a state")
}

func (s *BandwidthSuite) TestSearchingACityNeedsTheStateItIsIn() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", Locality: "Springfield"})

	s.ErrorContains(err, "needs the state")
}

func (s *BandwidthSuite) TestOrderingNamesTheSiteTheNumberIsBilledTo() {
	s.answer(`<OrderResponse><Order><id>order-9</id></Order><OrderStatus>RECEIVED</OrderStatus></OrderResponse>`)

	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234", Country: "US"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/api/accounts/acc-1/orders", s.seen.path)
	s.Contains(s.seen.body, "<SiteId>site-7</SiteId>")
	s.Contains(s.seen.body, "<TelephoneNumber>7195551234</TelephoneNumber>",
		"bandwidth names a number without its country code")
	s.Contains(s.seen.body, "ExistingTelephoneNumberOrderType")

	s.Equal("+17195551234", bought.E164)
	s.Equal("order-9", bought.VendorID)
}

func (s *BandwidthSuite) TestOrderingWithoutASiteSaysWhatIsMissing() {
	provider, err := New(Options{
		AccountID: "acc-1",
		Username:  "user",
		Password:  "pass",
		BaseURL:   s.server.URL,
	})
	s.Require().NoError(err)

	_, err = provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234"})

	s.ErrorContains(err, "BANDWIDTH_SITE_ID")
	s.Empty(s.seen.path)
}

func (s *BandwidthSuite) TestReleasingIsAnotherOrderRatherThanADelete() {
	s.answer(`<DisconnectTelephoneNumberOrderResponse/>`)

	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+17195551234"))

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/api/accounts/acc-1/disconnects", s.seen.path)
	s.Contains(s.seen.body, "<TelephoneNumber>7195551234</TelephoneNumber>")
}

func (s *BandwidthSuite) TestPointingANumberAtATrunkSaysItIsNotHere() {
	s.ErrorIs(s.provider.ConfigureInbound(s.ctx, phone.Inbound{}), phone.ErrNotImplemented)
	s.ErrorIs(s.provider.SendDigits(s.ctx, "call-1", "1"), phone.ErrNotImplemented)
}

func (s *BandwidthSuite) TestDiallingOutNamesTheVoiceApplicationAndWhereToFetchThePlan() {
	s.answer(`{"callId":"c-1","state":"initiated"}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:        "+17195551234",
		To:          "+13035559876",
		Bridge:      phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"},
		RingTimeout: 25 * time.Second,
		AnswerURL:   "https://router.example.com/v1/phone/answer/tok-1",
	})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/api/v2/accounts/acc-1/calls", s.seen.path)
	s.Equal("user", s.seen.auth)

	var sent map[string]any
	s.Require().NoError(json.Unmarshal([]byte(s.seen.body), &sent))
	s.Equal("+13035559876", sent["to"])
	s.Equal("+17195551234", sent["from"])
	s.Equal("app-9", sent["applicationId"])
	s.Equal("https://router.example.com/v1/phone/answer/tok-1", sent["answerUrl"])
	s.Equal(float64(25), sent["callTimeout"])

	s.Equal("c-1", placed.VendorCallID)
	s.Equal("initiated", placed.Status)
}

func (s *BandwidthSuite) TestDiallingWithoutTheVoiceApplicationSaysWhichCredentialIsMissing() {
	s.T().Setenv(applicationIDEnvVar, "")
	provider, err := New(Options{
		AccountID:    "acc-1",
		Username:     "user",
		Password:     "pass",
		BaseURL:      s.server.URL,
		VoiceBaseURL: s.server.URL,
	})
	s.Require().NoError(err)

	_, err = provider.Dial(s.ctx, phone.Outbound{
		From:      "+17195551234",
		To:        "+13035559876",
		Bridge:    phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"},
		AnswerURL: "https://router.example.com/v1/phone/answer/tok-1",
	})

	s.ErrorContains(err, applicationIDEnvVar)
	s.Empty(s.seen.path)
}

func (s *BandwidthSuite) TestTheAnswerPlanPressesTheDigitsAtThePersonBeforeBridging() {
	// The BXML runs on the leg to the person, so SendDtmf reaches their keypad's far end
	// rather than the trunk's. That ordering is the whole point of hosting the plan.
	plan, err := s.provider.Answer(phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"}, "ww1234#")
	s.Require().NoError(err)

	s.Equal("application/xml", plan.ContentType)
	rendered := string(plan.Body)
	s.Less(strings.Index(rendered, "<SendDtmf>"), strings.Index(rendered, "<Transfer>"))
	s.Contains(rendered, "<SendDtmf>ww1234#</SendDtmf>")
	s.Contains(rendered, "<SipUri>sip:trunk@sip.stream-io-api.com</SipUri>")
}

func (s *BandwidthSuite) TestAnAnswerPlanWithNoDigitsPressesNothing() {
	plan, err := s.provider.Answer(phone.Bridge{URI: "sip:trunk@sip.stream-io-api.com"}, "")
	s.Require().NoError(err)

	s.NotContains(string(plan.Body), "SendDtmf")
}

func (s *BandwidthSuite) TestSearchingByPlaceIsClaimedAndDigitPatternsAreNot() {
	s.True(s.provider.Supports(phone.FilterAdministrativeArea))
	s.True(s.provider.Supports(phone.FilterLocality))
	s.False(s.provider.Supports(phone.FilterPrefix))
	s.False(s.provider.Supports(phone.FilterContains))
}

func (s *BandwidthSuite) TestAFailureFromBandwidthSaysWhatBandwidthSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`<ResponseSelectWrapper><Error><Description>invalid area code</Description></Error></ResponseSelectWrapper>`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", AreaCode: "000"})

	s.ErrorContains(err, "400")
	s.ErrorContains(err, "invalid area code")
}

func (s *BandwidthSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}
