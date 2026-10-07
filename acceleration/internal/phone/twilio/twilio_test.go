package twilio

import (
	"context"
	"encoding/xml"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

type TwilioSuite struct {
	suite.Suite
	ctx      context.Context
	server   *httptest.Server
	provider *Provider

	// seen is what the last request asked for, which is how these tests check that what
	// the caller wanted arrived in Twilio's own shape.
	seen request
	// seenPricing is the last request to the pricing host, kept apart so a search's own
	// request is still the one in seen.
	seenPricing request
	// pricingCalls counts requests to the pricing host.
	pricingCalls int
	// respond answers the next request.
	respond func(w http.ResponseWriter, r *http.Request)
}

// request is what a handler recorded about a call.
type request struct {
	method string
	path   string
	query  url.Values
	form   url.Values
	user   string
	pass   string
}

func TestTwilio(t *testing.T) { suite.Run(t, new(TwilioSuite)) }

func (s *TwilioSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.seenPricing = request{}
	s.pricingCalls = 0
	s.respond = func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) }

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		user, pass, _ := r.BasicAuth()
		got := request{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.Query(),
			form:   r.PostForm,
			user:   user,
			pass:   pass,
		}
		if strings.HasPrefix(r.URL.Path, "/v1/PhoneNumbers/Countries/") {
			s.pricingCalls++
			s.seenPricing = got
		} else {
			s.seen = got
		}
		w.Header().Set("Content-Type", "application/json")
		s.respond(w, r)
	}))

	provider, err := New(Options{
		AccountSID: "AC123",
		AuthToken:  "secret",
		BaseURL:    s.server.URL,
		PricingURL: s.server.URL,
	})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *TwilioSuite) TearDownTest() { s.server.Close() }

func (s *TwilioSuite) TestCredentialsAreRequired() {
	s.T().Setenv("TWILIO_ACCOUNT_SID", "")
	s.T().Setenv("TWILIO_AUTH_TOKEN", "")

	_, err := New(Options{})

	s.ErrorContains(err, "TWILIO_ACCOUNT_SID")
}

func (s *TwilioSuite) TestSearchingAsksForTheCountryAndReturnsWhatIsOffered() {
	s.answer(`{"available_phone_numbers":[
		{"phone_number":"+15125551234","iso_country":"US","region":"TX","locality":"Austin",
		 "capabilities":{"voice":true,"SMS":true,"MMS":false,"fax":false}}]}`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:      "us",
		AreaCode:     "512",
		Limit:        5,
		Capabilities: []phone.Capability{phone.Voice},
	})
	s.Require().NoError(err)

	s.Equal("/2010-04-01/Accounts/AC123/AvailablePhoneNumbers/US/Local.json", s.seen.path)
	s.Equal("512", s.seen.query.Get("AreaCode"))
	s.Equal("5", s.seen.query.Get("PageSize"))
	s.Equal("true", s.seen.query.Get("VoiceEnabled"))
	s.Equal("AC123", s.seen.user)
	s.Equal("secret", s.seen.pass)

	s.Require().Len(offered, 1)
	s.Equal("+15125551234", offered[0].E164)
	s.Equal("Austin", offered[0].Locality)
	s.Equal([]phone.Capability{phone.Voice, phone.SMS}, offered[0].Capabilities)
}

const usPrices = `{"iso_country":"US","price_unit":"USD","phone_number_prices":[
	{"number_type":"local","base_price":"1.15","current_price":"1.00"},
	{"number_type":"toll free","base_price":"2.15","current_price":"2.00"}]}`

const oneNumber = `{"available_phone_numbers":[{"phone_number":"+15125550100","iso_country":"US",
	"capabilities":{"voice":true}}]}`

func (s *TwilioSuite) TestSearchingQuotesTheCurrentPriceForTheTypeSearched() {
	s.answerWithPrices(oneNumber, usPrices)

	local, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "us"})
	s.Require().NoError(err)
	tollFree, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", Type: phone.TollFree})
	s.Require().NoError(err)

	s.Equal(int64(1_000_000), local[0].MonthlyCostMicros, "current_price, which has the account's discounts")
	s.Equal(int64(2_000_000), tollFree[0].MonthlyCostMicros, `twilio names this type "toll free", with a space`)
	s.Equal(1, s.pricingCalls, "a country's prices are asked for once")
}

func (s *TwilioSuite) TestPricesAreAskedForOnTheCountryWithTheAccountsCredentials() {
	s.answerWithPrices(oneNumber, usPrices)

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "us"})
	s.Require().NoError(err)

	s.Equal("/v1/PhoneNumbers/Countries/US", s.seenPricing.path)
	s.Equal("AC123", s.seenPricing.user)
	s.Equal("secret", s.seenPricing.pass)
}

func (s *TwilioSuite) TestSearchingStillReturnsNumbersWhenPricesCannotBeHad() {
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/") {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(oneNumber))
	}

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})
	s.Require().NoError(err)

	s.Require().Len(offered, 1)
	s.Zero(offered[0].MonthlyCostMicros)
}

func (s *TwilioSuite) TestAPriceInAnotherCurrencyIsNotQuotedAsDollars() {
	s.answerWithPrices(oneNumber, `{"price_unit":"EUR","phone_number_prices":[
		{"number_type":"local","current_price":"1.00"}]}`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})
	s.Require().NoError(err)

	s.Zero(offered[0].MonthlyCostMicros)
}

func (s *TwilioSuite) TestATypeTwilioHasNoPriceForIsNotQuoted() {
	s.answerWithPrices(oneNumber, usPrices)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", Type: phone.Mobile})
	s.Require().NoError(err)

	s.Zero(offered[0].MonthlyCostMicros)
}

func (s *TwilioSuite) TestSearchingWithoutACountryIsRejectedBeforeAnyCall() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{})

	s.ErrorContains(err, "country is required")
	s.Empty(s.seen.path, "nothing was asked of twilio")
}

func (s *TwilioSuite) TestBuyingANumberReturnsWhatIsNeededToManageItLater() {
	s.answer(`{"sid":"PN9","phone_number":"+15125551234","iso_country":"US",
		"monthly_price":"1.15","capabilities":{"voice":true,"SMS":true}}`)

	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+15125551234"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/2010-04-01/Accounts/AC123/IncomingPhoneNumbers.json", s.seen.path)
	s.Equal("+15125551234", s.seen.form.Get("PhoneNumber"))

	s.Equal("PN9", bought.VendorID, "the sid is what releasing it later needs")
	s.Equal("twilio", bought.Vendor)
	s.Equal(int64(1_150_000), bought.MonthlyCostMicros)
}

func (s *TwilioSuite) TestReleasingANumberLooksUpItsSidFirst() {
	var paths []string
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		_, _ = w.Write([]byte(`{"incoming_phone_numbers":[{"sid":"PN9","phone_number":"+15125551234"}]}`))
	}

	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+15125551234"))

	s.Equal([]string{
		"GET /2010-04-01/Accounts/AC123/IncomingPhoneNumbers.json",
		"DELETE /2010-04-01/Accounts/AC123/IncomingPhoneNumbers/PN9.json",
	}, paths)
}

func (s *TwilioSuite) TestANumberThisAccountDoesNotOwnCannotBeReleased() {
	s.answer(`{"incoming_phone_numbers":[]}`)

	err := s.provider.ReleaseNumber(s.ctx, "+15125551234")

	s.ErrorContains(err, "not one of this account's numbers")
}

func (s *TwilioSuite) TestConfiguringInboundPointsTheNumberAtTheBridge() {
	s.answer(`{"incoming_phone_numbers":[{"sid":"PN9","phone_number":"+15125551234"}],
		"sid":"PN9","phone_number":"+15125551234"}`)

	err := s.provider.ConfigureInbound(s.ctx, phone.Inbound{
		E164: "+15125551234",
		Bridge: phone.Bridge{
			URI:      "sip:trunk-7@sip.stream-io-api.com",
			Username: "agent",
			Password: "hunter2",
		},
	})
	s.Require().NoError(err)

	voiceURL := s.seen.form.Get("VoiceUrl")
	s.Require().NotEmpty(voiceURL)
	instructions := s.twiml(voiceURL)
	s.Contains(instructions, "sip:trunk-7@sip.stream-io-api.com")
	s.Contains(instructions, `username="agent"`)
	s.Contains(instructions, `password="hunter2"`)
}

func (s *TwilioSuite) TestInboundIsRefusedWithoutASipBridge() {
	err := s.provider.ConfigureInbound(s.ctx, phone.Inbound{
		E164:   "+15125551234",
		Bridge: phone.Bridge{URI: "https://example.com/hook"},
	})

	s.ErrorContains(err, "not a sip uri")
	s.Empty(s.seen.path, "a bad bridge is caught before the number is looked up")
}

func (s *TwilioSuite) TestDiallingOutBridgesTheAnsweredCallIntoTheTrunk() {
	s.answer(`{"sid":"CA1","status":"queued"}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125551234",
		To:     "+15550001111",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})
	s.Require().NoError(err)

	s.Equal("/2010-04-01/Accounts/AC123/Calls.json", s.seen.path)
	s.Equal("+15550001111", s.seen.form.Get("To"))
	s.Contains(s.seen.form.Get("Twiml"), "<Sip>sip:trunk-7@sip.stream-io-api.com</Sip>")
	s.Equal("CA1", placed.VendorCallID)
	s.Equal("queued", placed.Status)
}

func (s *TwilioSuite) TestTheRingTimeoutAndInitialDigitsRideOnTheCallRatherThanTheTwiml() {
	// Both concern the leg to the person: Timeout is how long they are rung for, and
	// SendDigits is pressed at them once they answer. The TwiML runs after that and only
	// says where to bridge to.
	s.answer(`{"sid":"CA1","status":"queued"}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:          "+15125551234",
		To:            "+15550001111",
		Bridge:        phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
		RingTimeout:   25 * time.Second,
		InitialDigits: "ww1234#",
	})
	s.Require().NoError(err)

	s.Equal("25", s.seen.form.Get("Timeout"))
	s.Equal("ww1234#", s.seen.form.Get("SendDigits"))
	s.NotContains(s.seen.form.Get("Twiml"), "1234", "the digits are pressed at the person, not the trunk")
}

func (s *TwilioSuite) TestACallWithNoTermsAsksTwilioForNoneOfThem() {
	// An unset ring timeout has to leave Twilio's own default rather than becoming zero,
	// which would be a call that gives up before it rings.
	s.answer(`{"sid":"CA1","status":"queued"}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125551234",
		To:     "+15550001111",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})
	s.Require().NoError(err)

	s.False(s.seen.form.Has("Timeout"))
	s.False(s.seen.form.Has("SendDigits"))
}

func (s *TwilioSuite) TestRingingForLongerThanTwilioAllowsIsRefusedBeforeItIsAsked() {
	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:        "+15125551234",
		To:          "+15550001111",
		Bridge:      phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
		RingTimeout: 20 * time.Minute,
	})

	s.ErrorContains(err, "600s twilio will ring for")
	s.Empty(s.seen.path, "a call twilio would reject is not sent")
}

func (s *TwilioSuite) TestPressingDigitsIsRefusedRatherThanEndingTheCall() {
	// The only way to make tones at Twilio is to replace the TwiML the leg is running,
	// which is the <Dial> holding the agent on the call. Refusing is the honest answer.
	err := s.provider.SendDigits(s.ctx, "CA1", "1")

	s.ErrorIs(err, phone.ErrNotImplemented)
	s.ErrorContains(err, "without ending the call")
}

func (s *TwilioSuite) TestACallWithoutBothEndsIsRejected() {
	_, err := s.provider.Dial(s.ctx, phone.Outbound{From: "+15125551234"})

	s.ErrorContains(err, "from and a to")
}

func (s *TwilioSuite) TestAFailureFromTwilioSaysWhatTwilioSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"message":"Authenticate"}`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	s.ErrorContains(err, "401")
	s.ErrorContains(err, "Authenticate")
}

func (s *TwilioSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}

// answerWithPrices answers a number search with numbers and the pricing call with prices.
func (s *TwilioSuite) answerWithPrices(numbers, prices string) {
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/PhoneNumbers/Countries/") {
			_, _ = w.Write([]byte(prices))
			return
		}
		_, _ = w.Write([]byte(numbers))
	}
}

// twiml pulls the instructions back out of the echo url they were configured through.
func (s *TwilioSuite) twiml(voiceURL string) string {
	parsed, err := url.Parse(voiceURL)
	s.Require().NoError(err)
	return parsed.Query().Get("Twiml")
}

// TestTheInstructionsTwilioIsGivenAreValidXML matters because a malformed one would only
// be discovered when somebody called the number.
func (s *TwilioSuite) TestTheInstructionsTwilioIsGivenAreValidXML() {
	instructions, err := dialBridge(phone.Bridge{URI: "sip:trunk@sip.example.com", Username: "u"})
	s.Require().NoError(err)
	s.True(strings.HasPrefix(instructions, "<?xml"), "twiml needs a declaration")

	var parsed twiml
	s.Require().NoError(xml.Unmarshal([]byte(instructions), &parsed))
	s.Equal("sip:trunk@sip.example.com", parsed.Dial.SIP.URI)
	s.Equal("u", parsed.Dial.SIP.Username)
}
