package telnyx

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

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
		AccountSID:   "acct-1",
		ConnectionID: "conn-1",
		BaseURL:      s.server.URL,
	})
	s.Require().NoError(err)
	provider.lookupPause = time.Millisecond
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
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(`{"data":[{"phone_number":"+15125551234",
				"cost_information":{"monthly_cost":"1.00000"}}]}`))
			return
		}
		_, _ = w.Write([]byte(`{"data":{"id":"order-9","phone_numbers":[
			{"phone_number":"+15125551234","country_code":"US","features":[{"name":"voice"}]}]}}`))
	}

	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+15125551234"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v2/number_orders", s.seen.path)
	s.Equal("conn-1", s.seen.body["connection_id"])

	s.Equal("order-9", bought.VendorID)
	s.Equal("telnyx", bought.Vendor)
	s.Equal("US", bought.Country)
	s.Equal([]phone.Capability{phone.Voice}, bought.Capabilities)
	s.Equal(int64(1_000_000), bought.MonthlyCostMicros, "the order carries no price, so it is the offer's")
}

func (s *TelnyxSuite) TestANumberNoLongerForSaleIsNotOrdered() {
	s.answer(`{"data":[]}`)

	_, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+15125551234"})

	s.ErrorContains(err, "not for sale")
	s.Equal(http.MethodGet, s.seen.method)
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

func (s *TelnyxSuite) TestDiallingOutStartsATeXMLCallThatDialsTheTrunk() {
	s.answer(`{"call_sid":"v3:cc-1","from":"+15125550100","to":"+15125550123","status":"queued"}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From: "+15125550100",
		To:   "+15125550123",
		Bridge: phone.Bridge{
			URI:      "sip:+15125550100@app.sip.example.com",
			Username: "u",
			Password: "p",
		},
	})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v2/texml/Accounts/acct-1/Calls", s.seen.path)
	s.Equal("conn-1", s.seen.body["ApplicationSid"])
	s.Equal("+15125550100", s.seen.body["From"])
	s.Equal("+15125550123", s.seen.body["To"])
	s.NotContains(s.seen.body, "link_to")
	s.NotContains(s.seen.body, "Timeout", "no timeout leaves Telnyx's own default")
	s.NotContains(s.seen.body, "CustomHeaders")

	texml := s.texml()
	s.Equal("sip:+15125550100@app.sip.example.com", texml.Dial.Sip.URI)
	s.Equal("u", texml.Dial.Sip.Username)
	s.Equal("p", texml.Dial.Sip.Password)

	s.Equal("v3:cc-1", placed.VendorCallID)
	s.Equal("queued", placed.Status)
}

func (s *TelnyxSuite) TestCredentialsWithXMLSpecialCharactersAreEscaped() {
	s.answer(`{"call_sid":"v3:cc-1","status":"queued"}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From: "+15125550100",
		To:   "+15125550123",
		Bridge: phone.Bridge{
			URI:      "sip:+15125550100@app.sip.example.com;a=1&b=<2>",
			Username: `u"&<`,
			Password: `a&b<"c`,
		},
	})
	s.Require().NoError(err)

	texml := s.texml()
	s.Equal("sip:+15125550100@app.sip.example.com;a=1&b=<2>", texml.Dial.Sip.URI)
	s.Equal(`u"&<`, texml.Dial.Sip.Username)
	s.Equal(`a&b<"c`, texml.Dial.Sip.Password)
}

func (s *TelnyxSuite) TestTermsTelnyxIsKnownToCarryAreTheOnesItDials() {
	s.True(s.provider.Dials(phone.FeatureRingTimeout))
	s.True(s.provider.Dials(phone.FeatureInitialDigits))
	// TeXML's CustomHeaders are documented as HTTP headers, so they may never reach the
	// person's SIP leg.
	s.False(s.provider.Dials(phone.FeatureCustomHeaders))
}

func (s *TelnyxSuite) TestTheRingTimeoutIsCarriedOnTheCall() {
	s.answer(`{"call_sid":"v3:cc-1","status":"queued"}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:        "+15125550100",
		To:          "+15125550123",
		Bridge:      phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"},
		RingTimeout: 25 * time.Second,
	})
	s.Require().NoError(err)

	s.Equal(float64(25), s.seen.body["Timeout"])
	s.Empty(s.texml().Dial.Timeout, "the person's ring timeout is not the trunk's")
}

// Telnyx's answer to the account's TeXML call names no call, so the call is looked up in
// the account's call list right after it is placed.
func (s *TelnyxSuite) TestACallWhoseIDTelnyxDoesNotSayIsLookedUp() {
	before := time.Now().Add(-time.Hour).UTC().Format(time.RFC1123Z)
	now := time.Now().UTC().Format(time.RFC1123Z)
	var listed request
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_, _ = w.Write([]byte(`{"from":"+15125550100","to":"+15125550123","status":"queued"}`))
			return
		}
		listed = s.seen
		_, _ = w.Write([]byte(`{"calls":[
			{"sid":"v3:old","status":"completed","start_time":"` + before + `","date_created":"` + before + `"},
			{"sid":"v3:new","status":"ringing","start_time":"` + now + `","date_created":"` + now + `"}]}`))
	}

	placed, err := s.provider.Dial(s.ctx, s.outbound())
	s.Require().NoError(err)

	s.Equal(http.MethodGet, listed.method)
	s.Equal("/v2/texml/Accounts/acct-1/Calls", listed.path)
	s.Equal("+15125550123", listed.query.Get("To"))
	s.Equal("+15125550100", listed.query.Get("From"))
	s.Equal("v3:new", placed.VendorCallID)
	s.Equal("queued", placed.Status)
}

func (s *TelnyxSuite) TestACallStillMissingFromTheListIsPlacedWithoutAnID() {
	logged := s.logged()
	lists := 0
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_, _ = w.Write([]byte(`{"from":"+15125550100","to":"+15125550123","status":"queued"}`))
			return
		}
		lists++
		_, _ = w.Write([]byte(`{"calls":[],"page":0,"page_size":20}`))
	}

	placed, err := s.provider.Dial(s.ctx, s.outbound())
	s.Require().NoError(err)

	s.Equal(2, lists, "the list is eventually consistent, so it is asked once more")
	s.Empty(placed.VendorCallID)
	s.Equal("queued", placed.Status)
	s.unknownIDWarned(logged)
}

func (s *TelnyxSuite) TestACallTheListOnlyShowsBeforeThisOneIsNotTaken() {
	before := time.Now().Add(-time.Hour).UTC().Format(time.RFC1123Z)
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_, _ = w.Write([]byte(`{"status":"queued"}`))
			return
		}
		_, _ = w.Write([]byte(`{"calls":[{"sid":"v3:old","status":"in-progress","start_time":"` + before + `"}]}`))
	}

	placed, err := s.provider.Dial(s.ctx, s.outbound())
	s.Require().NoError(err)

	s.Empty(placed.VendorCallID)
}

func (s *TelnyxSuite) TestACallWhoseLookupFailsIsStillPlaced() {
	logged := s.logged()
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_, _ = w.Write([]byte(`{"from":"+15125550100","to":"+15125550123","status":"queued"}`))
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"errors":[{"detail":"boom"}]}`))
	}

	placed, err := s.provider.Dial(s.ctx, s.outbound())
	s.Require().NoError(err)

	s.Empty(placed.VendorCallID)
	s.Equal("queued", placed.Status)
	s.unknownIDWarned(logged)
}

// The README promises a status for every vendor's call, and Telnyx's reference does not
// show one on this answer.
func (s *TelnyxSuite) TestACallPlacedWithNoStatusIsQueued() {
	s.answer(`{"call_sid":"v3:cc-1"}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From: "+15125550100",
		To:   "+15125550123",
		Bridge: phone.Bridge{
			URI:      "sip:+15125550100@app.sip.example.com",
			Username: "u",
			Password: "p",
		},
	})
	s.Require().NoError(err)

	s.Equal("queued", placed.Status)
}

func (s *TelnyxSuite) TestDiallingRefusesTermsItCannotSend() {
	bridge := phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"}
	for refusal, outbound := range map[string]phone.Outbound{
		"telnyx: telnyx rings for 5 to 120 seconds, not 3":   {RingTimeout: 3 * time.Second},
		"telnyx: telnyx rings for 5 to 120 seconds, not 121": {RingTimeout: 121 * time.Second},
		`telnyx: TeXML plays only "0123456789*#w", not "W1"`: {InitialDigits: "W1"},
		`telnyx: TeXML plays only "0123456789*#w", not "1A"`: {InitialDigits: "1A"},
		"telnyx: custom SIP headers are not known to reach the person on a TeXML call": {
			Headers: map[string]string{"X-Ticket": "42"},
		},
	} {
		outbound.From, outbound.To, outbound.Bridge = "+15125550100", "+15125550123", bridge
		s.seen = request{}

		_, err := s.provider.Dial(s.ctx, outbound)

		s.EqualError(err, refusal)
		s.Empty(s.seen.path, refusal+" is refused before anything is sent")
	}
}

func (s *TelnyxSuite) TestDigitsTeXMLCannotPlayAreRefusedBeforeTheCallIsSetUp() {
	// Unsupported is what the service checks before it creates a Stream trunk.
	s.Equal([]phone.CallFeature{phone.FeatureInitialDigits},
		phone.Outbound{InitialDigits: "W1"}.Unsupported(s.provider))
	s.Equal([]phone.CallFeature{phone.FeatureInitialDigits},
		phone.Outbound{InitialDigits: "1D"}.Unsupported(s.provider))
	s.Empty(phone.Outbound{InitialDigits: "ww1234#*"}.Unsupported(s.provider))
}

func (s *TelnyxSuite) TestInitialDigitsArePlayedBeforeTheTrunkIsDialled() {
	s.answer(`{"call_sid":"v3:cc-1","status":"queued"}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:          "+15125550100",
		To:            "+15125550123",
		Bridge:        phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"},
		InitialDigits: "ww1234#",
	})
	s.Require().NoError(err)

	raw, _ := s.seen.body["Texml"].(string)
	play, dial := strings.Index(raw, "<Play"), strings.Index(raw, "<Dial")
	s.Require().NotEqual(-1, play, raw)
	s.Less(play, dial, "the menu is answered before the agent joins")
	s.Equal("ww1234#", s.texml().Play.Digits)
}

func (s *TelnyxSuite) TestACallWithNoTermsOnlyDialsTheTrunk() {
	s.answer(`{"call_sid":"v3:cc-1","status":"queued"}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125550100",
		To:     "+15125550123",
		Bridge: phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"},
	})
	s.Require().NoError(err)

	raw, _ := s.seen.body["Texml"].(string)
	s.NotContains(raw, "<Play")
	s.NotContains(raw, "timeout=")
	s.NotContains(raw, "username=")
	s.NotContains(raw, "password=")
}

func (s *TelnyxSuite) TestDiallingWithoutAConnectionIDIsRefused() {
	provider, err := New(Options{APIKey: "KEY123", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125550100",
		To:     "+15125550123",
		Bridge: phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"},
	})

	s.EqualError(err, "telnyx: a connection id is required to place a call")
	s.Empty(s.seen.path)
}

func (s *TelnyxSuite) TestDiallingWithoutAnAccountSIDIsRefused() {
	s.T().Setenv("TELNYX_ACCOUNT_SID", "")
	provider, err := New(Options{APIKey: "KEY123", ConnectionID: "conn-1", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125550100",
		To:     "+15125550123",
		Bridge: phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"},
	})

	s.EqualError(err, "telnyx: TELNYX_ACCOUNT_SID is required to place a call")
	s.Empty(s.seen.path)
}

func (s *TelnyxSuite) TestTheAccountSIDFallsBackToTheEnvironment() {
	s.T().Setenv("TELNYX_ACCOUNT_SID", "acct-env")
	provider, err := New(Options{APIKey: "KEY123", ConnectionID: "conn-1", BaseURL: s.server.URL})
	s.Require().NoError(err)
	s.answer(`{"call_sid":"v3:cc-1","status":"queued"}`)

	_, err = provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125550100",
		To:     "+15125550123",
		Bridge: phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"},
	})
	s.Require().NoError(err)

	s.Equal("/v2/texml/Accounts/acct-env/Calls", s.seen.path)
}

func (s *TelnyxSuite) TestTheTrunkRingsForAsLongAsAsked() {
	raw, err := dialTrunk(phone.Bridge{URI: "sip:+15125550100@app.sip.example.com"}, 25)
	s.Require().NoError(err)

	var parsed parsedTexml
	s.Require().NoError(xml.Unmarshal(raw, &parsed))
	s.Equal("25", parsed.Dial.Timeout)
	s.True(strings.HasPrefix(string(raw), "<?xml"))
}

func (s *TelnyxSuite) TestAnInboundCallIsSentToTheNumbersTrunk() {
	plan, err := RenderInbound(phone.Bridge{
		URI: "sip:+15125550100@app.sip.example.com", Username: "trunk-1", Password: "s3cret",
	})
	s.Require().NoError(err)

	s.Equal("application/xml", plan.ContentType)
	var parsed parsedTexml
	s.Require().NoError(xml.Unmarshal(plan.Body, &parsed))
	s.Equal("sip:+15125550100@app.sip.example.com", parsed.Dial.Sip.URI)
	s.Equal("trunk-1", parsed.Dial.Sip.Username)
	s.Equal("s3cret", parsed.Dial.Sip.Password)
	s.Empty(parsed.Dial.Timeout, "Telnyx's own ring time")
}

func (s *TelnyxSuite) TestAnInboundCallWithNoTrunkIsNotRendered() {
	_, err := RenderInbound(phone.Bridge{})

	s.Error(err)
}

func (s *TelnyxSuite) TestHangingUpIsTeXMLThatEndsTheCall() {
	plan := RenderHangup()

	s.Equal("application/xml", plan.ContentType)
	s.Equal(`<?xml version="1.0" encoding="UTF-8"?>`+"\n"+`<Response><Hangup/></Response>`, string(plan.Body))
}

func (s *TelnyxSuite) TestATrunkWithoutAnAddressIsNotDialled() {
	_, err := dialTrunk(phone.Bridge{}, 0)

	s.ErrorContains(err, "bridge uri is required")
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

func (s *TelnyxSuite) TestARefusalIsAVendorErrorWithTelnyxsCode() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"errors":[{"code":"10015","title":"Invalid value","detail":"no such number"}]}`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	vendorErr, ok := errors.AsType[*phone.VendorError](err)
	s.Require().True(ok)
	s.Equal("telnyx", vendorErr.Vendor)
	s.Equal(http.StatusUnprocessableEntity, vendorErr.Status)
	s.Equal("10015", vendorErr.Code)
	s.Equal("no such number", vendorErr.Message)
}

func (s *TelnyxSuite) TestTelnyxNotAnsweringIsAVendorErrorToo() {
	s.server.Close()

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	vendorErr, ok := errors.AsType[*phone.VendorError](err)
	s.Require().True(ok)
	s.Zero(vendorErr.Status)
	s.Error(vendorErr.Cause)
}

// parsedTexml is the TeXML a call was placed with, read back the way Telnyx reads it.
type parsedTexml struct {
	Play struct {
		Digits string `xml:"digits,attr"`
	} `xml:"Play"`
	Dial struct {
		Timeout string `xml:"timeout,attr"`
		Sip     struct {
			URI      string `xml:",chardata"`
			Username string `xml:"username,attr"`
			Password string `xml:"password,attr"`
		} `xml:"Sip"`
	} `xml:"Dial"`
}

func (s *TelnyxSuite) texml() parsedTexml {
	raw, ok := s.seen.body["Texml"].(string)
	s.Require().True(ok, "the call carries its TeXML inline")
	var parsed parsedTexml
	s.Require().NoError(xml.Unmarshal([]byte(raw), &parsed))
	return parsed
}

func (s *TelnyxSuite) outbound() phone.Outbound {
	return phone.Outbound{
		From: "+15125550100",
		To:   "+15125550123",
		Bridge: phone.Bridge{
			URI:      "sip:+15125550100@app.sip.example.com",
			Username: "u",
			Password: "secret-p",
		},
	}
}

// logged sends the provider's log to a buffer the test reads.
func (s *TelnyxSuite) logged() *bytes.Buffer {
	var logged bytes.Buffer
	s.provider.logger = slog.New(slog.NewTextHandler(&logged, nil))
	return &logged
}

func (s *TelnyxSuite) unknownIDWarned(logged *bytes.Buffer) {
	s.Contains(logged.String(), "level=WARN")
	s.Contains(logged.String(), "vendor call id is unknown")
	s.NotContains(logged.String(), "5550100")
	s.NotContains(logged.String(), "5550123")
	s.NotContains(logged.String(), "secret-p")
	s.NotContains(logged.String(), "acct-1")
	s.NotContains(logged.String(), "KEY123")
}

func (s *TelnyxSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}

func (s *TelnyxSuite) TestAnAnswerThatIsNotJSONIsAVendorErrorWithoutTheBody() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("<html>not json</html>")) }

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	vendorErr, ok := errors.AsType[*phone.VendorError](err)
	s.Require().True(ok)
	s.Equal("could not read the answer", vendorErr.Message)
	s.Equal(http.StatusOK, vendorErr.Status)
}
