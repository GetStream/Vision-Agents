package vonage

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
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

type VonageSuite struct {
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
	form   url.Values
	auth   string
	// body is the JSON the voice API is sent, which the number API never uses.
	body map[string]any
	// bearer is the voice API's token, unverified here beyond being present.
	bearer string
}

func TestVonage(t *testing.T) { suite.Run(t, new(VonageSuite)) }

func (s *VonageSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.answer(`{"numbers":[]}`)

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		form, _ := url.ParseQuery(string(raw))
		user, _, _ := r.BasicAuth()
		body := map[string]any{}
		_ = json.Unmarshal(raw, &body)
		s.seen = request{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.Query(),
			form:   form,
			auth:   user,
			body:   body,
			bearer: strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "),
		}
		w.Header().Set("Content-Type", "application/json")
		s.respond(w, r)
	}))

	provider, err := New(Options{
		APIKey:        "key",
		APISecret:     "secret",
		BaseURL:       s.server.URL,
		VoiceBaseURL:  s.server.URL,
		ApplicationID: "app-1",
		PrivateKey:    testKeyPEM(s.T()),
	})
	s.Require().NoError(err)
	s.provider = provider
}

// testKeyPEM makes a throwaway application key, so no real one has to live in the repo.
func testKeyPEM(t *testing.T) string {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	return string(pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	}))
}

func (s *VonageSuite) TearDownTest() { s.server.Close() }

func (s *VonageSuite) TestCredentialsAreRequired() {
	s.T().Setenv("VONAGE_API_KEY", "")
	s.T().Setenv("VONAGE_API_SECRET", "")

	_, err := New(Options{})

	s.ErrorContains(err, "VONAGE_API_KEY")
}

func (s *VonageSuite) TestSearchingAnchorsAPrefixAndReturnsWhatIsOffered() {
	s.answer(`{"count":1,"numbers":[{
		"country":"US","msisdn":"17195551234","type":"landline","cost":"0.90",
		"features":["VOICE","SMS"]}]}`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:      "us",
		Prefix:       "719",
		Type:         phone.Local,
		Capabilities: []phone.Capability{phone.Voice},
		Limit:        4,
	})
	s.Require().NoError(err)

	s.Equal("/number/search", s.seen.path)
	s.Equal("key", s.seen.auth)
	s.Equal("US", s.seen.query.Get("country"))
	s.Equal("719", s.seen.query.Get("pattern"))
	s.Equal("0", s.seen.query.Get("search_pattern"), "zero is how Vonage spells starts-with")
	s.Equal("landline", s.seen.query.Get("type"))
	s.Equal("VOICE", s.seen.query.Get("features"))
	s.Equal("4", s.seen.query.Get("size"))

	s.Require().Len(offered, 1)
	s.Equal("+17195551234", offered[0].E164)
	s.Equal("vonage", offered[0].Vendor)
	s.Equal(phone.Local, offered[0].Type)
	s.Equal([]phone.Capability{phone.Voice, phone.SMS}, offered[0].Capabilities)
	s.Equal(int64(900_000), offered[0].MonthlyCostMicros)
}

func (s *VonageSuite) TestSearchingAnywhereInTheNumberUsesTheOtherStrategy() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", Contains: "555"})
	s.Require().NoError(err)

	s.Equal("555", s.seen.query.Get("pattern"))
	s.Equal("1", s.seen.query.Get("search_pattern"))
}

func (s *VonageSuite) TestAPrefixAndASubstringCannotBothBeAsked() {
	// Vonage matches one pattern one way, so asking for both would silently drop one.
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:  "US",
		Prefix:   "719",
		Contains: "555",
	})

	s.ErrorContains(err, "not both")
	s.Empty(s.seen.path)
}

func (s *VonageSuite) TestBuyingNeedsTheCountryTheNumberIsSoldIn() {
	_, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234"})

	s.ErrorContains(err, "needs the country")
	s.Empty(s.seen.path)
}

func (s *VonageSuite) TestBuyingPostsTheNumberWithoutItsPlus() {
	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234", Country: "us"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/number/buy", s.seen.path)
	s.Equal("17195551234", s.seen.form.Get("msisdn"))
	s.Equal("US", s.seen.form.Get("country"))

	s.Equal("+17195551234", bought.E164)
	s.Equal("US", bought.Country)
}

func (s *VonageSuite) TestReleasingLooksUpWhichCountryTheNumberWasSoldIn() {
	// Cancelling names a country, and only the numbers this account holds know theirs.
	var paths []string
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		_, _ = w.Write([]byte(`{"numbers":[{"country":"US","msisdn":"17195551234"}]}`))
	}

	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+17195551234"))

	s.Equal([]string{"GET /account/numbers", "POST /number/cancel"}, paths)
	s.Equal("US", s.seen.form.Get("country"))
}

func (s *VonageSuite) TestANumberThisAccountDoesNotHoldCannotBeReleased() {
	s.answer(`{"numbers":[]}`)

	err := s.provider.ReleaseNumber(s.ctx, "+17195551234")

	s.ErrorContains(err, "not one of this account's numbers")
}

func (s *VonageSuite) TestPointingANumberAtATrunkSaysItIsNotHere() {
	s.ErrorIs(s.provider.ConfigureInbound(s.ctx, phone.Inbound{}), phone.ErrNotImplemented)
	s.ErrorIs(s.provider.SendDigits(s.ctx, "call-1", "1"), phone.ErrNotImplemented)
}

func (s *VonageSuite) TestDiallingOutCarriesTheCallPlanWithTheCall() {
	// Nothing is hosted to answer this call: the NCCO that connects the answered leg to
	// the trunk travels on the request that places it.
	s.answer(`{"uuid":"call-9","status":"started"}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125551234",
		To:     "+15550001111",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})
	s.Require().NoError(err)

	s.Equal("/v1/calls", s.seen.path)
	s.NotEmpty(s.seen.bearer, "the voice api is not authenticated by the key and secret")
	s.Equal("15550001111", s.seen.body["to"].([]any)[0].(map[string]any)["number"],
		"the voice api quotes numbers without a plus")

	connect := s.seen.body["ncco"].([]any)[0].(map[string]any)
	s.Equal("connect", connect["action"])
	s.Equal("sip:trunk-7@sip.stream-io-api.com",
		connect["endpoint"].([]any)[0].(map[string]any)["uri"])

	s.Equal("call-9", placed.VendorCallID)
	s.Equal("started", placed.Status)
}

func (s *VonageSuite) TestTheRingTimeoutIsCarriedOnlyWhenItWasAskedFor() {
	s.answer(`{"uuid":"call-9","status":"started"}`)
	call := phone.Outbound{
		From:   "+15125551234",
		To:     "+15550001111",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	}

	_, err := s.provider.Dial(s.ctx, call)
	s.Require().NoError(err)
	s.NotContains(s.seen.body, "ringing_timer", "zero would give up before it rang")

	call.RingTimeout = 25 * time.Second
	_, err = s.provider.Dial(s.ctx, call)
	s.Require().NoError(err)
	s.Equal(float64(25), s.seen.body["ringing_timer"])
}

func (s *VonageSuite) TestRingingOutsideWhatVonageAllowsIsRefusedBeforeItIsAsked() {
	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:        "+15125551234",
		To:          "+15550001111",
		Bridge:      phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
		RingTimeout: 5 * time.Minute,
	})

	s.ErrorContains(err, "1-120s")
	s.Empty(s.seen.path, "a call vonage would reject is not sent")
}

func (s *VonageSuite) TestDiallingWithoutTheVoiceCredentialsSaysWhichAreMissing() {
	// Buying numbers from Vonage should not require holding an application's private key,
	// so the two are separate and the call is what complains.
	numbersOnly, err := New(Options{
		APIKey:       "key",
		APISecret:    "secret",
		BaseURL:      s.server.URL,
		VoiceBaseURL: s.server.URL,
	})
	s.Require().NoError(err)

	_, err = numbersOnly.Dial(s.ctx, phone.Outbound{
		From:   "+15125551234",
		To:     "+15550001111",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})

	s.ErrorContains(err, "VONAGE_APPLICATION_ID")
	s.ErrorContains(err, "VONAGE_PRIVATE_KEY")
}

func (s *VonageSuite) TestAPrivateKeyThatIsNeitherAPemNorAFileIsRejectedOnOpening() {
	_, err := New(Options{APIKey: "key", APISecret: "secret", PrivateKey: "/nope/key.pem"})

	s.ErrorContains(err, "neither a pem nor a readable file")
}

func (s *VonageSuite) TestPlaceSearchesAreNotClaimed() {
	s.False(s.provider.Supports(phone.FilterAdministrativeArea))
	s.False(s.provider.Supports(phone.FilterLocality))
	s.True(s.provider.Supports(phone.FilterPrefix))
}

func (s *VonageSuite) TestAFailureFromVonageSaysWhatVonageSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error_code_label":"country not supported"}`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "ZZ"})

	s.ErrorContains(err, "400")
	s.ErrorContains(err, "country not supported")
}

func (s *VonageSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}
