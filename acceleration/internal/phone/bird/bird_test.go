package bird

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

type BirdSuite struct {
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
	// idempotency is the header that stops a retried order buying twice.
	idempotency string
}

func TestBird(t *testing.T) { suite.Run(t, new(BirdSuite)) }

func (s *BirdSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.answer(`{"data":[]}`)

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &body)
		}
		s.seen = request{
			method:      r.Method,
			path:        r.URL.Path,
			query:       r.URL.Query(),
			body:        body,
			auth:        r.Header.Get("Authorization"),
			idempotency: r.Header.Get("Idempotency-Key"),
		}
		w.Header().Set("Content-Type", "application/json")
		s.respond(w, r)
	}))

	provider, err := New(Options{
		AccessKey:    "bk_us1_abc",
		BaseURL:      s.server.URL,
		VoiceBaseURL: s.server.URL,
	})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *BirdSuite) TearDownTest() { s.server.Close() }

func (s *BirdSuite) TestAKeyIsRequired() {
	s.T().Setenv("BIRD_ACCESS_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "BIRD_ACCESS_KEY")
}

func (s *BirdSuite) TestTheHostComesFromTheKeysOwnRegion() {
	// A key is only valid against its region's host, and using the wrong one fails as an
	// authentication error rather than saying what is actually wrong.
	provider, err := New(Options{AccessKey: "bk_eu1_abc"})
	s.Require().NoError(err)

	s.Equal("https://eu1.platform.bird.com", provider.baseURL)
}

func (s *BirdSuite) TestAKeyWithoutARegionSaysThereIsNoHostToReach() {
	_, err := New(Options{AccessKey: "not-a-bird-key"})

	s.ErrorContains(err, "does not name a region")
}

func (s *BirdSuite) TestSearchingScopesToACountryAndReturnsWhatIsOffered() {
	s.answer(`{"data":[{
		"number":"+17195551234","country_code":"US","number_type":"local",
		"capabilities":["voice","sms"],"monthly_price":{"amount":"1.15","currency":"USD"}}]}`)

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{
		Country:      "us",
		Prefix:       "719",
		Type:         phone.Local,
		Capabilities: []phone.Capability{phone.Voice, phone.SMS},
		Limit:        7,
	})
	s.Require().NoError(err)

	s.Equal("/v1/numbers/available", s.seen.path)
	s.Equal("Bearer bk_us1_abc", s.seen.auth)
	s.Equal("US", s.seen.query.Get("country_code"))
	s.Equal("719", s.seen.query.Get("prefix"))
	s.Equal("local", s.seen.query.Get("number_type"))
	s.Equal([]string{"voice", "sms"}, s.seen.query["capabilities"],
		"bird repeats the parameter to require several at once")
	s.Equal("7", s.seen.query.Get("limit"))

	s.Require().Len(offered, 1)
	s.Equal("+17195551234", offered[0].E164)
	s.Equal("bird", offered[0].Vendor)
	s.Equal(phone.Local, offered[0].Type)
	s.Equal([]phone.Capability{phone.Voice, phone.SMS}, offered[0].Capabilities)
	s.Equal(int64(1_150_000), offered[0].MonthlyCostMicros)
}

func (s *BirdSuite) TestSearchingWithoutACountryIsRejectedBeforeAnyCall() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{})

	s.ErrorContains(err, "country is required")
	s.Empty(s.seen.path)
}

func (s *BirdSuite) TestOrderingCarriesAnIdempotencyKeySoARetryCannotBuyTwice() {
	s.answer(`{"id":"order-9","status":"completed","number":"+17195551234"}`)

	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234", Country: "US"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v1/numbers/orders", s.seen.path)
	s.Equal("+17195551234", s.seen.body["number"])
	s.NotEmpty(s.seen.idempotency)

	s.Equal("+17195551234", bought.E164)
	s.Equal("order-9", bought.VendorID)
	s.Equal("bird", bought.Vendor)
}

func (s *BirdSuite) TestTwoOrdersUseDifferentIdempotencyKeys() {
	// The key protects one order from being retried, not two orders from both happening.
	_, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234"})
	s.Require().NoError(err)
	first := s.seen.idempotency

	_, err = s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195555678"})
	s.Require().NoError(err)

	s.NotEqual(first, s.seen.idempotency)
}

func (s *BirdSuite) TestReleasingLooksUpBirdsOwnIdentifierFirst() {
	var paths []string
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		_, _ = w.Write([]byte(`{"data":[{"id":"num-3","number":"+17195551234"}]}`))
	}

	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+17195551234"))

	s.Equal([]string{"GET /v1/numbers", "DELETE /v1/numbers/num-3"}, paths)
}

func (s *BirdSuite) TestANumberThisWorkspaceDoesNotHoldCannotBeReleased() {
	s.answer(`{"data":[]}`)

	err := s.provider.ReleaseNumber(s.ctx, "+17195551234")

	s.ErrorContains(err, "not one of this workspace's numbers")
}

func (s *BirdSuite) TestPointingANumberAtATrunkSaysItIsNotHere() {
	s.ErrorIs(s.provider.ConfigureInbound(s.ctx, phone.Inbound{}), phone.ErrNotImplemented)
	s.ErrorIs(s.provider.SendDigits(s.ctx, "call-1", "1"), phone.ErrNotImplemented)
}

func (s *BirdSuite) TestDiallingOutTransfersTheAnsweredCallToTheTrunk() {
	// The steps travel with the call, so nothing is hosted to answer it, and a transfer
	// takes a SIP uri as readily as it takes a number.
	s.answer(`{"data":[{"id":"call-9","status":"queued"}]}`)

	placed, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125551234",
		To:     "+15550001111",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})
	s.Require().NoError(err)

	s.Equal("/calls", s.seen.path)
	s.Equal("AccessKey bk_us1_abc", s.seen.auth,
		"the voice api names its scheme differently from the platform api")
	s.Equal("15125551234", s.seen.body["source"], "the voice api quotes numbers without a plus")
	s.Equal("15550001111", s.seen.body["destination"])

	steps := s.seen.body["callFlow"].(map[string]any)["steps"].([]any)
	transfer := steps[0].(map[string]any)
	s.Equal("transfer", transfer["action"])
	s.Equal("sip:trunk-7@sip.stream-io-api.com",
		transfer["options"].(map[string]any)["destination"])

	s.Equal("call-9", placed.VendorCallID)
	s.Equal("queued", placed.Status)
}

func (s *BirdSuite) TestACallBirdAcceptsButPlacesNothingForIsAnError() {
	s.answer(`{"data":[]}`)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{
		From:   "+15125551234",
		To:     "+15550001111",
		Bridge: phone.Bridge{URI: "sip:trunk-7@sip.stream-io-api.com"},
	})

	s.ErrorContains(err, "placed no call")
}

func (s *BirdSuite) TestTermsBirdCannotExpressAreNotClaimed() {
	// A Bird call is a source, a destination and what to do on answer. There is no ring
	// timeout on it and no header anywhere, so the service refuses those before dialling
	// rather than this dropping them.
	s.False(s.provider.Dials(phone.FeatureRingTimeout))
	s.False(s.provider.Dials(phone.FeatureInitialDigits))
	s.False(s.provider.Dials(phone.FeatureCustomHeaders))
}

func (s *BirdSuite) TestPlaceSearchesAreNotClaimed() {
	s.False(s.provider.Supports(phone.FilterAdministrativeArea))
	s.False(s.provider.Supports(phone.FilterLocality))
	s.False(s.provider.Supports(phone.FilterContains))
	s.True(s.provider.Supports(phone.FilterPrefix))
}

func (s *BirdSuite) TestANumberTakenWhileItWasBeingChosenSaysWhatBirdSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte(`{"code":"E14000","message":"number no longer available"}`))
	}

	_, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234"})

	s.ErrorContains(err, "409")
	s.ErrorContains(err, "no longer available")
}

func (s *BirdSuite) answer(body string) {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}
}
