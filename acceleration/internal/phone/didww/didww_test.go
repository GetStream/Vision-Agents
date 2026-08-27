package didww

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

type DIDWWSuite struct {
	suite.Suite
	ctx      context.Context
	server   *httptest.Server
	provider *Provider

	// seen is what the last request asked for.
	seen request
	// answers is what to reply per path, since nothing DIDWW does takes one request.
	answers map[string]string
	// respond replaces the per-path answers when a test needs to fail or count.
	respond func(w http.ResponseWriter, r *http.Request)
}

type request struct {
	method string
	path   string
	query  url.Values
	body   map[string]any
	apiKey string
	accept string
}

func TestDIDWW(t *testing.T) { suite.Run(t, new(DIDWWSuite)) }

func (s *DIDWWSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = request{}
	s.answers = map[string]string{
		"/v3/countries":       `{"data":[{"id":"country-us","type":"countries","attributes":{"iso":"US"}}]}`,
		"/v3/did_group_types": `{"data":[{"id":"type-local","type":"did_group_types","attributes":{"name":"Local"}}]}`,
		"/v3/available_dids":  `{"data":[]}`,
		"/v3/dids":            `{"data":[]}`,
		"/v3/orders":          `{"data":{"id":"order-9","type":"orders"}}`,
	}
	s.respond = nil

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
			apiKey: r.Header.Get("Api-Key"),
			accept: r.Header.Get("Accept"),
		}
		w.Header().Set("Content-Type", contentType)
		if s.respond != nil {
			s.respond(w, r)
			return
		}
		answer, ok := s.answers[r.URL.Path]
		if !ok {
			answer = `{"data":{}}`
		}
		_, _ = w.Write([]byte(answer))
	}))

	provider, err := New(Options{APIKey: "KEY123", BaseURL: s.server.URL})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *DIDWWSuite) TearDownTest() { s.server.Close() }

func (s *DIDWWSuite) TestAKeyIsRequired() {
	s.T().Setenv("DIDWW_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "DIDWW_API_KEY")
}

func (s *DIDWWSuite) TestSearchingResolvesTheCountryToDidwwsOwnIdentifierFirst() {
	// DIDWW filters by its own ids rather than by name, so a country has to be looked up
	// before anything can be searched in it.
	var paths []string
	s.answers["/v3/available_dids"] = `{"data":[{
		"id":"avail-1","type":"available_dids",
		"attributes":{"number":"17195551234","features":["voice_in","sms_in","emergency"]},
		"relationships":{"did_group":{"data":{"id":"group-1","type":"did_groups"}}}}],
		"included":[{
			"id":"sku-1","type":"stock_keeping_units",
			"attributes":{"monthly_recurring_charge":"1.00"},
			"relationships":{"did_group":{"data":{"id":"group-1","type":"did_groups"}}}}]}`
	s.respond = func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		_, _ = w.Write([]byte(s.answers[r.URL.Path]))
	}

	offered, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "us", Limit: 5})
	s.Require().NoError(err)

	s.Equal([]string{"/v3/countries", "/v3/available_dids"}, paths)
	s.Equal("KEY123", s.seen.apiKey)
	s.Equal(contentType, s.seen.accept)
	s.Equal("country-us", s.seen.query.Get("filter[country.id]"))
	s.Equal("did_group.stock_keeping_units", s.seen.query.Get("include"))
	s.Equal("5", s.seen.query.Get("page[size]"))

	s.Require().Len(offered, 1)
	s.Equal("+17195551234", offered[0].E164)
	s.Equal("didww", offered[0].Vendor)
	s.Equal("US", offered[0].Country)
	s.Equal([]phone.Capability{phone.Voice, phone.SMS, phone.Emergency}, offered[0].Capabilities)
	s.Equal(int64(1_000_000), offered[0].MonthlyCostMicros,
		"the price is on the sku alongside the number, not inside it")
}

func (s *DIDWWSuite) TestSearchingATypeResolvesThatToAnIdentifierToo() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US", Type: phone.Local})
	s.Require().NoError(err)

	s.Equal("type-local", s.seen.query.Get("filter[did_group_type.id]"))
}

func (s *DIDWWSuite) TestACountryDidwwDoesNotSellInSaysSo() {
	s.answers["/v3/countries"] = `{"data":[]}`

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "ZZ"})

	s.ErrorContains(err, "does not sell numbers in ZZ")
}

func (s *DIDWWSuite) TestSearchingWithoutACountryIsRejectedBeforeAnyCall() {
	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{})

	s.ErrorContains(err, "country is required")
	s.Empty(s.seen.path)
}

func (s *DIDWWSuite) TestBuyingOrdersTheSkuThatSellsTheNumber() {
	// DIDWW sells a stock-keeping unit rather than a number, so buying one found earlier
	// means finding which sku offers it.
	s.answers["/v3/available_dids"] = `{"data":[{
		"id":"avail-1","type":"available_dids",
		"attributes":{"number":"17195551234"},
		"relationships":{"did_group":{"data":{"id":"group-1","type":"did_groups"}}}}],
		"included":[{
			"id":"sku-1","type":"stock_keeping_units",
			"attributes":{"monthly_recurring_charge":"1.00"},
			"relationships":{"did_group":{"data":{"id":"group-1","type":"did_groups"}}}}]}`

	bought, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234", Country: "US"})
	s.Require().NoError(err)

	s.Equal(http.MethodPost, s.seen.method)
	s.Equal("/v3/orders", s.seen.path)

	data, _ := s.seen.body["data"].(map[string]any)
	attributes, _ := data["attributes"].(map[string]any)
	items, _ := attributes["items"].([]any)
	s.Require().Len(items, 1)
	item, _ := items[0].(map[string]any)
	itemAttributes, _ := item["attributes"].(map[string]any)
	s.Equal("sku-1", itemAttributes["sku_id"])
	s.Equal("avail-1", itemAttributes["available_did_id"])

	s.Equal("+17195551234", bought.E164)
	s.Equal("order-9", bought.VendorID)
}

func (s *DIDWWSuite) TestANumberNotOnOfferCannotBeBought() {
	_, err := s.provider.BuyNumber(s.ctx, phone.Order{E164: "+17195551234"})

	s.ErrorContains(err, "not one of the numbers on offer")
}

func (s *DIDWWSuite) TestReleasingTerminatesTheDidRatherThanDeletingIt() {
	// DIDWW keeps a cancelled number's history, the same way this service's own rows do.
	s.answers["/v3/dids"] = `{"data":[{"id":"did-3","type":"dids","attributes":{"number":"17195551234"}}]}`

	s.Require().NoError(s.provider.ReleaseNumber(s.ctx, "+17195551234"))

	s.Equal(http.MethodPatch, s.seen.method)
	s.Equal("/v3/dids/did-3", s.seen.path)

	data, _ := s.seen.body["data"].(map[string]any)
	attributes, _ := data["attributes"].(map[string]any)
	s.Equal(true, attributes["terminated"])
}

func (s *DIDWWSuite) TestANumberThisAccountDoesNotHoldCannotBeReleased() {
	err := s.provider.ReleaseNumber(s.ctx, "+17195551234")

	s.ErrorContains(err, "not one of this account's numbers")
}

func (s *DIDWWSuite) TestBridgingACallSaysItIsNotHere() {
	s.ErrorIs(s.provider.ConfigureInbound(s.ctx, phone.Inbound{}), phone.ErrNotImplemented)

	_, err := s.provider.Dial(s.ctx, phone.Outbound{})
	s.ErrorIs(err, phone.ErrNotImplemented)
	s.ErrorIs(s.provider.SendDigits(s.ctx, "call-1", "1"), phone.ErrNotImplemented)
}

func (s *DIDWWSuite) TestPlaceAndAreaCodeSearchesAreNotClaimed() {
	s.False(s.provider.Supports(phone.FilterAdministrativeArea))
	s.False(s.provider.Supports(phone.FilterLocality))
	s.False(s.provider.Supports(phone.FilterAreaCode))
	s.True(s.provider.Supports(phone.FilterContains))
}

func (s *DIDWWSuite) TestAFailureFromDidwwSaysWhatDidwwSaid() {
	s.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"errors":[{"detail":"available_dids is not enabled"}]}`))
	}

	_, err := s.provider.SearchNumbers(s.ctx, phone.Search{Country: "US"})

	s.ErrorContains(err, "403")
	s.ErrorContains(err, "available_dids is not enabled")
}
