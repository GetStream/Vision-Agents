// Package didww buys and releases phone numbers at DIDWW.
//
// DIDWW is JSON:API: everything is a typed resource with relationships, an Api-Key header
// authenticates, and the content type is application/vnd.api+json rather than plain JSON.
//
// Nothing here is filtered by name. A country, a city or a number type is named by DIDWW's own
// identifier for it, so a search resolves the country to an id first. Buying is not a request
// against a number either: it is an order for a stock-keeping unit, so buying a number found
// earlier means looking up which SKU sells it.
//
// Note that /v3/available_dids is disabled on a new account until DIDWW enables it, which is
// what searching for a specific number needs.
//
// Only search, buy and release are wrapped. Bridging a DIDWW number to a Stream trunk means a
// voice trunk configured at DIDWW, which is separate work from buying the number.
package didww

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

const apiKeyEnvVar = "DIDWW_API_KEY"

const (
	defaultBaseURL = "https://api.didww.com"
	defaultTimeout = 30 * time.Second
)

// contentType is JSON:API's own, which DIDWW rejects requests without.
const contentType = "application/vnd.api+json"

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// microsPerDollar converts DIDWW's decimal strings into the micros used everywhere else.
const microsPerDollar = 1_000_000

// Options configures a Provider. The key falls back to the environment.
type Options struct {
	// APIKey defaults to DIDWW_API_KEY.
	APIKey string
	// BaseURL defaults to DIDWW's API host.
	BaseURL string
	// Timeout bounds one call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is DIDWW. It satisfies phone.Provider.
type Provider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("didww: " + apiKeyEnvVar + " is required")
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: options.Timeout}
	}

	return &Provider{
		apiKey:  options.APIKey,
		baseURL: strings.TrimSuffix(options.BaseURL, "/"),
		client:  options.HTTPClient,
	}, nil
}

// SearchNumbers returns numbers DIDWW is offering in a country.
//
// The country and the number type are resolved to DIDWW's identifiers first, because that is
// the only way it will filter on either.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, errors.New("didww: a country is required to search for numbers")
	}

	countryID, err := p.countryID(ctx, search.Country)
	if err != nil {
		return nil, err
	}

	query := url.Values{
		"filter[country.id]": {countryID},
		// The SKUs are what a number costs and what an order is placed against, so they
		// are fetched alongside rather than per number.
		"include": {"did_group.stock_keeping_units"},
	}
	if search.Contains != "" {
		query.Set("filter[number_contains]", search.Contains)
	}
	if search.Type != "" {
		typeID, err := p.groupTypeID(ctx, search.Type)
		if err != nil {
			return nil, err
		}
		query.Set("filter[did_group_type.id]", typeID)
	}
	if search.Limit > 0 {
		query.Set("page[size]", strconv.Itoa(search.Limit))
	}

	var response document[[]resource]
	if err := p.do(ctx, http.MethodGet, "/v3/available_dids", query, nil, &response); err != nil {
		return nil, err
	}

	// The monthly charge is on the SKUs, which arrive alongside the numbers rather than
	// inside them, so they are read once and looked up per number.
	prices := monthlyBySKU(response.Included)

	offered := make([]phone.Available, 0, len(response.Data))
	for _, number := range response.Data {
		offered = append(offered, phone.Available{
			E164:              e164(number.Attributes.Number),
			Vendor:            p.Vendor(),
			Country:           strings.ToUpper(search.Country),
			Type:              search.Type,
			Capabilities:      capabilities(number.Attributes.Features),
			MonthlyCostMicros: prices[number.groupID()],
		})
	}
	return offered, nil
}

// BuyNumber orders a number.
//
// DIDWW sells a stock-keeping unit rather than a number, so the number is looked up to find
// which SKU offers it and which available DID it is, and the order names both.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, errors.New("didww: a number is required")
	}

	availableID, skuID, err := p.offerFor(ctx, order.E164)
	if err != nil {
		return phone.Number{}, err
	}

	request := document[orderResource]{
		Data: orderResource{
			Type: "orders",
			Attributes: orderAttributes{
				Items: []orderItem{{
					Type: "did_order_items",
					Attributes: orderItemAttributes{
						SKUID:          skuID,
						AvailableDIDID: availableID,
					},
				}},
			},
		},
	}

	var response document[resource]
	if err := p.do(ctx, http.MethodPost, "/v3/orders", nil, request, &response); err != nil {
		return phone.Number{}, err
	}

	return phone.Number{
		E164:     order.E164,
		Vendor:   p.Vendor(),
		Country:  strings.ToUpper(order.Country),
		VendorID: response.Data.ID,
	}, nil
}

// ReleaseNumber cancels a number, which is what stops the charge. DIDWW spells cancelling as
// an update setting the DID terminated rather than as a delete, so a cancelled number keeps
// its history the same way this service's own rows do.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	id, err := p.didID(ctx, e164)
	if err != nil {
		return err
	}

	request := document[didResource]{
		Data: didResource{
			ID:         id,
			Type:       "dids",
			Attributes: didAttributes{Terminated: true},
		},
	}
	return p.do(ctx, http.MethodPatch, "/v3/dids/"+url.PathEscape(id), nil, request, nil)
}

// ConfigureInbound is not wrapped for DIDWW.
func (p *Provider) ConfigureInbound(context.Context, phone.Inbound) error {
	return fmt.Errorf("%w: didww numbers are bought here but bridged elsewhere", phone.ErrNotImplemented)
}

// Dial is not wrapped for DIDWW.
func (p *Provider) Dial(context.Context, phone.Outbound) (phone.Dialed, error) {
	return phone.Dialed{}, fmt.Errorf("%w: didww", phone.ErrNotImplemented)
}

// SendDigits is not wrapped for DIDWW, since nothing here places a DIDWW call to press on.
func (p *Provider) SendDigits(context.Context, string, string) error {
	return fmt.Errorf("%w: didww", phone.ErrNotImplemented)
}

// Supports is a country, a substring and a number type.
//
// DIDWW's city and region filters take its own identifiers rather than names, and resolving a
// city name would be another lookup per search for an inventory whose cities are named its own
// way. An area code is not a filter it has at all: its digit matching is a substring.
func (p *Provider) Supports(filter phone.Filter) bool {
	switch filter {
	case phone.FilterCountry, phone.FilterContains, phone.FilterNumberType:
		return true
	default:
		return false
	}
}

// Dials claims every feature so that a call reaches Dial and is refused there by name.
// DIDWW has no call control API at all, and "didww cannot place a call" is the answer worth
// getting rather than a complaint about one of the call's terms.
func (p *Provider) Dials(phone.CallFeature) bool { return true }

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "didww" }

// Client exposes the HTTP client, so a caller can reach parts of DIDWW's API this does not
// wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// countryID finds DIDWW's identifier for a country, which is what every other filter is
// scoped by.
func (p *Provider) countryID(ctx context.Context, country string) (string, error) {
	query := url.Values{"filter[iso]": {strings.ToUpper(country)}}

	var response document[[]resource]
	if err := p.do(ctx, http.MethodGet, "/v3/countries", query, nil, &response); err != nil {
		return "", err
	}
	if len(response.Data) == 0 {
		return "", fmt.Errorf("didww: does not sell numbers in %s", strings.ToUpper(country))
	}
	return response.Data[0].ID, nil
}

// groupTypeID finds DIDWW's identifier for a kind of number, which it names rather than codes.
func (p *Provider) groupTypeID(ctx context.Context, kind phone.NumberType) (string, error) {
	named, ok := kinds[kind]
	if !ok {
		return "", fmt.Errorf("didww: does not sell %s numbers", kind)
	}
	query := url.Values{"filter[name]": {named}}

	var response document[[]resource]
	if err := p.do(ctx, http.MethodGet, "/v3/did_group_types", query, nil, &response); err != nil {
		return "", err
	}
	for _, group := range response.Data {
		if strings.EqualFold(group.Attributes.Name, named) {
			return group.ID, nil
		}
	}
	return "", fmt.Errorf("didww: does not sell %s numbers", kind)
}

// offerFor finds which available DID a number is and which SKU sells it, which is what an
// order names instead of the number.
func (p *Provider) offerFor(ctx context.Context, e164 string) (string, string, error) {
	query := url.Values{
		"filter[number_contains]": {digits(e164)},
		"include":                 {"did_group.stock_keeping_units"},
	}

	var response document[[]resource]
	if err := p.do(ctx, http.MethodGet, "/v3/available_dids", query, nil, &response); err != nil {
		return "", "", err
	}

	skus := skuBySKUGroup(response.Included)
	for _, number := range response.Data {
		if digits(number.Attributes.Number) != digits(e164) {
			continue
		}
		sku, ok := skus[number.groupID()]
		if !ok {
			return "", "", fmt.Errorf("didww: %s is offered without a price to order it at", e164)
		}
		return number.ID, sku, nil
	}
	return "", "", fmt.Errorf("didww: %s is not one of the numbers on offer", e164)
}

// didID finds DIDWW's identifier for a number this account holds, which is what cancelling
// it needs.
func (p *Provider) didID(ctx context.Context, e164 string) (string, error) {
	if e164 == "" {
		return "", errors.New("didww: a number is required")
	}
	query := url.Values{"filter[number]": {digits(e164)}}

	var response document[[]resource]
	if err := p.do(ctx, http.MethodGet, "/v3/dids", query, nil, &response); err != nil {
		return "", err
	}
	for _, did := range response.Data {
		if digits(did.Attributes.Number) == digits(e164) {
			return did.ID, nil
		}
	}
	return "", fmt.Errorf("didww: %s is not one of this account's numbers", e164)
}

func (p *Provider) do(ctx context.Context, method, path string, query url.Values, body, into any) error {
	endpoint := p.baseURL + path
	if len(query) > 0 {
		endpoint += "?" + query.Encode()
	}

	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("didww: encode %s: %w", path, err)
		}
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, payload)
	if err != nil {
		return fmt.Errorf("didww: %s: %w", path, err)
	}
	request.Header.Set("Api-Key", p.apiKey)
	request.Header.Set("Accept", contentType)
	if body != nil {
		request.Header.Set("Content-Type", contentType)
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("didww: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("didww: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("didww: decode %s: %w", path, err)
	}
	return nil
}

// kinds are DIDWW's names for the number types this service knows, which it spells as words
// rather than codes.
var kinds = map[phone.NumberType]string{
	phone.Local:    "Local",
	phone.TollFree: "Toll-free",
	phone.Mobile:   "Mobile",
}

// monthlyBySKU reads the monthly charge out of the included SKUs, keyed by the DID group they
// belong to, which is how a number finds its own price.
func monthlyBySKU(included []resource) map[string]int64 {
	prices := map[string]int64{}
	for _, each := range included {
		if each.Type != "stock_keeping_units" {
			continue
		}
		group := each.groupID()
		if group == "" {
			continue
		}
		if _, seen := prices[group]; seen {
			continue
		}
		prices[group] = dollarsToMicros(each.Attributes.MonthlyRecurringCharge)
	}
	return prices
}

// skuBySKUGroup is which SKU sells each DID group, which is what an order names.
func skuBySKUGroup(included []resource) map[string]string {
	skus := map[string]string{}
	for _, each := range included {
		if each.Type != "stock_keeping_units" {
			continue
		}
		group := each.groupID()
		if group == "" {
			continue
		}
		if _, seen := skus[group]; !seen {
			skus[group] = each.ID
		}
	}
	return skus
}

// capabilities maps DIDWW's feature names onto the contract's. DIDWW names the direction of
// voice and of messaging separately, and only the inbound ones matter to an agent answering.
func capabilities(features []string) []phone.Capability {
	var has []phone.Capability
	for _, feature := range features {
		switch feature {
		case "voice_in":
			has = append(has, phone.Voice)
		case "sms_in":
			has = append(has, phone.SMS)
		case "t38":
			has = append(has, phone.Fax)
		case "emergency":
			has = append(has, phone.Emergency)
		}
	}
	return has
}

// e164 renders a DIDWW number, which is quoted as digits without a plus.
func e164(number string) string {
	if number == "" || strings.HasPrefix(number, "+") {
		return number
	}
	return "+" + number
}

func digits(number string) string { return strings.TrimPrefix(number, "+") }

func dollarsToMicros(dollars string) int64 {
	if dollars == "" {
		return 0
	}
	amount, err := strconv.ParseFloat(dollars, 64)
	if err != nil {
		return 0
	}
	return int64(amount * microsPerDollar)
}

// document is JSON:API's envelope. Included carries the resources asked for alongside the
// ones requested, which is where the prices arrive.
type document[T any] struct {
	Data     T          `json:"data"`
	Included []resource `json:"included,omitempty"`
}

// resource is one JSON:API resource, read loosely enough to serve for countries, group types,
// available numbers, DIDs and SKUs, which differ only in which attributes are filled in.
type resource struct {
	ID            string     `json:"id"`
	Type          string     `json:"type"`
	Attributes    attributes `json:"attributes"`
	Relationships map[string]struct {
		Data struct {
			ID   string `json:"id"`
			Type string `json:"type"`
		} `json:"data"`
	} `json:"relationships"`
}

// groupID is which DID group a resource belongs to, which is what ties a number to its price.
func (r resource) groupID() string {
	return r.Relationships["did_group"].Data.ID
}

type attributes struct {
	Name                   string   `json:"name"`
	ISO                    string   `json:"iso"`
	Number                 string   `json:"number"`
	Features               []string `json:"features"`
	MonthlyRecurringCharge string   `json:"monthly_recurring_charge"`
}

type didResource struct {
	ID         string        `json:"id"`
	Type       string        `json:"type"`
	Attributes didAttributes `json:"attributes"`
}

type didAttributes struct {
	Terminated bool `json:"terminated"`
}

type orderResource struct {
	Type       string          `json:"type"`
	Attributes orderAttributes `json:"attributes"`
}

type orderAttributes struct {
	Items []orderItem `json:"items"`
}

type orderItem struct {
	Type       string              `json:"type"`
	Attributes orderItemAttributes `json:"attributes"`
}

type orderItemAttributes struct {
	SKUID          string `json:"sku_id"`
	AvailableDIDID string `json:"available_did_id"`
}
