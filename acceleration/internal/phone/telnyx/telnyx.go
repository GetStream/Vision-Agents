// Package telnyx buys and operates phone numbers at Telnyx.
//
// Telnyx's v2 API is JSON with a bearer key, and everything is wrapped in a "data"
// envelope. Numbers are assigned to a TeXML application, and an outbound call is a TeXML
// call on that application whose TeXML dials the Stream trunk, because Stream cannot dial
// out itself.
package telnyx

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
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const apiKeyEnvVar = "TELNYX_API_KEY"

const (
	defaultBaseURL = "https://api.telnyx.com"
	defaultTimeout = 30 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// microsPerDollar converts Telnyx's dollar strings into the micros used everywhere else.
const microsPerDollar = 1_000_000

// Options configures a Provider. The key falls back to the environment.
type Options struct {
	// APIKey defaults to TELNYX_API_KEY.
	APIKey string
	// ConnectionID is the TeXML application numbers are assigned to and calls are placed
	// on, so inbound and outbound both need it.
	ConnectionID string
	// BaseURL defaults to Telnyx's API host.
	BaseURL string
	// Timeout bounds one call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is Telnyx. It satisfies phone.Provider.
type Provider struct {
	apiKey       string
	connectionID string
	baseURL      string
	client       *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("telnyx: " + apiKeyEnvVar + " is required"))
	}
	if options.ConnectionID == "" {
		options.ConnectionID = os.Getenv("TELNYX_CONNECTION_ID")
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
		apiKey:       options.APIKey,
		connectionID: options.ConnectionID,
		baseURL:      strings.TrimSuffix(options.BaseURL, "/"),
		client:       options.HTTPClient,
	}, nil
}

// SearchNumbers returns numbers Telnyx is offering in a country.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, stack.Wrap(errors.New("telnyx: a country is required to search for numbers"))
	}

	// Telnyx filters with bracketed query parameters rather than a request body.
	query := url.Values{"filter[country_code]": {strings.ToUpper(search.Country)}}
	if search.AreaCode != "" {
		query.Set("filter[national_destination_code]", search.AreaCode)
	}
	if search.Contains != "" {
		query.Set("filter[phone_number][contains]", search.Contains)
	}
	if search.Prefix != "" {
		query.Set("filter[phone_number][starts_with]", search.Prefix)
	}
	if search.Locality != "" {
		query.Set("filter[locality]", search.Locality)
	}
	if search.AdministrativeArea != "" {
		query.Set("filter[administrative_area]", strings.ToUpper(search.AdministrativeArea))
	}
	if search.Type != "" {
		query.Set("filter[phone_number_type]", string(search.Type))
	}
	if search.Limit > 0 {
		query.Set("filter[limit]", strconv.Itoa(search.Limit))
	}
	// Telnyx's feature names are the ones the contract uses, so a capability is its own
	// filter value.
	for _, capability := range search.Capabilities {
		query.Add("filter[features][]", string(capability))
	}

	var response envelope[[]availableNumber]
	if err := p.do(ctx, http.MethodGet, "/v2/available_phone_numbers", query, nil, &response); err != nil {
		return nil, err
	}

	offered := make([]phone.Available, 0, len(response.Data))
	for _, number := range response.Data {
		offered = append(offered, phone.Available{
			E164:              number.PhoneNumber,
			Vendor:            p.Vendor(),
			Country:           number.CountryCode,
			Region:            number.RegionInformation.value("state"),
			Locality:          number.RegionInformation.value("rate_center"),
			Type:              numberType(number.PhoneNumberType),
			Capabilities:      features(number.Features),
			MonthlyCostMicros: dollarsToMicros(number.CostInformation.MonthlyCost),
		})
	}
	return offered, nil
}

// Supports is every filter: Telnyx's search is the widest of these vendors, which is why
// the contract's vocabulary is shaped like it.
func (p *Provider) Supports(phone.Filter) bool { return true }

// Dials digits to press on answer, which TeXML plays before dialling the trunk. Telnyx's
// TeXML call has no field for how long to ring the person or for custom SIP headers, and
// <Dial>'s timeout rings the trunk rather than the person.
func (p *Provider) Dials(feature phone.CallFeature) bool {
	return feature == phone.FeatureInitialDigits
}

// BuyNumber orders a number. Telnyx fulfils orders asynchronously, so this returns as soon
// as the order is accepted rather than when the number is usable. Telnyx orders by number,
// so the order's country is not needed.
//
// Neither the order nor the number Telnyx then holds says what it costs, so the price is
// read off the offer just before ordering it, while the number is still for sale.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, stack.Wrap(errors.New("telnyx: a number is required"))
	}

	cost, err := p.priceOf(ctx, order.E164)
	if err != nil {
		return phone.Number{}, err
	}

	request := numberOrder{PhoneNumbers: []orderedNumber{{PhoneNumber: order.E164}}}
	if p.connectionID != "" {
		request.ConnectionID = p.connectionID
	}

	var response envelope[placedOrder]
	if err := p.do(ctx, http.MethodPost, "/v2/number_orders", nil, request, &response); err != nil {
		return phone.Number{}, err
	}

	bought := phone.Number{
		E164:              order.E164,
		Vendor:            p.Vendor(),
		VendorID:          response.Data.ID,
		MonthlyCostMicros: cost,
	}
	for _, ordered := range response.Data.PhoneNumbers {
		if ordered.PhoneNumber != order.E164 {
			continue
		}
		bought.Country = ordered.CountryCode
		bought.Capabilities = features(ordered.Features)
	}
	return bought, nil
}

// ReleaseNumber gives a number back, which is what stops the charge.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	id, err := p.idFor(ctx, e164)
	if err != nil {
		return err
	}
	return p.do(ctx, http.MethodDelete, "/v2/phone_numbers/"+id, nil, nil, nil)
}

// ConfigureInbound assigns the number to the SIP connection that points at the Stream
// trunk, which is how Telnyx knows where to send a call to it.
func (p *Provider) ConfigureInbound(ctx context.Context, inbound phone.Inbound) error {
	if err := inbound.Bridge.Validate(); err != nil {
		return err
	}
	if p.connectionID == "" {
		return stack.Wrap(errors.New("telnyx: a connection id is required to route calls to the bridge"))
	}
	id, err := p.idFor(ctx, inbound.E164)
	if err != nil {
		return err
	}

	update := numberUpdate{ConnectionID: p.connectionID}
	return p.do(ctx, http.MethodPatch, "/v2/phone_numbers/"+id, nil, update, nil)
}

// Dial places a call to a SIP address, which for an agent is the Stream trunk. Stream's
// SIP is inbound only, so the vendor originates and the agent is already waiting on it.
func (p *Provider) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if err := outbound.Validate(); err != nil {
		return phone.Dialed{}, stack.Wrap(fmt.Errorf("telnyx: %w", err))
	}
	if p.connectionID == "" {
		return phone.Dialed{}, stack.Wrap(errors.New("telnyx: a connection id is required to place a call"))
	}

	// Dials is not consulted on every path here, so a term Telnyx would drop is refused.
	if outbound.RingTimeout > 0 {
		return phone.Dialed{}, stack.Wrap(errors.New("telnyx: a TeXML call cannot say how long to ring the person"))
	}
	if len(outbound.Headers) > 0 {
		return phone.Dialed{}, stack.Wrap(errors.New("telnyx: a TeXML call cannot carry custom SIP headers"))
	}
	if strings.ContainsFunc(outbound.InitialDigits, func(key rune) bool {
		return !strings.ContainsRune(playableDigits, key)
	}) {
		return phone.Dialed{}, stack.Wrap(fmt.Errorf("telnyx: TeXML plays only %q, not %q",
			playableDigits, outbound.InitialDigits))
	}

	// The person is called from one of this service's numbers, and once they answer the
	// TeXML dials the trunk, so the agent and the person are on the same call.
	texml, err := renderTexml(outbound.Bridge, 0, outbound.InitialDigits)
	if err != nil {
		return phone.Dialed{}, err
	}
	request := texmlCall{From: outbound.From, To: outbound.To, Texml: string(texml)}

	var placed texmlCallPlaced
	path := "/v2/texml/calls/" + url.PathEscape(p.connectionID)
	if err := p.do(ctx, http.MethodPost, path, nil, request, &placed); err != nil {
		return phone.Dialed{}, err
	}
	return phone.Dialed{VendorCallID: placed.CallSID, Status: placed.Status}, nil
}

// SendDigits presses digits on a leg Telnyx is holding, which is how an agent gets past a
// menu on a call it placed.
//
// Telnyx's call control sends the tones without disturbing what the leg is already doing,
// so the agent stays bridged to the call while the menu is answered.
func (p *Provider) SendDigits(ctx context.Context, vendorCallID, digits string) error {
	if vendorCallID == "" {
		return stack.Wrap(errors.New("telnyx: pressing digits needs the call to press them on"))
	}
	if digits == "" {
		return stack.Wrap(errors.New("telnyx: pressing needs digits to press"))
	}

	path := "/v2/calls/" + url.PathEscape(vendorCallID) + "/actions/send_dtmf"
	return p.do(ctx, http.MethodPost, path, nil, sendDigits{Digits: digits}, nil)
}

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "telnyx" }

// Client exposes the HTTP client, so a caller can reach parts of Telnyx's API this does
// not wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// priceOf is the monthly price Telnyx is offering a number at.
func (p *Provider) priceOf(ctx context.Context, e164 string) (int64, error) {
	query := url.Values{"filter[phone_number][contains]": {strings.TrimPrefix(e164, "+")}}

	var response envelope[[]availableNumber]
	if err := p.do(ctx, http.MethodGet, "/v2/available_phone_numbers", query, nil, &response); err != nil {
		return 0, err
	}
	for _, number := range response.Data {
		if number.PhoneNumber == e164 {
			return dollarsToMicros(number.CostInformation.MonthlyCost), nil
		}
	}
	return 0, stack.Wrap(fmt.Errorf("telnyx: %s is not for sale", e164))
}

// idFor finds Telnyx's own identifier for a number this account owns, which is what
// changing or releasing it needs.
func (p *Provider) idFor(ctx context.Context, e164 string) (string, error) {
	if e164 == "" {
		return "", stack.Wrap(errors.New("telnyx: a number is required"))
	}

	query := url.Values{"filter[phone_number]": {e164}}

	var response envelope[[]ownedNumber]
	if err := p.do(ctx, http.MethodGet, "/v2/phone_numbers", query, nil, &response); err != nil {
		return "", err
	}
	for _, number := range response.Data {
		if number.PhoneNumber == e164 {
			return number.ID, nil
		}
	}
	return "", stack.Wrap(fmt.Errorf("telnyx: %s is not one of this account's numbers", e164))
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
			return stack.Wrap(fmt.Errorf("telnyx: encode %s: %w", path, err))
		}
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, payload)
	if err != nil {
		return stack.Wrap(&phone.VendorError{Vendor: p.Vendor(), Path: path, Message: "could not build the request", Cause: err})
	}
	request.Header.Set("Authorization", "Bearer "+p.apiKey)
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return stack.Wrap(&phone.VendorError{Vendor: p.Vendor(), Path: path, Message: "no answer", Cause: err})
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return stack.Wrap(refusal(p.Vendor(), path, response.StatusCode, detail))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return stack.Wrap(&phone.VendorError{Vendor: p.Vendor(), Path: path, Status: response.StatusCode,
			Message: "could not read the answer", Cause: err})
	}
	return nil
}

// refusal is Telnyx's error answer: the first error's code, and its detail or else its title,
// when the body is Telnyx's JSON, and the body as it came otherwise.
func refusal(vendor, path string, status int, body []byte) *phone.VendorError {
	refused := &phone.VendorError{Vendor: vendor, Path: path, Status: status, Message: strings.TrimSpace(string(body))}
	var answer struct {
		Errors []struct {
			Code   string `json:"code"`
			Title  string `json:"title"`
			Detail string `json:"detail"`
		} `json:"errors"`
	}
	if json.Unmarshal(body, &answer) == nil && len(answer.Errors) > 0 {
		first := answer.Errors[0]
		refused.Code = first.Code
		refused.Message = first.Detail
		if refused.Message == "" {
			refused.Message = first.Title
		}
	}
	return refused
}

// features maps Telnyx's feature names onto capabilities, ignoring the ones that do not
// correspond to anything a caller here asks for.
func features(offered []feature) []phone.Capability {
	var has []phone.Capability
	for _, each := range offered {
		switch phone.Capability(each.Name) {
		case phone.Voice, phone.SMS, phone.MMS, phone.Fax,
			phone.Emergency, phone.HDVoice, phone.InternationalSMS, phone.LocalCalling:
			has = append(has, phone.Capability(each.Name))
		}
	}
	return has
}

// numberType maps Telnyx's number types onto the three this service names, leaving the
// national and shared-cost ones it does not sell empty rather than guessing.
func numberType(kind string) phone.NumberType {
	switch phone.NumberType(kind) {
	case phone.Local, phone.TollFree, phone.Mobile:
		return phone.NumberType(kind)
	default:
		return ""
	}
}

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

// envelope is Telnyx's "data" wrapper, which every response has.
type envelope[T any] struct {
	Data T `json:"data"`
}

type feature struct {
	Name string `json:"name"`
}

// regions is Telnyx's list of region facts, which names what each one is rather than
// giving them fields.
type regions []struct {
	Type  string `json:"region_type"`
	Name  string `json:"region_name"`
	Value string `json:"value"`
}

func (r regions) value(kind string) string {
	for _, region := range r {
		if region.Type != kind {
			continue
		}
		if region.Value != "" {
			return region.Value
		}
		return region.Name
	}
	return ""
}

type availableNumber struct {
	PhoneNumber       string    `json:"phone_number"`
	CountryCode       string    `json:"country_code"`
	PhoneNumberType   string    `json:"phone_number_type"`
	Features          []feature `json:"features"`
	RegionInformation regions   `json:"region_information"`
	CostInformation   struct {
		MonthlyCost string `json:"monthly_cost"`
	} `json:"cost_information"`
}

type ownedNumber struct {
	ID          string `json:"id"`
	PhoneNumber string `json:"phone_number"`
}

type orderedNumber struct {
	PhoneNumber string `json:"phone_number"`
}

type numberOrder struct {
	PhoneNumbers []orderedNumber `json:"phone_numbers"`
	ConnectionID string          `json:"connection_id,omitempty"`
}

type placedOrder struct {
	ID           string `json:"id"`
	PhoneNumbers []struct {
		PhoneNumber string    `json:"phone_number"`
		CountryCode string    `json:"country_code"`
		Features    []feature `json:"features"`
	} `json:"phone_numbers"`
}

type numberUpdate struct {
	ConnectionID string `json:"connection_id"`
}

// playableDigits are the keys TeXML's <Play digits> takes, w being half a second's pause.
const playableDigits = "0123456789*#w"

// texmlCall is a TeXML call on one application. Telnyx's names here are case-sensitive.
type texmlCall struct {
	From  string `json:"From"`
	To    string `json:"To"`
	Texml string `json:"Texml"`
}

// texmlCallPlaced is Telnyx's answer to a TeXML call, which has no "data" envelope.
type texmlCallPlaced struct {
	// CallSID is the call control id, which is what pressing digits acts on.
	CallSID string `json:"call_sid"`
	Status  string `json:"status"`
}

type sendDigits struct {
	Digits string `json:"digits"`
}
