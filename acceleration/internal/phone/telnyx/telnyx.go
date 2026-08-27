// Package telnyx buys and operates phone numbers at Telnyx.
//
// Telnyx's v2 API is JSON with a bearer key, and everything is wrapped in a "data"
// envelope. A number reaches an agent by being assigned to a SIP connection that points at
// the Stream trunk, and an outbound call is created against that same connection, because
// Stream cannot dial out itself.
package telnyx

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
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
	// ConnectionID is the SIP connection numbers are assigned to and calls are placed on.
	// It is what carries the bridge configuration on Telnyx's side, so inbound and
	// outbound both need it.
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
		return nil, errors.New("telnyx: " + apiKeyEnvVar + " is required")
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
		return nil, errors.New("telnyx: a country is required to search for numbers")
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

// Dials every feature: Telnyx's call takes a ring timeout, digits to press on answer and
// custom SIP headers, all as parameters on the one request.
func (p *Provider) Dials(phone.CallFeature) bool { return true }

// BuyNumber orders a number. Telnyx fulfils orders asynchronously, so this returns as soon
// as the order is accepted rather than when the number is usable. Telnyx orders by number,
// so the order's country is not needed.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, errors.New("telnyx: a number is required")
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
		E164:     order.E164,
		Vendor:   p.Vendor(),
		VendorID: response.Data.ID,
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
		return errors.New("telnyx: a connection id is required to route calls to the bridge")
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
		return phone.Dialed{}, fmt.Errorf("telnyx: %w", err)
	}
	if p.connectionID == "" {
		return phone.Dialed{}, errors.New("telnyx: a connection id is required to place a call")
	}

	// The person is called from one of this service's numbers and the answered leg is
	// joined to the trunk, so the agent and the person are on the same call.
	request := dialRequest{
		ConnectionID:       p.connectionID,
		From:               outbound.From,
		To:                 outbound.To,
		SIPAuthUsername:    outbound.Bridge.Username,
		SIPAuthPassword:    outbound.Bridge.Password,
		LinkTo:             outbound.Bridge.URI,
		TimeoutSecs:        int(outbound.RingTimeout.Seconds()),
		SendDigitsOnAnswer: outbound.InitialDigits,
		CustomHeaders:      headers(outbound.Headers),
	}

	var response envelope[dialedCall]
	if err := p.do(ctx, http.MethodPost, "/v2/calls", nil, request, &response); err != nil {
		return phone.Dialed{}, err
	}
	// Telnyx reports whether the call is up rather than naming a state, so this says what
	// that means in the same words the other vendors use.
	status := "ended"
	if response.Data.IsAlive {
		status = "dialing"
	}
	return phone.Dialed{VendorCallID: response.Data.CallControlID, Status: status}, nil
}

// SendDigits presses digits on a leg Telnyx is holding, which is how an agent gets past a
// menu on a call it placed.
//
// Telnyx's call control sends the tones without disturbing what the leg is already doing,
// so the agent stays bridged to the call while the menu is answered.
func (p *Provider) SendDigits(ctx context.Context, vendorCallID, digits string) error {
	if vendorCallID == "" {
		return errors.New("telnyx: pressing digits needs the call to press them on")
	}
	if digits == "" {
		return errors.New("telnyx: pressing needs digits to press")
	}

	path := "/v2/calls/" + url.PathEscape(vendorCallID) + "/actions/send_dtmf"
	return p.do(ctx, http.MethodPost, path, nil, sendDigits{Digits: digits}, nil)
}

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "telnyx" }

// Client exposes the HTTP client, so a caller can reach parts of Telnyx's API this does
// not wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// idFor finds Telnyx's own identifier for a number this account owns, which is what
// changing or releasing it needs.
func (p *Provider) idFor(ctx context.Context, e164 string) (string, error) {
	if e164 == "" {
		return "", errors.New("telnyx: a number is required")
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
	return "", fmt.Errorf("telnyx: %s is not one of this account's numbers", e164)
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
			return fmt.Errorf("telnyx: encode %s: %w", path, err)
		}
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, payload)
	if err != nil {
		return fmt.Errorf("telnyx: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+p.apiKey)
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("telnyx: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("telnyx: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("telnyx: decode %s: %w", path, err)
	}
	return nil
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

type dialRequest struct {
	ConnectionID    string `json:"connection_id"`
	From            string `json:"from"`
	To              string `json:"to"`
	SIPAuthUsername string `json:"sip_auth_username,omitempty"`
	SIPAuthPassword string `json:"sip_auth_password,omitempty"`
	// LinkTo is the SIP address the answered call is joined to, which is the trunk the
	// agent is on.
	LinkTo string `json:"link_to,omitempty"`
	// TimeoutSecs is how long to ring before giving up.
	TimeoutSecs int `json:"timeout_secs,omitempty"`
	// SendDigitsOnAnswer are pressed once the person picks up.
	SendDigitsOnAnswer string `json:"send_digits_on_answer,omitempty"`
	// CustomHeaders travel on the INVITE to the person.
	CustomHeaders []customHeader `json:"custom_headers,omitempty"`
}

// customHeader is one SIP header carried to the person's leg.
type customHeader struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// headers renders custom headers in the shape Telnyx takes them, sorted so that the same
// call twice sends the same request.
func headers(wanted map[string]string) []customHeader {
	if len(wanted) == 0 {
		return nil
	}
	rendered := make([]customHeader, 0, len(wanted))
	for name, value := range wanted {
		rendered = append(rendered, customHeader{Name: name, Value: value})
	}
	slices.SortFunc(rendered, func(a, b customHeader) int {
		return cmp.Compare(a.Name, b.Name)
	})
	return rendered
}

type dialedCall struct {
	CallControlID string `json:"call_control_id"`
	IsAlive       bool   `json:"is_alive"`
}

type sendDigits struct {
	Digits string `json:"digits"`
}
