// Package plivo buys and releases phone numbers at Plivo.
//
// Plivo's v1 API is JSON with basic auth on the account's auth id and token, and the account
// id is part of every path. Numbers are quoted without a leading plus, so they are rendered
// into E.164 on the way out and stripped on the way in.
//
// Plivo will not take a call plan on the request that places a call: answer_url is mandatory
// and there is no inline alternative. Dialling therefore needs this service to be publicly
// reachable, and the plan it serves is Answer here.
//
// Pointing a number at a trunk for inbound calls is still not wrapped: that is a Plivo
// application rather than a property of the number.
package plivo

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
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

const (
	authIDEnvVar    = "PLIVO_AUTH_ID"
	authTokenEnvVar = "PLIVO_AUTH_TOKEN"
)

const (
	defaultBaseURL = "https://api.plivo.com"
	defaultTimeout = 30 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// microsPerDollar converts Plivo's dollar strings into the micros used everywhere else.
const microsPerDollar = 1_000_000

// Options configures a Provider. The credentials fall back to the environment.
type Options struct {
	// AuthID defaults to PLIVO_AUTH_ID.
	AuthID string
	// AuthToken defaults to PLIVO_AUTH_TOKEN.
	AuthToken string
	// BaseURL defaults to Plivo's API host.
	BaseURL string
	// Timeout bounds one call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is Plivo. It satisfies phone.Provider.
type Provider struct {
	authID    string
	authToken string
	baseURL   string
	client    *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.AuthID == "" {
		options.AuthID = os.Getenv(authIDEnvVar)
	}
	if options.AuthToken == "" {
		options.AuthToken = os.Getenv(authTokenEnvVar)
	}
	if options.AuthID == "" || options.AuthToken == "" {
		return nil, errors.New("plivo: " + authIDEnvVar + " and " + authTokenEnvVar + " are required")
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
		authID:    options.AuthID,
		authToken: options.AuthToken,
		baseURL:   strings.TrimSuffix(options.BaseURL, "/"),
		client:    options.HTTPClient,
	}, nil
}

// SearchNumbers returns numbers Plivo is offering in a country.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, errors.New("plivo: a country is required to search for numbers")
	}

	query := url.Values{"country_iso": {strings.ToUpper(search.Country)}}
	// Plivo's pattern is anchored after the country dial code, so an area code and a
	// prefix are the same request to it.
	if pattern := firstOf(search.Prefix, search.AreaCode); pattern != "" {
		query.Set("pattern", pattern)
	}
	if search.Locality != "" {
		query.Set("city", search.Locality)
	}
	if search.Type != "" {
		kind, ok := kinds[search.Type]
		if !ok {
			return nil, fmt.Errorf("plivo: does not sell %s numbers", search.Type)
		}
		query.Set("type", kind)
	}
	if services := services(search.Capabilities); services != "" {
		query.Set("services", services)
	}
	if search.Limit > 0 {
		query.Set("limit", strconv.Itoa(search.Limit))
	}

	var response listed
	if err := p.do(ctx, http.MethodGet, p.path("PhoneNumber", ""), query, nil, &response); err != nil {
		return nil, err
	}

	offered := make([]phone.Available, 0, len(response.Objects))
	for _, number := range response.Objects {
		offered = append(offered, phone.Available{
			E164:              e164(number.Number),
			Vendor:            p.Vendor(),
			Country:           strings.ToUpper(number.CountryISO),
			Region:            number.Region,
			Locality:          number.City,
			Type:              numberType(number.Type),
			Capabilities:      number.capabilities(),
			MonthlyCostMicros: dollarsToMicros(number.MonthlyRentalRate),
		})
	}
	return offered, nil
}

// BuyNumber rents a number. Plivo rents by number, so the order's country is not needed.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, errors.New("plivo: a number is required")
	}

	var response rented
	path := p.path("PhoneNumber", digits(order.E164))
	if err := p.do(ctx, http.MethodPost, path, nil, struct{}{}, &response); err != nil {
		return phone.Number{}, err
	}

	// Plivo answers a rental with a status rather than the number's details, so what is
	// known about it is what was asked for.
	return phone.Number{
		E164:     order.E164,
		Vendor:   p.Vendor(),
		Country:  strings.ToUpper(order.Country),
		VendorID: digits(order.E164),
	}, nil
}

// ReleaseNumber gives a number back, which is what stops the charge. Plivo calls a rented
// number a Number rather than a PhoneNumber, which is a different path from buying it.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	if e164 == "" {
		return errors.New("plivo: a number is required")
	}
	return p.do(ctx, http.MethodDelete, p.path("Number", digits(e164)), nil, nil, nil)
}

// ConfigureInbound is not wrapped for Plivo.
//
// Plivo routes an incoming call by handing it to an application that answers with XML, so
// pointing a number at a Stream trunk means hosting that XML. Buying the number does not, so
// this says what is missing rather than half-doing it.
func (p *Provider) ConfigureInbound(context.Context, phone.Inbound) error {
	return fmt.Errorf("%w: plivo numbers are bought here but bridged elsewhere", phone.ErrNotImplemented)
}

// Dial calls a person and has Plivo fetch, on answer, the XML that bridges them to the trunk.
//
// Plivo will not take a call plan on the request: answer_url is mandatory and there is no
// inline alternative, so the plan is served from this service instead and the url naming it
// is minted per call. Plivo can present the trunk's digest credentials, which is why its
// trunk needs no address allowlist.
func (p *Provider) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if err := outbound.Validate(); err != nil {
		return phone.Dialed{}, fmt.Errorf("plivo: %w", err)
	}
	if outbound.AnswerURL == "" {
		return phone.Dialed{}, errors.New(
			"plivo: fetches its call plan when the person answers, so it needs somewhere to fetch it from")
	}

	request := callRequest{
		From: digits(outbound.From),
		To:   digits(outbound.To),
		// Plivo defaults to POST, and the plan is the same either way, so GET keeps the
		// fetch to something that can be tried by hand.
		AnswerURL:    outbound.AnswerURL,
		AnswerMethod: http.MethodGet,
		RingTimeout:  int(outbound.RingTimeout.Seconds()),
	}

	var placed dialedCall
	if err := p.do(ctx, http.MethodPost, p.path("Call", ""), nil, request, &placed); err != nil {
		return phone.Dialed{}, err
	}
	// Plivo answers with the id of the request rather than of the call: the call does not
	// exist yet, because Plivo has not dialled it. It is what its API takes to look the
	// call up later, so it is what is reported.
	return phone.Dialed{VendorCallID: placed.RequestUUID, Status: "queued"}, nil
}

// Answer renders Plivo's XML for pressing nothing and bridging to the trunk.
//
// The XML runs on the leg to the person, and by the time it runs they have answered, so
// there is no verb here that presses keys at them. Plivo takes the trunk's credentials on
// the User element, which is what lets its trunk keep a password instead of an allowlist.
func (p *Provider) Answer(bridge phone.Bridge, _ string) (phone.Plan, error) {
	if err := bridge.Validate(); err != nil {
		return phone.Plan{}, err
	}

	rendered, err := xml.Marshal(response{
		Dial: dial{User: user{
			URI:             bridge.URI,
			SIPAuthUsername: bridge.Username,
			SIPAuthPassword: bridge.Password,
		}},
	})
	if err != nil {
		return phone.Plan{}, fmt.Errorf("plivo: render answer: %w", err)
	}
	return phone.Plan{
		ContentType: "application/xml",
		Body:        append([]byte(xml.Header), rendered...),
	}, nil
}

// SendDigits is not wrapped for Plivo, since nothing here places a Plivo call to press on.
func (p *Provider) SendDigits(context.Context, string, string) error {
	return fmt.Errorf("%w: plivo", phone.ErrNotImplemented)
}

// Supports is a country, an anchored prefix, a city and a number type.
//
// Plivo's pattern matches from the front, so it cannot answer a search for digits anywhere
// in the number. Its region filter takes a state's full name rather than its code, so a
// search for "CO" would come back empty rather than wrong, which is worse than not asking.
func (p *Provider) Supports(filter phone.Filter) bool {
	switch filter {
	case phone.FilterCountry, phone.FilterAreaCode, phone.FilterPrefix,
		phone.FilterLocality, phone.FilterNumberType:
		return true
	default:
		return false
	}
}

// Dials a ring timeout only.
//
// Plivo takes ring_timeout on the call itself. The rest would have to be expressed in the
// XML it fetches when the person answers, and by then they have picked up: there is no verb
// that presses keys at them, and nowhere to put a header on a leg already up.
func (p *Provider) Dials(feature phone.CallFeature) bool {
	return feature == phone.FeatureRingTimeout
}

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "plivo" }

// Client exposes the HTTP client, so a caller can reach parts of Plivo's API this does not
// wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// path builds an account-scoped path. Plivo's resources are capitalised and every one of
// them ends in a slash, which it is strict about.
func (p *Provider) path(resource, id string) string {
	if id == "" {
		return "/v1/Account/" + p.authID + "/" + resource + "/"
	}
	return "/v1/Account/" + p.authID + "/" + resource + "/" + url.PathEscape(id) + "/"
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
			return fmt.Errorf("plivo: encode %s: %w", path, err)
		}
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, payload)
	if err != nil {
		return fmt.Errorf("plivo: %s: %w", path, err)
	}
	request.SetBasicAuth(p.authID, p.authToken)
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("plivo: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("plivo: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("plivo: decode %s: %w", path, err)
	}
	return nil
}

// kinds are Plivo's names for the number types this service knows.
var kinds = map[phone.NumberType]string{
	phone.Local:    "local",
	phone.TollFree: "tollfree",
	phone.Mobile:   "mobile",
}

// numberType maps Plivo's types back, leaving the national and fixed ones it also sells
// empty rather than calling them something they are not.
func numberType(kind string) phone.NumberType {
	for named, plivo := range kinds {
		if plivo == kind {
			return named
		}
	}
	return ""
}

// services renders the capabilities Plivo can filter on. It only knows voice and SMS, so
// asking for anything else narrows nothing here and is checked on the results instead.
func services(capabilities []phone.Capability) string {
	var wanted []string
	for _, capability := range capabilities {
		switch capability {
		case phone.Voice:
			wanted = append(wanted, "voice")
		case phone.SMS:
			wanted = append(wanted, "sms")
		}
	}
	return strings.Join(wanted, ",")
}

// e164 renders a Plivo number, which is quoted without a plus.
func e164(number string) string {
	if number == "" || strings.HasPrefix(number, "+") {
		return number
	}
	return "+" + number
}

// digits strips the plus, which is how Plivo names a number in a path.
func digits(number string) string { return strings.TrimPrefix(number, "+") }

func firstOf(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
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

type listed struct {
	Objects []availableNumber `json:"objects"`
}

type availableNumber struct {
	Number     string `json:"number"`
	CountryISO string `json:"country_iso"`
	Region     string `json:"region"`
	City       string `json:"city"`
	Type       string `json:"type"`
	// MonthlyRentalRate is a dollar string, as everything Plivo prices is.
	MonthlyRentalRate string `json:"monthly_rental_rate"`
	VoiceEnabled      bool   `json:"voice_enabled"`
	SMSEnabled        bool   `json:"sms_enabled"`
	MMSEnabled        bool   `json:"mms_enabled"`
}

func (a availableNumber) capabilities() []phone.Capability {
	var has []phone.Capability
	if a.VoiceEnabled {
		has = append(has, phone.Voice)
	}
	if a.SMSEnabled {
		has = append(has, phone.SMS)
	}
	if a.MMSEnabled {
		has = append(has, phone.MMS)
	}
	return has
}

type rented struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

// callRequest places a call whose plan Plivo fetches when the person answers.
type callRequest struct {
	From string `json:"from"`
	To   string `json:"to"`
	// AnswerURL is where Plivo fetches the plan from. Plivo requires it.
	AnswerURL    string `json:"answer_url"`
	AnswerMethod string `json:"answer_method,omitempty"`
	// RingTimeout is how long to ring before giving up, in seconds.
	RingTimeout int `json:"ring_timeout,omitempty"`
}

// dialedCall is what Plivo answers a placed call with. The uuid names the request rather
// than the call, since Plivo has not dialled anything yet when it replies.
type dialedCall struct {
	RequestUUID string `json:"request_uuid"`
	Message     string `json:"message"`
}

// response is the root of Plivo's answer XML.
type response struct {
	XMLName xml.Name `xml:"Response"`
	Dial    dial     `xml:"Dial"`
}

type dial struct {
	User user `xml:"User"`
}

// user is a SIP destination, which carries the credentials to reach it.
type user struct {
	URI             string `xml:",chardata"`
	SIPAuthUsername string `xml:"sipAuthUsername,attr,omitempty"`
	SIPAuthPassword string `xml:"sipAuthPassword,attr,omitempty"`
}
