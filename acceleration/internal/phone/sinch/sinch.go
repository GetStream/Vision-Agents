// Package sinch buys and releases phone numbers at Sinch.
//
// Sinch is the only vendor here that does not take its credentials on the request. A key id
// and secret are exchanged for a bearer token at a separate host, and the token is held until
// it is close to expiring, because doing that exchange per call would double every request.
//
// Renting a number and giving it back are actions rather than resources: Sinch spells them as
// a colon and a verb on the end of the path.
//
// Placing a call is a third host and a second pair of credentials: the calling API is
// authenticated by an application key and secret rather than by the project. Its callbacks
// are configured on that application rather than named per call, so a call that needs to say
// something of its own carries the plan inline instead.
//
// Pointing a number at a trunk for inbound calls is still separate work from buying it.
package sinch

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
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

const (
	projectIDEnvVar = "SINCH_PROJECT_ID"
	keyIDEnvVar     = "SINCH_KEY_ID"
	keySecretEnvVar = "SINCH_KEY_SECRET"
	// The calling API is authenticated by an application rather than by the project, so
	// dialling needs a second pair of credentials that buying a number does not.
	applicationKeyEnvVar    = "SINCH_APPLICATION_KEY"
	applicationSecretEnvVar = "SINCH_APPLICATION_SECRET"
)

const (
	defaultBaseURL = "https://numbers.api.sinch.com"
	defaultAuthURL = "https://auth.sinch.com/oauth2/token"
	// defaultCallingBaseURL is the calling API, which is a third host again.
	defaultCallingBaseURL = "https://calling.api.sinch.com"
	defaultTimeout        = 30 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// microsPerDollar converts Sinch's decimal amounts into the micros used everywhere else.
const microsPerDollar = 1_000_000

// tokenMargin is how long before expiry a token is replaced, so a call is not made with one
// that dies in flight.
const tokenMargin = 30 * time.Second

// Options configures a Provider. The credentials fall back to the environment.
type Options struct {
	// ProjectID defaults to SINCH_PROJECT_ID. It is part of every path.
	ProjectID string
	// KeyID defaults to SINCH_KEY_ID.
	KeyID string
	// KeySecret defaults to SINCH_KEY_SECRET.
	KeySecret string
	// ApplicationKey defaults to SINCH_APPLICATION_KEY. Only placing a call needs it.
	ApplicationKey string
	// ApplicationSecret defaults to SINCH_APPLICATION_SECRET.
	ApplicationSecret string
	// BaseURL defaults to Sinch's numbers host.
	BaseURL string
	// AuthURL defaults to Sinch's token host, which is a different one.
	AuthURL string
	// CallingBaseURL defaults to Sinch's calling host, which is a third one.
	CallingBaseURL string
	// Timeout bounds one call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is Sinch. It satisfies phone.Provider.
type Provider struct {
	projectID         string
	keyID             string
	keySecret         string
	applicationKey    string
	applicationSecret string
	baseURL           string
	authURL           string
	callingBaseURL    string
	client            *http.Client

	// mu guards the token, which every call reads and any call may replace.
	mu      sync.Mutex
	token   string
	expires time.Time
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.ProjectID == "" {
		options.ProjectID = os.Getenv(projectIDEnvVar)
	}
	if options.KeyID == "" {
		options.KeyID = os.Getenv(keyIDEnvVar)
	}
	if options.KeySecret == "" {
		options.KeySecret = os.Getenv(keySecretEnvVar)
	}
	if options.ApplicationKey == "" {
		options.ApplicationKey = os.Getenv(applicationKeyEnvVar)
	}
	if options.ApplicationSecret == "" {
		options.ApplicationSecret = os.Getenv(applicationSecretEnvVar)
	}
	if options.ProjectID == "" || options.KeyID == "" || options.KeySecret == "" {
		return nil, errors.New("sinch: " + projectIDEnvVar + ", " + keyIDEnvVar +
			" and " + keySecretEnvVar + " are required")
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.AuthURL == "" {
		options.AuthURL = defaultAuthURL
	}
	if options.CallingBaseURL == "" {
		options.CallingBaseURL = defaultCallingBaseURL
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: options.Timeout}
	}

	return &Provider{
		projectID:         options.ProjectID,
		keyID:             options.KeyID,
		keySecret:         options.KeySecret,
		applicationKey:    options.ApplicationKey,
		applicationSecret: options.ApplicationSecret,
		baseURL:           strings.TrimSuffix(options.BaseURL, "/"),
		authURL:           options.AuthURL,
		callingBaseURL:    strings.TrimSuffix(options.CallingBaseURL, "/"),
		client:            options.HTTPClient,
	}, nil
}

// SearchNumbers returns numbers Sinch is offering in a region.
//
// Sinch requires a number type as well as a country, so a search that does not name one is
// asking for local numbers, which is what an agent wants.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, errors.New("sinch: a country is required to search for numbers")
	}
	// Sinch applies one pattern one way, so it cannot be asked for digits at the front and
	// digits anywhere at the same time.
	if search.Prefix != "" && search.Contains != "" {
		return nil, errors.New("sinch: can match a prefix or a substring, not both")
	}

	kind := search.Type
	if kind == "" {
		kind = phone.Local
	}
	named, ok := kinds[kind]
	if !ok {
		return nil, fmt.Errorf("sinch: does not sell %s numbers", kind)
	}

	query := url.Values{
		"regionCode": {strings.ToUpper(search.Country)},
		"type":       {named},
	}
	if prefix := firstOf(search.Prefix, search.AreaCode); prefix != "" {
		query.Set("numberPattern.pattern", prefix)
		query.Set("numberPattern.searchPattern", "START")
	}
	if search.Contains != "" {
		query.Set("numberPattern.pattern", search.Contains)
		query.Set("numberPattern.searchPattern", "CONTAINS")
	}
	for _, capability := range search.Capabilities {
		switch capability {
		case phone.Voice:
			query.Add("capabilities", "VOICE")
		case phone.SMS:
			query.Add("capabilities", "SMS")
		}
	}
	if search.Limit > 0 {
		query.Set("size", strconv.Itoa(search.Limit))
	}

	var response available
	if err := p.do(ctx, http.MethodGet, p.path("availableNumbers", ""), query, nil, &response); err != nil {
		return nil, err
	}

	offered := make([]phone.Available, 0, len(response.AvailableNumbers))
	for _, number := range response.AvailableNumbers {
		offered = append(offered, phone.Available{
			E164:              number.PhoneNumber,
			Vendor:            p.Vendor(),
			Country:           strings.ToUpper(number.RegionCode),
			Type:              numberType(number.Type),
			Capabilities:      capabilities(number.Capability),
			MonthlyCostMicros: number.MonthlyPrice.micros(),
		})
	}
	return offered, nil
}

// BuyNumber rents a number. Sinch rents by number, so the order's country is not needed.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, errors.New("sinch: a number is required")
	}

	var rented activeNumber
	path := p.path("availableNumbers", order.E164) + ":rent"
	if err := p.do(ctx, http.MethodPost, path, nil, struct{}{}, &rented); err != nil {
		return phone.Number{}, err
	}

	bought := phone.Number{
		E164:              order.E164,
		Vendor:            p.Vendor(),
		Country:           strings.ToUpper(firstOf(rented.RegionCode, order.Country)),
		VendorID:          order.E164,
		Capabilities:      capabilities(rented.Capability),
		MonthlyCostMicros: rented.Money.micros(),
	}
	if rented.PhoneNumber != "" {
		bought.E164 = rented.PhoneNumber
	}
	return bought, nil
}

// ReleaseNumber gives a number back, which is what stops the charge. A number held is an
// active number at Sinch, which is a different resource from the one it was bought from.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	if e164 == "" {
		return errors.New("sinch: a number is required")
	}
	path := p.path("activeNumbers", e164) + ":release"
	return p.do(ctx, http.MethodPost, path, nil, struct{}{}, nil)
}

// ConfigureInbound is not wrapped for Sinch.
func (p *Provider) ConfigureInbound(context.Context, phone.Inbound) error {
	return fmt.Errorf("%w: sinch numbers are bought here but bridged elsewhere", phone.ErrNotImplemented)
}

// Dial calls a person and bridges the answered call to the trunk.
//
// Sinch is the one vendor of the three that will not fetch a plan per call: its callbacks are
// configured once on the application, not named on a request. What it takes instead is the
// plan itself, as a string of SVAML on the callout, and that is what ace carries here. The
// digits are a field on the callout rather than an action in the plan.
func (p *Provider) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if err := outbound.Validate(); err != nil {
		return phone.Dialed{}, fmt.Errorf("sinch: %w", err)
	}
	if p.applicationKey == "" || p.applicationSecret == "" {
		return phone.Dialed{}, fmt.Errorf(
			"sinch: placing a call needs %s and %s, which buying a number does not",
			applicationKeyEnvVar, applicationSecretEnvVar)
	}

	answered, err := json.Marshal(svaml{Action: connectSip{
		Name:        "connectSip",
		Destination: sipEndpoint{Endpoint: strings.TrimPrefix(outbound.Bridge.URI, "sip:")},
		CLI:         outbound.From,
	}})
	if err != nil {
		return phone.Dialed{}, fmt.Errorf("sinch: render answer: %w", err)
	}

	request := calloutRequest{
		Method: "customCallout",
		Custom: custom{
			CLI:         outbound.From,
			Destination: destination{Type: "number", Endpoint: outbound.To},
			ACE:         string(answered),
			DTMF:        outbound.InitialDigits,
		},
	}

	var placed dialedCall
	if err := p.doCalling(ctx, http.MethodPost, "/calling/v1/callouts", request, &placed); err != nil {
		return phone.Dialed{}, err
	}
	if placed.CallID == "" {
		return phone.Dialed{}, errors.New("sinch: the callout came back without a call id")
	}
	return phone.Dialed{VendorCallID: placed.CallID, Status: "queued"}, nil
}

// SendDigits is not wrapped for Sinch, since nothing here places a Sinch call to press on.
func (p *Provider) SendDigits(context.Context, string, string) error {
	return fmt.Errorf("%w: sinch", phone.ErrNotImplemented)
}

// Supports is a country, a pattern either way round and a number type. Sinch's search has no
// city or state filter.
func (p *Provider) Supports(filter phone.Filter) bool {
	switch filter {
	case phone.FilterCountry, phone.FilterAreaCode, phone.FilterPrefix,
		phone.FilterContains, phone.FilterNumberType:
		return true
	default:
		return false
	}
}

// Dials digits on answer only.
//
// A Sinch callout takes the digits to press when the person picks up. What it has instead of
// a ring timeout is a maximum duration for the whole call, which is a different promise, and
// its call headers travel to the SIP leg rather than to the person.
func (p *Provider) Dials(feature phone.CallFeature) bool {
	return feature == phone.FeatureInitialDigits
}

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "sinch" }

// Client exposes the HTTP client, so a caller can reach parts of Sinch's API this does not
// wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// path builds a project-scoped path.
func (p *Provider) path(resource, number string) string {
	base := "/v1/projects/" + url.PathEscape(p.projectID) + "/" + resource
	if number == "" {
		return base
	}
	return base + "/" + url.PathEscape(number)
}

// bearer returns a usable access token, exchanging the key for a new one when the one in hand
// is missing or nearly expired.
func (p *Provider) bearer(ctx context.Context) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.token != "" && time.Now().Before(p.expires.Add(-tokenMargin)) {
		return p.token, nil
	}

	form := url.Values{"grant_type": {"client_credentials"}}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.authURL,
		strings.NewReader(form.Encode()))
	if err != nil {
		return "", fmt.Errorf("sinch: asking for a token: %w", err)
	}
	request.SetBasicAuth(p.keyID, p.keySecret)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return "", fmt.Errorf("sinch: asking for a token: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return "", fmt.Errorf("sinch: asking for a token: %s: %s",
			response.Status, strings.TrimSpace(string(detail)))
	}

	var granted token
	if err := json.NewDecoder(response.Body).Decode(&granted); err != nil {
		return "", fmt.Errorf("sinch: decoding a token: %w", err)
	}
	if granted.AccessToken == "" {
		return "", errors.New("sinch: the token request came back without a token")
	}

	p.token = granted.AccessToken
	p.expires = time.Now().Add(time.Duration(granted.ExpiresIn) * time.Second)
	return p.token, nil
}

// doCalling calls the calling API, which is authenticated by the application key and secret
// rather than by the project token the numbers API uses.
func (p *Provider) doCalling(ctx context.Context, method, path string, payload, into any) error {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("sinch: encode %s: %w", path, err)
	}

	request, err := http.NewRequestWithContext(ctx, method, p.callingBaseURL+path,
		bytes.NewReader(encoded))
	if err != nil {
		return fmt.Errorf("sinch: %s: %w", path, err)
	}
	request.SetBasicAuth(p.applicationKey, p.applicationSecret)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("sinch: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("sinch: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("sinch: decode %s: %w", path, err)
	}
	return nil
}

func (p *Provider) do(ctx context.Context, method, path string, query url.Values, body, into any) error {
	bearer, err := p.bearer(ctx)
	if err != nil {
		return err
	}

	endpoint := p.baseURL + path
	if len(query) > 0 {
		endpoint += "?" + query.Encode()
	}

	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("sinch: encode %s: %w", path, err)
		}
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, payload)
	if err != nil {
		return fmt.Errorf("sinch: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+bearer)
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("sinch: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("sinch: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("sinch: decode %s: %w", path, err)
	}
	return nil
}

// kinds are Sinch's names for the number types this service knows.
var kinds = map[phone.NumberType]string{
	phone.Local:    "LOCAL",
	phone.TollFree: "TOLL_FREE",
	phone.Mobile:   "MOBILE",
}

func numberType(kind string) phone.NumberType {
	for named, sinch := range kinds {
		if sinch == kind {
			return named
		}
	}
	return ""
}

// capabilities maps Sinch's two onto the contract's. Sinch does not sell fax or the rest, so
// there is nothing else to map.
func capabilities(offered []string) []phone.Capability {
	var has []phone.Capability
	for _, capability := range offered {
		switch strings.ToUpper(capability) {
		case "VOICE":
			has = append(has, phone.Voice)
		case "SMS":
			has = append(has, phone.SMS)
		}
	}
	return has
}

func firstOf(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

type token struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
}

// money is Sinch's amount object, whose amount is a decimal string.
type money struct {
	CurrencyCode string `json:"currencyCode"`
	Amount       string `json:"amount"`
}

func (m money) micros() int64 {
	if m.Amount == "" {
		return 0
	}
	amount, err := strconv.ParseFloat(m.Amount, 64)
	if err != nil {
		return 0
	}
	return int64(amount * microsPerDollar)
}

type available struct {
	AvailableNumbers []availableNumber `json:"availableNumbers"`
}

type availableNumber struct {
	PhoneNumber  string   `json:"phoneNumber"`
	RegionCode   string   `json:"regionCode"`
	Type         string   `json:"type"`
	Capability   []string `json:"capability"`
	MonthlyPrice money    `json:"monthlyPrice"`
}

type activeNumber struct {
	PhoneNumber string   `json:"phoneNumber"`
	RegionCode  string   `json:"regionCode"`
	Type        string   `json:"type"`
	Capability  []string `json:"capability"`
	Money       money    `json:"money"`
}

// calloutRequest places a call. Sinch has several kinds of callout and names which by a
// field rather than by a path.
type calloutRequest struct {
	Method string `json:"method"`
	Custom custom `json:"customCallout"`
}

type custom struct {
	CLI         string      `json:"cli"`
	Destination destination `json:"destination"`
	// ACE is the plan to run when the person answers, as a string of SVAML. Sinch's
	// callbacks are configured on the application, so this is the only way to say
	// something about one call.
	ACE string `json:"ace,omitempty"`
	// DTMF is pressed when the person answers.
	DTMF string `json:"dtmf,omitempty"`
}

type destination struct {
	Type     string `json:"type"`
	Endpoint string `json:"endpoint"`
}

// svaml wraps one action, which is the shape Sinch reads a plan in.
type svaml struct {
	Action connectSip `json:"action"`
}

type connectSip struct {
	Name        string      `json:"name"`
	Destination sipEndpoint `json:"destination"`
	CLI         string      `json:"cli,omitempty"`
}

type sipEndpoint struct {
	Endpoint string `json:"endpoint"`
}

type dialedCall struct {
	CallID string `json:"callId"`
}
