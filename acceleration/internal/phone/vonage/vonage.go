// Package vonage buys and releases phone numbers at Vonage.
//
// Vonage's number API is the old Nexmo one: form-encoded requests, basic auth on the api key
// and secret, JSON responses, and numbers quoted without a leading plus.
//
// Buying and cancelling both name a country as well as a number, because Vonage sells out of
// a country's inventory rather than by number alone. Buying gets it from the order, which the
// search that found the number filled in; cancelling looks it up among the numbers this
// account already holds.
//
// Placing a call is a different API on a different host with different credentials: the voice
// API is authenticated by a JWT signed with an application's private key rather than by the
// key and secret, which is why dialling needs two more environment variables than buying. The
// call carries its NCCO inline, so nothing has to be hosted to answer it.
//
// Pointing a number at a trunk for inbound calls is still not wrapped: that means a Vonage
// application whose answer_url this service serves, which is more than buying a number.
package vonage

import (
	"bytes"
	"context"
	"crypto/rsa"
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

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

const (
	apiKeyEnvVar        = "VONAGE_API_KEY"
	apiSecretEnvVar     = "VONAGE_API_SECRET"
	applicationIDEnvVar = "VONAGE_APPLICATION_ID"
	privateKeyEnvVar    = "VONAGE_PRIVATE_KEY"
)

const (
	defaultBaseURL = "https://rest.nexmo.com"
	// defaultVoiceBaseURL is the voice API, which lives on a different host from the
	// number API and is authenticated differently.
	defaultVoiceBaseURL = "https://api.nexmo.com"
	defaultTimeout      = 30 * time.Second
)

// Vonage's ringing timer is bounded, and a call outside the bounds is rejected.
const (
	minRingSeconds = 1
	maxRingSeconds = 120
)

// tokenLifetime is how long a voice API token is good for. It is minted per request, so this
// only has to outlive the request it is sent on.
const tokenLifetime = 2 * time.Minute

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// microsPerDollar converts Vonage's cost strings into the micros used everywhere else. The
// figure is in the account's own currency, which a deployment is assumed to keep in dollars.
const microsPerDollar = 1_000_000

// Vonage names its search strategies with numbers rather than words.
const (
	patternStartsWith = "0"
	patternAnywhere   = "1"
)

// Options configures a Provider. The credentials fall back to the environment.
type Options struct {
	// APIKey defaults to VONAGE_API_KEY.
	APIKey string
	// APISecret defaults to VONAGE_API_SECRET.
	APISecret string
	// ApplicationID defaults to VONAGE_APPLICATION_ID. Only placing a call needs it.
	ApplicationID string
	// PrivateKey defaults to VONAGE_PRIVATE_KEY, which may be the PEM itself or the path
	// to a file holding it, since a PEM in an environment variable is awkward. Only
	// placing a call needs it.
	PrivateKey string
	// BaseURL defaults to Vonage's number API host.
	BaseURL string
	// VoiceBaseURL defaults to Vonage's voice API host, which is a different one.
	VoiceBaseURL string
	// Timeout bounds one call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is Vonage. It satisfies phone.Provider.
type Provider struct {
	apiKey        string
	apiSecret     string
	applicationID string
	privateKey    *rsa.PrivateKey
	baseURL       string
	voiceBaseURL  string
	client        *http.Client
}

// New validates the options and returns a Provider.
//
// The voice credentials are optional: a deployment that only buys numbers from Vonage should
// not have to hold an application's private key, so a missing one is refused at the call
// rather than here.
func New(options Options) (*Provider, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APISecret == "" {
		options.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if options.APIKey == "" || options.APISecret == "" {
		return nil, errors.New("vonage: " + apiKeyEnvVar + " and " + apiSecretEnvVar + " are required")
	}
	if options.ApplicationID == "" {
		options.ApplicationID = os.Getenv(applicationIDEnvVar)
	}
	if options.PrivateKey == "" {
		options.PrivateKey = os.Getenv(privateKeyEnvVar)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.VoiceBaseURL == "" {
		options.VoiceBaseURL = defaultVoiceBaseURL
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: options.Timeout}
	}

	provider := &Provider{
		apiKey:        options.APIKey,
		apiSecret:     options.APISecret,
		applicationID: options.ApplicationID,
		baseURL:       strings.TrimSuffix(options.BaseURL, "/"),
		voiceBaseURL:  strings.TrimSuffix(options.VoiceBaseURL, "/"),
		client:        options.HTTPClient,
	}
	if options.PrivateKey != "" {
		key, err := parsePrivateKey(options.PrivateKey)
		if err != nil {
			return nil, err
		}
		provider.privateKey = key
	}
	return provider, nil
}

// parsePrivateKey reads the application key, from either the PEM itself or a file holding it.
func parsePrivateKey(value string) (*rsa.PrivateKey, error) {
	pem := []byte(value)
	if !strings.Contains(value, "-----BEGIN") {
		read, err := os.ReadFile(value)
		if err != nil {
			return nil, fmt.Errorf("vonage: %s is neither a pem nor a readable file: %w",
				privateKeyEnvVar, err)
		}
		pem = read
	}

	key, err := jwt.ParseRSAPrivateKeyFromPEM(pem)
	if err != nil {
		return nil, fmt.Errorf("vonage: %s is not an rsa private key: %w", privateKeyEnvVar, err)
	}
	return key, nil
}

// SearchNumbers returns numbers Vonage is offering in a country.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, errors.New("vonage: a country is required to search for numbers")
	}
	// Vonage matches one pattern one way, so it cannot be asked for digits at the front
	// and digits anywhere in the same breath.
	if search.Prefix != "" && search.Contains != "" {
		return nil, errors.New("vonage: can match a prefix or a substring, not both")
	}

	query := url.Values{"country": {strings.ToUpper(search.Country)}}
	if pattern := firstOf(search.Prefix, search.AreaCode); pattern != "" {
		query.Set("pattern", pattern)
		query.Set("search_pattern", patternStartsWith)
	}
	if search.Contains != "" {
		query.Set("pattern", search.Contains)
		query.Set("search_pattern", patternAnywhere)
	}
	if search.Type != "" {
		kind, ok := kinds[search.Type]
		if !ok {
			return nil, fmt.Errorf("vonage: does not sell %s numbers", search.Type)
		}
		query.Set("type", kind)
	}
	if wanted := features(search.Capabilities); wanted != "" {
		query.Set("features", wanted)
	}
	if search.Limit > 0 {
		query.Set("size", strconv.Itoa(search.Limit))
	}

	var response numbers
	if err := p.do(ctx, http.MethodGet, "/number/search", query, nil, &response); err != nil {
		return nil, err
	}

	offered := make([]phone.Available, 0, len(response.Numbers))
	for _, number := range response.Numbers {
		offered = append(offered, phone.Available{
			E164:              e164(number.MSISDN),
			Vendor:            p.Vendor(),
			Country:           strings.ToUpper(number.Country),
			Type:              numberType(number.Type),
			Capabilities:      capabilities(number.Features),
			MonthlyCostMicros: dollarsToMicros(number.Cost),
		})
	}
	return offered, nil
}

// BuyNumber buys a number out of a country's inventory, which is why the order names one.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, errors.New("vonage: a number is required")
	}
	if order.Country == "" {
		return phone.Number{}, errors.New("vonage: buying a number needs the country it is sold in")
	}

	form := url.Values{
		"country": {strings.ToUpper(order.Country)},
		"msisdn":  {digits(order.E164)},
	}
	if err := p.do(ctx, http.MethodPost, "/number/buy", nil, form, nil); err != nil {
		return phone.Number{}, err
	}

	// Vonage answers a purchase with a status rather than the number, so what is known
	// about it is what was ordered. Its own identifier for a number is the number.
	return phone.Number{
		E164:     order.E164,
		Vendor:   p.Vendor(),
		Country:  strings.ToUpper(order.Country),
		VendorID: digits(order.E164),
	}, nil
}

// ReleaseNumber cancels a number, which is what stops the charge. Cancelling names the
// country too, so this finds it among the numbers this account holds.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	country, err := p.countryFor(ctx, e164)
	if err != nil {
		return err
	}

	form := url.Values{"country": {country}, "msisdn": {digits(e164)}}
	return p.do(ctx, http.MethodPost, "/number/cancel", nil, form, nil)
}

// ConfigureInbound is not wrapped for Vonage.
//
// Vonage routes an incoming call by handing it to an application that answers with an NCCO,
// so pointing a number at a Stream trunk means hosting that NCCO. Buying the number does not.
func (p *Provider) ConfigureInbound(context.Context, phone.Inbound) error {
	return fmt.Errorf("%w: vonage numbers are bought here but bridged elsewhere", phone.ErrNotImplemented)
}

// Dial calls a person and connects the answered leg to the Stream trunk.
//
// The NCCO travels with the call rather than being fetched, so nothing has to be hosted for
// this to work. Vonage cannot send a SIP password from a connect action, so the trunk has to
// recognise Vonage by the address it calls from; the service arranges that before dialling.
func (p *Provider) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if err := outbound.Validate(); err != nil {
		return phone.Dialed{}, fmt.Errorf("vonage: %w", err)
	}
	if p.applicationID == "" || p.privateKey == nil {
		return phone.Dialed{}, fmt.Errorf(
			"vonage: placing a call needs %s and %s, which buying a number does not",
			applicationIDEnvVar, privateKeyEnvVar)
	}

	request := callRequest{
		To:   []endpoint{{Type: "phone", Number: digits(outbound.To)}},
		From: endpoint{Type: "phone", Number: digits(outbound.From)},
		NCCO: []action{{
			Action:   "connect",
			Endpoint: []endpoint{{Type: "sip", URI: outbound.Bridge.URI}},
		}},
	}
	if outbound.RingTimeout > 0 {
		seconds := int(outbound.RingTimeout.Seconds())
		if seconds < minRingSeconds || seconds > maxRingSeconds {
			return phone.Dialed{}, fmt.Errorf(
				"vonage: %ds is outside the %d-%ds vonage will ring for",
				seconds, minRingSeconds, maxRingSeconds)
		}
		request.RingingTimer = seconds
	}

	var placed dialedCall
	if err := p.doVoice(ctx, http.MethodPost, "/v1/calls", request, &placed); err != nil {
		return phone.Dialed{}, err
	}
	return phone.Dialed{VendorCallID: placed.UUID, Status: placed.Status}, nil
}

// SendDigits is not wrapped for Vonage, since nothing here places a Vonage call to press on.
func (p *Provider) SendDigits(context.Context, string, string) error {
	return fmt.Errorf("%w: vonage", phone.ErrNotImplemented)
}

// Supports is a country, a pattern either way round, and a number type. Vonage's search has
// no idea what a city or a state is.
func (p *Provider) Supports(filter phone.Filter) bool {
	switch filter {
	case phone.FilterCountry, phone.FilterAreaCode, phone.FilterPrefix,
		phone.FilterContains, phone.FilterNumberType:
		return true
	default:
		return false
	}
}

// Dials a ring timeout only.
//
// Vonage takes ringing_timer on the call. Its dtmfAnswer and its SIP headers both belong to
// an endpoint inside a connect action, which here is the trunk rather than the person, so
// neither reaches the leg they would have to reach.
func (p *Provider) Dials(feature phone.CallFeature) bool {
	return feature == phone.FeatureRingTimeout
}

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "vonage" }

// Client exposes the HTTP client, so a caller can reach parts of Vonage's API this does not
// wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// countryFor finds which country a number this account holds was sold in, which is what
// cancelling it needs.
func (p *Provider) countryFor(ctx context.Context, e164 string) (string, error) {
	if e164 == "" {
		return "", errors.New("vonage: a number is required")
	}

	query := url.Values{
		"pattern":        {digits(e164)},
		"search_pattern": {patternAnywhere},
	}

	var response numbers
	if err := p.do(ctx, http.MethodGet, "/account/numbers", query, nil, &response); err != nil {
		return "", err
	}
	for _, number := range response.Numbers {
		if digits(number.MSISDN) == digits(e164) {
			return strings.ToUpper(number.Country), nil
		}
	}
	return "", fmt.Errorf("vonage: %s is not one of this account's numbers", e164)
}

func (p *Provider) do(ctx context.Context, method, path string, query, form url.Values, into any) error {
	endpoint := p.baseURL + path
	if len(query) > 0 {
		endpoint += "?" + query.Encode()
	}

	var body io.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, body)
	if err != nil {
		return fmt.Errorf("vonage: %s: %w", path, err)
	}
	request.SetBasicAuth(p.apiKey, p.apiSecret)
	request.Header.Set("Accept", "application/json")
	if form != nil {
		request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("vonage: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("vonage: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("vonage: decode %s: %w", path, err)
	}
	return nil
}

// doVoice calls the voice API, which takes JSON and a bearer token rather than a form and
// basic auth.
func (p *Provider) doVoice(ctx context.Context, method, path string, payload, into any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("vonage: encode %s: %w", path, err)
	}
	token, err := p.token()
	if err != nil {
		return err
	}

	request, err := http.NewRequestWithContext(ctx, method, p.voiceBaseURL+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("vonage: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("vonage: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("vonage: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("vonage: decode %s: %w", path, err)
	}
	return nil
}

// token mints a voice API token, which names the application and is signed with its key.
//
// A fresh one per request costs one RSA signature and avoids holding a token that has to be
// noticed expiring, which for a call placed every few minutes is the better trade.
func (p *Provider) token() (string, error) {
	now := time.Now()
	claims := jwt.MapClaims{
		"application_id": p.applicationID,
		"iat":            now.Unix(),
		"exp":            now.Add(tokenLifetime).Unix(),
		// Vonage rejects a token it has seen before, so each one is named.
		"jti": uuid.NewString(),
	}

	token, err := jwt.NewWithClaims(jwt.SigningMethodRS256, claims).SignedString(p.privateKey)
	if err != nil {
		return "", fmt.Errorf("vonage: sign voice token: %w", err)
	}
	return token, nil
}

// kinds are Vonage's names for the number types this service knows. Vonage calls a
// geographic number a landline and a mobile one a long virtual number.
var kinds = map[phone.NumberType]string{
	phone.Local:    "landline",
	phone.TollFree: "landline-toll-free",
	phone.Mobile:   "mobile-lvn",
}

func numberType(kind string) phone.NumberType {
	for named, vonage := range kinds {
		if vonage == kind {
			return named
		}
	}
	return ""
}

// features renders the capabilities Vonage can filter on, which it names in capitals and
// separates with commas.
func features(wanted []phone.Capability) string {
	var named []string
	for _, capability := range wanted {
		switch capability {
		case phone.Voice:
			named = append(named, "VOICE")
		case phone.SMS:
			named = append(named, "SMS")
		case phone.MMS:
			named = append(named, "MMS")
		}
	}
	return strings.Join(named, ",")
}

func capabilities(offered []string) []phone.Capability {
	var has []phone.Capability
	for _, feature := range offered {
		switch strings.ToUpper(feature) {
		case "VOICE":
			has = append(has, phone.Voice)
		case "SMS":
			has = append(has, phone.SMS)
		case "MMS":
			has = append(has, phone.MMS)
		}
	}
	return has
}

// e164 renders a Vonage number, which is quoted without a plus.
func e164(number string) string {
	if number == "" || strings.HasPrefix(number, "+") {
		return number
	}
	return "+" + number
}

func digits(number string) string { return strings.TrimPrefix(number, "+") }

func firstOf(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func dollarsToMicros(cost string) int64 {
	if cost == "" {
		return 0
	}
	amount, err := strconv.ParseFloat(cost, 64)
	if err != nil {
		return 0
	}
	return int64(amount * microsPerDollar)
}

// callRequest places a call whose instructions travel with it.
type callRequest struct {
	To   []endpoint `json:"to"`
	From endpoint   `json:"from"`
	// NCCO is the call plan, carried inline so answering does not have to fetch it.
	NCCO []action `json:"ncco"`
	// RingingTimer is how long to ring before giving up, in seconds.
	RingingTimer int `json:"ringing_timer,omitempty"`
}

// endpoint is one end of a call: a telephone, or a SIP address.
type endpoint struct {
	Type   string `json:"type"`
	Number string `json:"number,omitempty"`
	URI    string `json:"uri,omitempty"`
}

// action is one step of an NCCO.
type action struct {
	Action   string     `json:"action"`
	Endpoint []endpoint `json:"endpoint,omitempty"`
}

// dialedCall is what Vonage answers a placed call with.
type dialedCall struct {
	UUID   string `json:"uuid"`
	Status string `json:"status"`
}

// numbers is what both the search and the owned-number listing answer with.
type numbers struct {
	Numbers []number `json:"numbers"`
}

type number struct {
	Country  string   `json:"country"`
	MSISDN   string   `json:"msisdn"`
	Type     string   `json:"type"`
	Cost     string   `json:"cost"`
	Features []string `json:"features"`
}
