// Package bird buys and releases phone numbers at Bird.
//
// Bird's API is JSON with a bearer access key, and the key says which region it belongs to:
// a key beginning bk_us1_ is only valid against us1.platform.bird.com, so the host is read
// off the key rather than configured separately and got wrong.
//
// Buying is an order rather than a purchase. It usually completes inside the request, and it
// carries an idempotency key so that a retry after a timeout cannot buy the number twice.
//
// Placing a call is the older MessageBird voice API on its own host, authenticated with the
// same key under a different scheme, because Bird has not moved voice onto the platform API.
// The call carries its steps inline, so nothing has to be hosted to answer it.
//
// Pointing a number at a trunk for inbound calls is still not wrapped: that is a Bird flow
// rather than a property of the number.
package bird

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
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

const accessKeyEnvVar = "BIRD_ACCESS_KEY"

// defaultVoiceBaseURL is the voice API, which is not on the platform host and is not
// regional.
const defaultVoiceBaseURL = "https://voice.messagebird.com"

const defaultTimeout = 30 * time.Second

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// microsPerDollar converts Bird's decimal prices into the micros used everywhere else.
const microsPerDollar = 1_000_000

// Options configures a Provider. The key falls back to the environment.
type Options struct {
	// AccessKey defaults to BIRD_ACCESS_KEY. Its bk_{region}_ prefix decides the host.
	AccessKey string
	// BaseURL overrides the host the key implies, which is what the tests use.
	BaseURL string
	// VoiceBaseURL defaults to the voice API host, which is a different one.
	VoiceBaseURL string
	// Timeout bounds one call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is Bird. It satisfies phone.Provider.
type Provider struct {
	accessKey    string
	baseURL      string
	voiceBaseURL string
	client       *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.AccessKey == "" {
		options.AccessKey = os.Getenv(accessKeyEnvVar)
	}
	if options.AccessKey == "" {
		return nil, errors.New("bird: " + accessKeyEnvVar + " is required")
	}
	if options.BaseURL == "" {
		host, err := hostFor(options.AccessKey)
		if err != nil {
			return nil, err
		}
		options.BaseURL = host
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

	return &Provider{
		accessKey:    options.AccessKey,
		baseURL:      strings.TrimSuffix(options.BaseURL, "/"),
		voiceBaseURL: strings.TrimSuffix(options.VoiceBaseURL, "/"),
		client:       options.HTTPClient,
	}, nil
}

// hostFor reads the region out of a Bird key, since a key is only valid against its own
// region's host and using the wrong one fails as an authentication error rather than saying
// what is actually wrong.
func hostFor(accessKey string) (string, error) {
	parts := strings.Split(accessKey, "_")
	if len(parts) < 3 || parts[0] != "bk" || parts[1] == "" {
		return "", fmt.Errorf("bird: %s does not name a region, so there is no host to reach", accessKeyEnvVar)
	}
	return "https://" + parts[1] + ".platform.bird.com", nil
}

// SearchNumbers returns numbers Bird is offering in a country.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, errors.New("bird: a country is required to search for numbers")
	}

	query := url.Values{"country_code": {strings.ToUpper(search.Country)}}
	// Bird's prefix is matched right after the country dial code, so an area code is a
	// prefix to it.
	if prefix := firstOf(search.Prefix, search.AreaCode); prefix != "" {
		query.Set("prefix", prefix)
	}
	if search.Type != "" {
		kind, ok := kinds[search.Type]
		if !ok {
			return nil, fmt.Errorf("bird: does not sell %s numbers", search.Type)
		}
		query.Set("number_type", kind)
	}
	// Bird repeats the parameter to require several capabilities at once.
	for _, capability := range search.Capabilities {
		switch capability {
		case phone.Voice, phone.SMS, phone.MMS:
			query.Add("capabilities", string(capability))
		}
	}
	if search.Limit > 0 {
		query.Set("limit", strconv.Itoa(search.Limit))
	}

	var response page[availableNumber]
	if err := p.do(ctx, http.MethodGet, "/v1/numbers/available", query, nil, &response); err != nil {
		return nil, err
	}

	offered := make([]phone.Available, 0, len(response.Data))
	for _, number := range response.Data {
		offered = append(offered, phone.Available{
			E164:              number.Number,
			Vendor:            p.Vendor(),
			Country:           strings.ToUpper(number.CountryCode),
			Type:              numberType(number.NumberType),
			Capabilities:      capabilities(number.Capabilities),
			MonthlyCostMicros: number.MonthlyPrice.micros(),
		})
	}
	return offered, nil
}

// BuyNumber orders a number. Bird buys by number, so the order's country is not needed.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, errors.New("bird: a number is required")
	}

	key, err := idempotencyKey()
	if err != nil {
		return phone.Number{}, err
	}

	var placed numberOrder
	headers := http.Header{"Idempotency-Key": {key}}
	err = p.send(ctx, http.MethodPost, "/v1/numbers/orders", nil,
		orderRequest{Number: order.E164}, headers, &placed)
	if err != nil {
		return phone.Number{}, err
	}

	bought := phone.Number{
		E164:     order.E164,
		Vendor:   p.Vendor(),
		Country:  strings.ToUpper(order.Country),
		VendorID: placed.ID,
	}
	if placed.Number != "" {
		bought.E164 = placed.Number
	}
	return bought, nil
}

// ReleaseNumber gives a number back, which is what stops the charge. Bird releases by its
// own identifier rather than by number, so this finds it first.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	id, err := p.idFor(ctx, e164)
	if err != nil {
		return err
	}
	return p.do(ctx, http.MethodDelete, "/v1/numbers/"+url.PathEscape(id), nil, nil, nil)
}

// ConfigureInbound is not wrapped for Bird.
func (p *Provider) ConfigureInbound(context.Context, phone.Inbound) error {
	return fmt.Errorf("%w: bird numbers are bought here but bridged elsewhere", phone.ErrNotImplemented)
}

// Dial calls a person and transfers the answered leg to the Stream trunk.
//
// The steps travel with the call rather than being fetched, so nothing has to be hosted for
// this to work. A transfer step carries only where to transfer to, with no field for a SIP
// password, so the trunk has to recognise Bird by the address it calls from; the service
// arranges that before dialling.
func (p *Provider) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if err := outbound.Validate(); err != nil {
		return phone.Dialed{}, fmt.Errorf("bird: %w", err)
	}

	request := callRequest{
		// Bird quotes numbers without a plus on this API.
		Source:      digits(outbound.From),
		Destination: digits(outbound.To),
		CallFlow: callFlow{
			Title: "bridge to stream",
			Steps: []step{{
				Action:  "transfer",
				Options: stepOptions{Destination: outbound.Bridge.URI},
			}},
		},
	}

	var response voiceEnvelope[dialedCall]
	if err := p.doVoice(ctx, http.MethodPost, "/calls", request, &response); err != nil {
		return phone.Dialed{}, err
	}
	if len(response.Data) == 0 {
		return phone.Dialed{}, errors.New("bird: placed no call")
	}
	return phone.Dialed{
		VendorCallID: response.Data[0].ID,
		Status:       response.Data[0].Status,
	}, nil
}

// SendDigits is not wrapped for Bird, since nothing here places a Bird call to press on.
func (p *Provider) SendDigits(context.Context, string, string) error {
	return fmt.Errorf("%w: bird", phone.ErrNotImplemented)
}

// Supports is a country, an anchored prefix and a number type. Bird's search knows nothing
// of cities or states.
func (p *Provider) Supports(filter phone.Filter) bool {
	switch filter {
	case phone.FilterCountry, phone.FilterAreaCode, phone.FilterPrefix, phone.FilterNumberType:
		return true
	default:
		return false
	}
}

// Dials nothing beyond the call itself.
//
// A Bird call is a source, a destination and the steps to run when it is answered. There is
// no ring timeout on it, and a transfer step carries only where to transfer to.
func (p *Provider) Dials(phone.CallFeature) bool { return false }

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "bird" }

// Client exposes the HTTP client, so a caller can reach parts of Bird's API this does not
// wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// idFor finds Bird's own identifier for a number this workspace holds, which is what
// releasing it needs.
func (p *Provider) idFor(ctx context.Context, e164 string) (string, error) {
	if e164 == "" {
		return "", errors.New("bird: a number is required")
	}

	var response page[heldNumber]
	if err := p.do(ctx, http.MethodGet, "/v1/numbers", nil, nil, &response); err != nil {
		return "", err
	}
	for _, number := range response.Data {
		if number.Number == e164 {
			return number.ID, nil
		}
	}
	return "", fmt.Errorf("bird: %s is not one of this workspace's numbers", e164)
}

func (p *Provider) do(ctx context.Context, method, path string, query url.Values, body, into any) error {
	return p.send(ctx, method, path, query, body, nil, into)
}

// doVoice calls the voice API, which is a different host and names its scheme AccessKey
// rather than Bearer.
func (p *Provider) doVoice(ctx context.Context, method, path string, payload, into any) error {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("bird: encode %s: %w", path, err)
	}

	request, err := http.NewRequestWithContext(ctx, method, p.voiceBaseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return fmt.Errorf("bird: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "AccessKey "+p.accessKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("bird: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("bird: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("bird: decode %s: %w", path, err)
	}
	return nil
}

func (p *Provider) send(
	ctx context.Context,
	method, path string,
	query url.Values,
	body any,
	headers http.Header,
	into any,
) error {
	endpoint := p.baseURL + path
	if len(query) > 0 {
		endpoint += "?" + query.Encode()
	}

	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("bird: encode %s: %w", path, err)
		}
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, payload)
	if err != nil {
		return fmt.Errorf("bird: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+p.accessKey)
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	for name, values := range headers {
		for _, value := range values {
			request.Header.Set(name, value)
		}
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("bird: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("bird: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("bird: decode %s: %w", path, err)
	}
	return nil
}

// idempotencyKey is what stops a retried order from buying twice.
func idempotencyKey() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("bird: an order needs an idempotency key: %w", err)
	}
	return hex.EncodeToString(raw), nil
}

// kinds are Bird's names for the number types this service knows.
var kinds = map[phone.NumberType]string{
	phone.Local:    "local",
	phone.TollFree: "toll_free",
	phone.Mobile:   "mobile",
}

func numberType(kind string) phone.NumberType {
	for named, bird := range kinds {
		if bird == kind {
			return named
		}
	}
	return ""
}

func capabilities(offered []string) []phone.Capability {
	var has []phone.Capability
	for _, capability := range offered {
		switch phone.Capability(capability) {
		case phone.Voice, phone.SMS, phone.MMS, phone.Fax:
			has = append(has, phone.Capability(capability))
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

// digits drops the leading plus, which the voice API does not want even though the number
// API quotes numbers with one.
func digits(number string) string { return strings.TrimPrefix(number, "+") }

// page is Bird's list envelope.
type page[T any] struct {
	Data []T `json:"data"`
}

// voiceEnvelope is the voice API's list envelope, which answers a single placed call with a
// list of one.
type voiceEnvelope[T any] struct {
	Data []T `json:"data"`
}

// callRequest places a call whose steps travel with it.
type callRequest struct {
	Source      string   `json:"source"`
	Destination string   `json:"destination"`
	CallFlow    callFlow `json:"callFlow"`
}

// callFlow is what to do once the person answers.
type callFlow struct {
	Title string `json:"title"`
	Steps []step `json:"steps"`
}

type step struct {
	Action  string      `json:"action"`
	Options stepOptions `json:"options"`
}

type stepOptions struct {
	// Destination is where to transfer to, which takes a SIP uri as readily as a number.
	Destination string `json:"destination"`
}

// dialedCall is what Bird answers a placed call with.
type dialedCall struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

// price is Bird's money object. It is absent on numbers Bird does not quote a price for,
// which reads as zero, the same as everywhere else here.
type price struct {
	Amount   string `json:"amount"`
	Currency string `json:"currency"`
}

func (p price) micros() int64 {
	if p.Amount == "" {
		return 0
	}
	amount, err := strconv.ParseFloat(p.Amount, 64)
	if err != nil {
		return 0
	}
	return int64(amount * microsPerDollar)
}

type availableNumber struct {
	Number       string   `json:"number"`
	CountryCode  string   `json:"country_code"`
	NumberType   string   `json:"number_type"`
	Capabilities []string `json:"capabilities"`
	MonthlyPrice price    `json:"monthly_price"`
}

type heldNumber struct {
	ID     string `json:"id"`
	Number string `json:"number"`
}

type orderRequest struct {
	Number string `json:"number"`
}

type numberOrder struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Number string `json:"number"`
}
