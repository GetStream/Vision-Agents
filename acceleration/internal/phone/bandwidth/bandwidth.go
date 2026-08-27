// Package bandwidth buys and releases phone numbers at Bandwidth.
//
// Bandwidth's numbers API is XML in both directions with basic auth, which makes it the odd
// one out here. Numbers are quoted as ten digits with no country code, because the API is
// North American: they are rendered into E.164 on the way out and stripped on the way in.
//
// Buying and giving back are both orders rather than requests against the number, and both
// are fulfilled asynchronously: a successful response means the order was accepted, not that
// the number is usable yet. An order also has to name a site, which is Bandwidth's word for
// the sub-account numbers are billed and configured under.
//
// Placing a call is a different API on a different host, and JSON rather than XML: the voice
// API takes the call and fetches BXML from an answer url when the person picks up. It also
// names a voice application, which is one more credential than buying a number needs.
//
// Pointing a number at a trunk for inbound calls is still not wrapped: that means a SIP peer
// pointed at it, which is separate work from buying the number.
package bandwidth

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
	accountIDEnvVar = "BANDWIDTH_ACCOUNT_ID"
	usernameEnvVar  = "BANDWIDTH_USERNAME"
	passwordEnvVar  = "BANDWIDTH_PASSWORD"
	siteIDEnvVar    = "BANDWIDTH_SITE_ID"
	// applicationIDEnvVar names the voice application a placed call belongs to. Only
	// dialling needs it.
	applicationIDEnvVar = "BANDWIDTH_APPLICATION_ID"
)

const (
	defaultBaseURL = "https://dashboard.bandwidth.com"
	// defaultVoiceBaseURL is the voice API, which is a different host and speaks JSON.
	defaultVoiceBaseURL = "https://voice.bandwidth.com"
	defaultTimeout      = 30 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// defaultQuantity is how many numbers a search asks for when the caller does not say.
const defaultQuantity = 10

// Options configures a Provider. The credentials fall back to the environment.
type Options struct {
	// AccountID defaults to BANDWIDTH_ACCOUNT_ID. It is part of every path.
	AccountID string
	// Username defaults to BANDWIDTH_USERNAME.
	Username string
	// Password defaults to BANDWIDTH_PASSWORD.
	Password string
	// SiteID defaults to BANDWIDTH_SITE_ID. Searching does not need it; ordering does.
	SiteID string
	// ApplicationID defaults to BANDWIDTH_APPLICATION_ID. Only placing a call needs it.
	ApplicationID string
	// BaseURL defaults to Bandwidth's dashboard host.
	BaseURL string
	// VoiceBaseURL defaults to Bandwidth's voice host, which is a different one.
	VoiceBaseURL string
	// Timeout bounds one call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is Bandwidth. It satisfies phone.Provider.
type Provider struct {
	accountID     string
	username      string
	password      string
	siteID        string
	applicationID string
	baseURL       string
	voiceBaseURL  string
	client        *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.AccountID == "" {
		options.AccountID = os.Getenv(accountIDEnvVar)
	}
	if options.Username == "" {
		options.Username = os.Getenv(usernameEnvVar)
	}
	if options.Password == "" {
		options.Password = os.Getenv(passwordEnvVar)
	}
	if options.SiteID == "" {
		options.SiteID = os.Getenv(siteIDEnvVar)
	}
	if options.ApplicationID == "" {
		options.ApplicationID = os.Getenv(applicationIDEnvVar)
	}
	if options.AccountID == "" || options.Username == "" || options.Password == "" {
		return nil, errors.New("bandwidth: " + accountIDEnvVar + ", " + usernameEnvVar +
			" and " + passwordEnvVar + " are required")
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

	return &Provider{
		accountID:     options.AccountID,
		username:      options.Username,
		password:      options.Password,
		siteID:        options.SiteID,
		applicationID: options.ApplicationID,
		baseURL:       strings.TrimSuffix(options.BaseURL, "/"),
		voiceBaseURL:  strings.TrimSuffix(options.VoiceBaseURL, "/"),
		client:        options.HTTPClient,
	}, nil
}

// SearchNumbers returns numbers Bandwidth is offering.
//
// This is Bandwidth's North American inventory, which is the one that takes a city and a
// state, and the reason this vendor can answer a search for Colorado at all. Anywhere else is
// a different endpoint with different filters, so it says so rather than searching the wrong
// inventory.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, errors.New("bandwidth: a country is required to search for numbers")
	}
	if country := strings.ToUpper(search.Country); country != "US" && country != "CA" {
		return nil, fmt.Errorf("bandwidth: only the north american inventory is wrapped here, not %s", country)
	}
	if search.Type != "" && search.Type != phone.Local {
		return nil, fmt.Errorf("bandwidth: only local numbers are wrapped here, not %s", search.Type)
	}
	if search.AreaCode == "" && search.Locality == "" && search.AdministrativeArea == "" {
		return nil, errors.New("bandwidth: a search needs an area code, a city or a state")
	}
	// Bandwidth requires a state alongside a city, because a city name is not unique.
	if search.Locality != "" && search.AdministrativeArea == "" {
		return nil, errors.New("bandwidth: searching a city needs the state it is in")
	}

	// enableTNDetail is what turns a bare list of numbers into ones that say where they
	// are, which is the whole point of searching by city or state.
	query := url.Values{"enableTNDetail": {"true"}}
	if search.AreaCode != "" {
		query.Set("areaCode", search.AreaCode)
	}
	if search.Locality != "" {
		query.Set("city", search.Locality)
	}
	if search.AdministrativeArea != "" {
		query.Set("state", strings.ToUpper(search.AdministrativeArea))
	}
	quantity := search.Limit
	if quantity <= 0 {
		quantity = defaultQuantity
	}
	query.Set("quantity", strconv.Itoa(quantity))

	var response searchResult
	path := "/api/accounts/" + url.PathEscape(p.accountID) + "/availableNumbers"
	if err := p.do(ctx, http.MethodGet, path, query, nil, &response); err != nil {
		return nil, err
	}

	offered := make([]phone.Available, 0, len(response.Details))
	for _, number := range response.Details {
		offered = append(offered, phone.Available{
			E164:     e164(number.FullNumber),
			Vendor:   p.Vendor(),
			Country:  "US",
			Region:   number.State,
			Locality: firstOf(number.City, number.RateCenter),
			Type:     phone.Local,
			// Bandwidth's search says what a number is and where, but not what it
			// costs: pricing lives on the account's rate plan rather than the number.
			Capabilities: []phone.Capability{phone.Voice, phone.SMS},
		})
	}
	// Without enableTNDetail, or on an account that does not return it, the numbers come
	// back as a bare list. Reading both means a search still answers rather than looking
	// like there was no inventory.
	for _, number := range response.Numbers {
		offered = append(offered, phone.Available{
			E164:         e164(number),
			Vendor:       p.Vendor(),
			Country:      "US",
			Type:         phone.Local,
			Capabilities: []phone.Capability{phone.Voice, phone.SMS},
		})
	}
	return offered, nil
}

// BuyNumber orders a number. Bandwidth fulfils orders asynchronously, so this returns once
// the order is accepted rather than when the number is usable.
func (p *Provider) BuyNumber(ctx context.Context, order phone.Order) (phone.Number, error) {
	if order.E164 == "" {
		return phone.Number{}, errors.New("bandwidth: a number is required")
	}
	if p.siteID == "" {
		return phone.Number{}, errors.New("bandwidth: " + siteIDEnvVar +
			" is required to order a number, since an order is billed to a site")
	}

	request := orderRequest{
		Name:   "phone-" + digits(order.E164),
		SiteID: p.siteID,
		Existing: existingNumbers{
			TelephoneNumbers: []string{digits(order.E164)},
		},
	}

	var response orderResponse
	path := "/api/accounts/" + url.PathEscape(p.accountID) + "/orders"
	if err := p.do(ctx, http.MethodPost, path, nil, request, &response); err != nil {
		return phone.Number{}, err
	}

	return phone.Number{
		E164:         order.E164,
		Vendor:       p.Vendor(),
		Country:      "US",
		VendorID:     response.Order.ID,
		Capabilities: []phone.Capability{phone.Voice, phone.SMS},
	}, nil
}

// ReleaseNumber disconnects a number, which is what stops the charge. Bandwidth spells giving
// a number back as another kind of order rather than as a delete.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	if e164 == "" {
		return errors.New("bandwidth: a number is required")
	}

	request := disconnectRequest{
		Name: "release-" + digits(e164),
		Disconnect: disconnectNumbers{
			TelephoneNumbers: []string{digits(e164)},
		},
	}

	path := "/api/accounts/" + url.PathEscape(p.accountID) + "/disconnects"
	return p.do(ctx, http.MethodPost, path, nil, request, nil)
}

// ConfigureInbound is not wrapped for Bandwidth.
//
// A Bandwidth number reaches somewhere by being assigned to a SIP peer on its site, which is
// configuration this does not create. Buying the number does not need it, so this says what
// is missing rather than half-doing it.
func (p *Provider) ConfigureInbound(context.Context, phone.Inbound) error {
	return fmt.Errorf("%w: bandwidth numbers are bought here but bridged elsewhere",
		phone.ErrNotImplemented)
}

// Dial calls a person and has Bandwidth fetch, on answer, the BXML that bridges them to the
// trunk.
//
// Bandwidth will not take a call plan on the request: answerUrl is how a call is controlled
// at all. What that buys is the one thing Twilio's inline TwiML cannot do here, because the
// BXML runs on the person's own leg: a SendDtmf there presses keys at them rather than at the
// trunk, which is what makes an extension behind a menu reachable.
func (p *Provider) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if err := outbound.Validate(); err != nil {
		return phone.Dialed{}, fmt.Errorf("bandwidth: %w", err)
	}
	if outbound.AnswerURL == "" {
		return phone.Dialed{}, errors.New(
			"bandwidth: fetches its call plan when the person answers, so it needs somewhere to fetch it from")
	}
	if p.applicationID == "" {
		return phone.Dialed{}, fmt.Errorf(
			"bandwidth: placing a call needs %s, which buying a number does not",
			applicationIDEnvVar)
	}

	request := callRequest{
		To:            outbound.To,
		From:          outbound.From,
		ApplicationID: p.applicationID,
		AnswerURL:     outbound.AnswerURL,
		AnswerMethod:  http.MethodGet,
		CallTimeout:   outbound.RingTimeout.Seconds(),
	}

	var placed dialedCall
	path := "/api/v2/accounts/" + url.PathEscape(p.accountID) + "/calls"
	if err := p.doVoice(ctx, http.MethodPost, path, request, &placed); err != nil {
		return phone.Dialed{}, err
	}
	return phone.Dialed{VendorCallID: placed.CallID, Status: placed.State}, nil
}

// Answer renders Bandwidth's BXML for pressing the digits and then bridging to the trunk.
//
// This runs on the leg to the person, so the digits go to them. A Bandwidth SipUri has no
// field for a SIP password, which is why its trunk is one that recognises Bandwidth by the
// address it calls from instead.
func (p *Provider) Answer(bridge phone.Bridge, initialDigits string) (phone.Plan, error) {
	if err := bridge.Validate(); err != nil {
		return phone.Plan{}, err
	}

	plan := bxml{Transfer: transfer{SipURI: bridge.URI}}
	if initialDigits != "" {
		plan.SendDtmf = &initialDigits
	}

	rendered, err := xml.Marshal(plan)
	if err != nil {
		return phone.Plan{}, fmt.Errorf("bandwidth: render answer: %w", err)
	}
	return phone.Plan{
		ContentType: "application/xml",
		Body:        append([]byte(xml.Header), rendered...),
	}, nil
}

// SendDigits is not wrapped for Bandwidth, since nothing here places a Bandwidth call.
func (p *Provider) SendDigits(context.Context, string, string) error {
	return fmt.Errorf("%w: bandwidth", phone.ErrNotImplemented)
}

// Supports is a country, an area code, a city and a state, which makes this and Telnyx the
// two vendors that can answer a search for a place. Its digit patterns are fixed-width
// prefixes rather than the free ones the contract describes, so those are not claimed.
func (p *Provider) Supports(filter phone.Filter) bool {
	switch filter {
	case phone.FilterCountry, phone.FilterAreaCode, phone.FilterLocality,
		phone.FilterAdministrativeArea, phone.FilterNumberType:
		return true
	default:
		return false
	}
}

// Dials a ring timeout and digits on answer.
//
// Bandwidth takes callTimeout on the call, and the BXML it fetches runs on the person's own
// leg, so a SendDtmf there presses keys at them. Its only header is User-to-User, which is
// one named header rather than whatever a caller wants to send.
func (p *Provider) Dials(feature phone.CallFeature) bool {
	return feature != phone.FeatureCustomHeaders
}

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "bandwidth" }

// Client exposes the HTTP client, so a caller can reach parts of Bandwidth's API this does
// not wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// doVoice calls the voice API, which is a different host and speaks JSON where the numbers
// API speaks XML.
func (p *Provider) doVoice(ctx context.Context, method, path string, payload, into any) error {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("bandwidth: encode %s: %w", path, err)
	}

	request, err := http.NewRequestWithContext(ctx, method, p.voiceBaseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return fmt.Errorf("bandwidth: %s: %w", path, err)
	}
	request.SetBasicAuth(p.username, p.password)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("bandwidth: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("bandwidth: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("bandwidth: decode %s: %w", path, err)
	}
	return nil
}

func (p *Provider) do(ctx context.Context, method, path string, query url.Values, body, into any) error {
	endpoint := p.baseURL + path
	if len(query) > 0 {
		endpoint += "?" + query.Encode()
	}

	var payload io.Reader
	if body != nil {
		encoded, err := xml.Marshal(body)
		if err != nil {
			return fmt.Errorf("bandwidth: encode %s: %w", path, err)
		}
		payload = strings.NewReader(xml.Header + string(encoded))
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, payload)
	if err != nil {
		return fmt.Errorf("bandwidth: %s: %w", path, err)
	}
	request.SetBasicAuth(p.username, p.password)
	request.Header.Set("Accept", "application/xml")
	if body != nil {
		request.Header.Set("Content-Type", "application/xml")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("bandwidth: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("bandwidth: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := xml.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("bandwidth: decode %s: %w", path, err)
	}
	return nil
}

// e164 renders a Bandwidth number, which is ten digits with the country code left off.
func e164(number string) string {
	switch {
	case number == "", strings.HasPrefix(number, "+"):
		return number
	case len(number) == 10:
		return "+1" + number
	default:
		return "+" + number
	}
}

// digits strips E.164 back to what Bandwidth quotes, which is the number without +1.
func digits(number string) string {
	stripped := strings.TrimPrefix(number, "+")
	if len(stripped) == 11 && strings.HasPrefix(stripped, "1") {
		return stripped[1:]
	}
	return stripped
}

func firstOf(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

type searchResult struct {
	XMLName xml.Name `xml:"SearchResult"`
	// Numbers is the bare list, which is what comes back without number detail.
	Numbers []string `xml:"TelephoneNumberList>TelephoneNumber"`
	// Details is the list with places attached, which is what enableTNDetail asks for.
	Details []numberDetail `xml:"TelephoneNumberDetailList>TelephoneNumberDetail"`
}

type numberDetail struct {
	City       string `xml:"City"`
	State      string `xml:"State"`
	RateCenter string `xml:"RateCenter"`
	FullNumber string `xml:"FullNumber"`
}

type orderRequest struct {
	XMLName  xml.Name        `xml:"Order"`
	Name     string          `xml:"Name"`
	SiteID   string          `xml:"SiteId"`
	Existing existingNumbers `xml:"ExistingTelephoneNumberOrderType"`
}

type existingNumbers struct {
	TelephoneNumbers []string `xml:"TelephoneNumberList>TelephoneNumber"`
}

type orderResponse struct {
	XMLName xml.Name `xml:"OrderResponse"`
	Order   struct {
		ID string `xml:"id"`
	} `xml:"Order"`
	OrderStatus string `xml:"OrderStatus"`
}

type disconnectRequest struct {
	XMLName    xml.Name          `xml:"DisconnectTelephoneNumberOrder"`
	Name       string            `xml:"name"`
	Disconnect disconnectNumbers `xml:"DisconnectTelephoneNumberOrderType"`
}

type disconnectNumbers struct {
	TelephoneNumbers []string `xml:"TelephoneNumberList>TelephoneNumber"`
}

// callRequest places a call whose BXML Bandwidth fetches when the person answers.
type callRequest struct {
	To            string `json:"to"`
	From          string `json:"from"`
	ApplicationID string `json:"applicationId"`
	AnswerURL     string `json:"answerUrl"`
	AnswerMethod  string `json:"answerMethod,omitempty"`
	// CallTimeout is how long to ring before giving up, in seconds, and fractional.
	CallTimeout float64 `json:"callTimeout,omitempty"`
}

type dialedCall struct {
	CallID string `json:"callId"`
	State  string `json:"state"`
}

// bxml is the root of Bandwidth's answer plan. SendDtmf is a pointer so that an empty one is
// left out rather than pressing nothing at the person.
type bxml struct {
	XMLName  xml.Name `xml:"Bxml"`
	SendDtmf *string  `xml:"SendDtmf,omitempty"`
	Transfer transfer `xml:"Transfer"`
}

type transfer struct {
	SipURI string `xml:"SipUri"`
}
