// Package twilio buys and operates phone numbers at Twilio.
//
// Twilio's number API is the 2010-era one: form-encoded requests, basic auth with the
// account SID and auth token, JSON responses. Inbound calls reach an agent by pointing the
// number's voice webhook at TwiML that dials the Stream trunk, and outbound calls are
// created with the same TwiML inline, because Stream cannot dial out itself.
package twilio

import (
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
	accountSIDEnvVar = "TWILIO_ACCOUNT_SID"
	authTokenEnvVar  = "TWILIO_AUTH_TOKEN"
)

const (
	defaultBaseURL = "https://api.twilio.com"
	defaultTimeout = 30 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// microsPerDollar converts Twilio's dollar strings into the micros used everywhere else.
const microsPerDollar = 1_000_000

// Options configures a Provider. The credentials fall back to the environment.
type Options struct {
	// AccountSID defaults to TWILIO_ACCOUNT_SID.
	AccountSID string
	// AuthToken defaults to TWILIO_AUTH_TOKEN.
	AuthToken string
	// BaseURL defaults to Twilio's API host.
	BaseURL string
	// Timeout bounds one call. Buying a number is slower than most, so this is generous.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Provider is Twilio. It satisfies phone.Provider.
type Provider struct {
	accountSID string
	authToken  string
	baseURL    string
	client     *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.AccountSID == "" {
		options.AccountSID = os.Getenv(accountSIDEnvVar)
	}
	if options.AuthToken == "" {
		options.AuthToken = os.Getenv(authTokenEnvVar)
	}
	if options.AccountSID == "" || options.AuthToken == "" {
		return nil, errors.New("twilio: " + accountSIDEnvVar + " and " + authTokenEnvVar + " are required")
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
		accountSID: options.AccountSID,
		authToken:  options.AuthToken,
		baseURL:    strings.TrimSuffix(options.BaseURL, "/"),
		client:     options.HTTPClient,
	}, nil
}

// SearchNumbers returns numbers Twilio is offering in a country.
func (p *Provider) SearchNumbers(ctx context.Context, search phone.Search) ([]phone.Available, error) {
	if search.Country == "" {
		return nil, errors.New("twilio: a country is required to search for numbers")
	}

	query := url.Values{}
	if search.AreaCode != "" {
		query.Set("AreaCode", search.AreaCode)
	}
	if search.Contains != "" {
		query.Set("Contains", search.Contains)
	}
	if search.Limit > 0 {
		query.Set("PageSize", strconv.Itoa(search.Limit))
	}
	// Twilio filters by capability with one flag each rather than a list.
	for _, capability := range search.Capabilities {
		switch capability {
		case phone.Voice:
			query.Set("VoiceEnabled", "true")
		case phone.SMS:
			query.Set("SmsEnabled", "true")
		case phone.MMS:
			query.Set("MmsEnabled", "true")
		case phone.Fax:
			query.Set("FaxEnabled", "true")
		}
	}

	path := fmt.Sprintf("/2010-04-01/Accounts/%s/AvailablePhoneNumbers/%s/Local.json",
		p.accountSID, strings.ToUpper(search.Country))

	var response availableNumbers
	if err := p.get(ctx, path, query, &response); err != nil {
		return nil, err
	}

	offered := make([]phone.Available, 0, len(response.AvailablePhoneNumbers))
	for _, number := range response.AvailablePhoneNumbers {
		offered = append(offered, phone.Available{
			E164:         number.PhoneNumber,
			Country:      number.ISOCountry,
			Region:       number.Region,
			Locality:     number.Locality,
			Capabilities: number.Capabilities.list(),
		})
	}
	return offered, nil
}

// BuyNumber buys a number, which is what starts Twilio charging for it.
func (p *Provider) BuyNumber(ctx context.Context, e164 string) (phone.Number, error) {
	if e164 == "" {
		return phone.Number{}, errors.New("twilio: a number is required")
	}

	form := url.Values{"PhoneNumber": {e164}}
	path := fmt.Sprintf("/2010-04-01/Accounts/%s/IncomingPhoneNumbers.json", p.accountSID)

	var bought incomingNumber
	if err := p.post(ctx, path, form, &bought); err != nil {
		return phone.Number{}, err
	}

	return phone.Number{
		E164:              bought.PhoneNumber,
		Vendor:            p.Vendor(),
		Country:           bought.ISOCountry,
		VendorID:          bought.SID,
		Capabilities:      bought.Capabilities.list(),
		MonthlyCostMicros: dollarsToMicros(bought.MonthlyPrice),
	}, nil
}

// ReleaseNumber gives a number back, which is what stops the charge.
func (p *Provider) ReleaseNumber(ctx context.Context, e164 string) error {
	sid, err := p.sidFor(ctx, e164)
	if err != nil {
		return err
	}

	path := fmt.Sprintf("/2010-04-01/Accounts/%s/IncomingPhoneNumbers/%s.json", p.accountSID, sid)
	return p.do(ctx, http.MethodDelete, path, nil, nil, nil)
}

// ConfigureInbound points the number's voice webhook at TwiML that dials the Stream trunk,
// so an incoming call is bridged straight to the agent.
//
// The TwiML is served inline through Twilio's echo endpoint rather than from a webhook of
// this service's own: the instruction never varies per call, so a server that answers with
// a constant would be one more thing to run and to be down.
func (p *Provider) ConfigureInbound(ctx context.Context, inbound phone.Inbound) error {
	if err := inbound.Bridge.Validate(); err != nil {
		return err
	}
	sid, err := p.sidFor(ctx, inbound.E164)
	if err != nil {
		return err
	}

	instructions, err := dialBridge(inbound.Bridge)
	if err != nil {
		return err
	}

	form := url.Values{
		"VoiceUrl":    {echoURL(instructions)},
		"VoiceMethod": {http.MethodGet},
	}
	path := fmt.Sprintf("/2010-04-01/Accounts/%s/IncomingPhoneNumbers/%s.json", p.accountSID, sid)
	return p.do(ctx, http.MethodPost, path, nil, form, nil)
}

// Dial places a call and bridges the answered leg into the Stream trunk. Stream's SIP is
// inbound only, so the vendor originates and the agent is already waiting on the trunk.
func (p *Provider) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if outbound.From == "" || outbound.To == "" {
		return phone.Dialed{}, errors.New("twilio: a call needs a from and a to")
	}
	if err := outbound.Bridge.Validate(); err != nil {
		return phone.Dialed{}, err
	}

	instructions, err := dialBridge(outbound.Bridge)
	if err != nil {
		return phone.Dialed{}, err
	}

	form := url.Values{
		"From": {outbound.From},
		"To":   {outbound.To},
		// Twiml carries the instruction inline, so answering does not have to fetch it.
		"Twiml": {instructions},
	}
	path := fmt.Sprintf("/2010-04-01/Accounts/%s/Calls.json", p.accountSID)

	var placed call
	if err := p.post(ctx, path, form, &placed); err != nil {
		return phone.Dialed{}, err
	}
	return phone.Dialed{VendorCallID: placed.SID, Status: placed.Status}, nil
}

// Vendor is the name this provider is recorded under.
func (p *Provider) Vendor() string { return "twilio" }

// Client exposes the HTTP client, so a caller can reach parts of Twilio's API this does
// not wrap without building a second client.
func (p *Provider) Client() *http.Client { return p.client }

// sidFor finds Twilio's own identifier for a number this account owns, which is what
// changing or releasing it needs.
func (p *Provider) sidFor(ctx context.Context, e164 string) (string, error) {
	if e164 == "" {
		return "", errors.New("twilio: a number is required")
	}

	path := fmt.Sprintf("/2010-04-01/Accounts/%s/IncomingPhoneNumbers.json", p.accountSID)

	var owned incomingNumbers
	if err := p.get(ctx, path, url.Values{"PhoneNumber": {e164}}, &owned); err != nil {
		return "", err
	}
	for _, number := range owned.IncomingPhoneNumbers {
		if number.PhoneNumber == e164 {
			return number.SID, nil
		}
	}
	return "", fmt.Errorf("twilio: %s is not one of this account's numbers", e164)
}

func (p *Provider) get(ctx context.Context, path string, query url.Values, into any) error {
	return p.do(ctx, http.MethodGet, path, query, nil, into)
}

func (p *Provider) post(ctx context.Context, path string, form url.Values, into any) error {
	return p.do(ctx, http.MethodPost, path, nil, form, into)
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
		return fmt.Errorf("twilio: %s: %w", path, err)
	}
	request.SetBasicAuth(p.accountSID, p.authToken)
	request.Header.Set("Accept", "application/json")
	if form != nil {
		request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("twilio: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("twilio: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("twilio: decode %s: %w", path, err)
	}
	return nil
}

// dialBridge renders the TwiML that connects a call to the Stream trunk.
func dialBridge(bridge phone.Bridge) (string, error) {
	if err := bridge.Validate(); err != nil {
		return "", err
	}

	instructions := twiml{
		Dial: twimlDial{
			SIP: twimlSIP{
				URI:      bridge.URI,
				Username: bridge.Username,
				Password: bridge.Password,
			},
		},
	}
	encoded, err := xml.Marshal(instructions)
	if err != nil {
		return "", fmt.Errorf("twilio: encode twiml: %w", err)
	}
	return xml.Header + string(encoded), nil
}

// echoURL wraps TwiML in Twilio's echo endpoint, which answers a webhook with whatever
// was put in it. It is what lets a fixed instruction be configured without hosting it.
func echoURL(instructions string) string {
	return "https://twimlets.com/echo?Twiml=" + url.QueryEscape(instructions)
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

type twiml struct {
	XMLName xml.Name  `xml:"Response"`
	Dial    twimlDial `xml:"Dial"`
}

type twimlDial struct {
	SIP twimlSIP `xml:"Sip"`
}

type twimlSIP struct {
	URI      string `xml:",chardata"`
	Username string `xml:"username,attr,omitempty"`
	Password string `xml:"password,attr,omitempty"`
}

// capabilities is Twilio's per-number capability object, which is flags rather than a list.
type capabilities struct {
	Voice bool `json:"voice"`
	SMS   bool `json:"SMS"`
	MMS   bool `json:"MMS"`
	Fax   bool `json:"fax"`
}

func (c capabilities) list() []phone.Capability {
	var has []phone.Capability
	if c.Voice {
		has = append(has, phone.Voice)
	}
	if c.SMS {
		has = append(has, phone.SMS)
	}
	if c.MMS {
		has = append(has, phone.MMS)
	}
	if c.Fax {
		has = append(has, phone.Fax)
	}
	return has
}

type availableNumbers struct {
	AvailablePhoneNumbers []struct {
		PhoneNumber  string       `json:"phone_number"`
		ISOCountry   string       `json:"iso_country"`
		Region       string       `json:"region"`
		Locality     string       `json:"locality"`
		Capabilities capabilities `json:"capabilities"`
	} `json:"available_phone_numbers"`
}

type incomingNumber struct {
	SID          string       `json:"sid"`
	PhoneNumber  string       `json:"phone_number"`
	ISOCountry   string       `json:"iso_country"`
	Capabilities capabilities `json:"capabilities"`
	// Twilio does not quote a price on the number itself, so this is only set when a
	// deployment's account exposes one.
	MonthlyPrice string `json:"monthly_price"`
}

type incomingNumbers struct {
	IncomingPhoneNumbers []incomingNumber `json:"incoming_phone_numbers"`
}

type call struct {
	SID    string `json:"sid"`
	Status string `json:"status"`
}
