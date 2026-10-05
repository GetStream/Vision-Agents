// Package telnyx registers brands and 10DLC campaigns at Telnyx.
//
// Telnyx's 10DLC API answers its objects bare, without the "data" envelope the rest of v2
// uses. A brand is verified by The Campaign Registry before a campaign may be built on it,
// and a campaign is only usable once every carrier has provisioned it.
package telnyx

import (
	"bytes"
	"cmp"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	apiKeyEnvVar    = "TELNYX_API_KEY"
	publicKeyEnvVar = "TELNYX_PUBLIC_KEY"
)

const (
	defaultBaseURL = "https://api.telnyx.com"
	defaultTimeout = 30 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// hookTolerance is how old a signed report may be and still be acted on.
const hookTolerance = 5 * time.Minute

// Telnyx signs a report with Ed25519 over its timestamp and its body.
const (
	signatureHeader = "Telnyx-Signature-Ed25519"
	timestampHeader = "Telnyx-Timestamp"
)

// Options configures a Registrar. The keys fall back to the environment.
type Options struct {
	// APIKey defaults to TELNYX_API_KEY.
	APIKey string
	// PublicKey is the base64 Ed25519 key Telnyx signs with, defaulting to
	// TELNYX_PUBLIC_KEY. Without it no report is believed, and the poller does the work.
	PublicKey string
	// BaseURL defaults to Telnyx's API host.
	BaseURL    string
	HTTPClient *http.Client
}

// Registrar is Telnyx. It satisfies dlc.Registrar.
type Registrar struct {
	apiKey    string
	publicKey ed25519.PublicKey
	baseURL   string
	client    *http.Client
}

// New validates the options and returns a Registrar.
func New(options Options) (*Registrar, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("telnyx: " + apiKeyEnvVar + " is required")
	}
	if options.PublicKey == "" {
		options.PublicKey = os.Getenv(publicKeyEnvVar)
	}
	var publicKey ed25519.PublicKey
	if options.PublicKey != "" {
		key, err := base64.StdEncoding.DecodeString(options.PublicKey)
		if err != nil || len(key) != ed25519.PublicKeySize {
			return nil, errors.New("telnyx: " + publicKeyEnvVar + " is not a base64 Ed25519 public key")
		}
		publicKey = key
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: defaultTimeout}
	}
	return &Registrar{
		apiKey:    options.APIKey,
		publicKey: publicKey,
		baseURL:   strings.TrimSuffix(options.BaseURL, "/"),
		client:    options.HTTPClient,
	}, nil
}

// Name is the vendor numbers bought here are recorded under.
func (r *Registrar) Name() string { return "telnyx" }

// brand is the part of a Telnyx brand this reads.
type brand struct {
	BrandID        string `json:"brandId"`
	Status         string `json:"status"`
	IdentityStatus string `json:"identityStatus"`
}

// RegisterBrand registers a business profile as a brand.
func (r *Registrar) RegisterBrand(ctx context.Context, profile store.BusinessProfile) (dlc.Brand, error) {
	request := map[string]any{
		"entityType":        entityType(profile),
		"displayName":       cmp.Or(profile.BrandName, profile.LegalBusinessName),
		"companyName":       profile.LegalBusinessName,
		"firstName":         profile.ContactFirstName,
		"lastName":          profile.ContactLastName,
		"ein":               profile.TaxID,
		"einIssuingCountry": cmp.Or(profile.TaxIDCountry, profile.RegistrationCountry),
		"phone":             profile.ContactPhone,
		"email":             profile.ContactEmail,
		"street":            profile.Address.Street,
		"city":              profile.Address.City,
		"state":             profile.Address.State,
		"postalCode":        profile.Address.PostalCode,
		"country":           cmp.Or(profile.Address.Country, profile.RegistrationCountry),
		"website":           profile.WebsiteURL,
		"vertical":          strings.ToUpper(profile.Industry),
		"stockSymbol":       profile.StockSymbol,
		"stockExchange":     profile.StockExchange,
	}
	var answered brand
	raw, err := r.do(ctx, http.MethodPost, "/v2/10dlc/brand", request, &answered)
	if err != nil {
		return dlc.Brand{}, err
	}
	return brandOf(answered, raw), nil
}

// Brand reads where a registered brand stands.
func (r *Registrar) Brand(ctx context.Context, brandID string) (dlc.Brand, error) {
	var answered brand
	raw, err := r.do(ctx, http.MethodGet, "/v2/10dlc/brand/"+brandID, nil, &answered)
	if err != nil {
		return dlc.Brand{}, err
	}
	return brandOf(answered, raw), nil
}

// brandOf reads a brand's outcome. A brand is usable once its identity is verified, and
// failed when registration did or the registry could not verify who it is.
func brandOf(answered brand, raw []byte) dlc.Brand {
	outcome := dlc.Pending
	switch {
	case answered.Status == "REGISTRATION_FAILED" || answered.IdentityStatus == "UNVERIFIED":
		outcome = dlc.Failed
	case slices.Contains([]string{"VERIFIED", "VETTED_VERIFIED", "SELF_DECLARED"}, answered.IdentityStatus):
		outcome = dlc.Accepted
	}
	status := answered.IdentityStatus
	if status == "" {
		status = answered.Status
	}
	return dlc.Brand{ID: answered.BrandID, Status: status, Outcome: outcome, Raw: raw}
}

// campaign is the part of a Telnyx campaign this reads.
type campaign struct {
	CampaignID     string `json:"campaignId"`
	CampaignStatus string `json:"campaignStatus"`
}

// RegisterCampaign builds a campaign for a use case under a brand.
func (r *Registrar) RegisterCampaign(ctx context.Context, brandID string, useCase store.UseCase, hookURL string) (dlc.Campaign, error) {
	request := map[string]any{
		"brandId":          brandID,
		"usecase":          useCase.UseCaseType,
		"description":      useCase.Description,
		"messageFlow":      useCase.MessageFlow,
		"helpMessage":      useCase.HelpMessage,
		"optoutMessage":    useCase.OptOutMessage,
		"optinMessage":     useCase.OptInMessage,
		"embeddedLink":     useCase.EmbeddedLinks,
		"embeddedPhone":    useCase.EmbeddedPhone,
		"ageGated":         useCase.AgeGated,
		"directLending":    useCase.DirectLending,
		"subscriberOptin":  useCase.OptInMessage != "",
		"subscriberOptout": true,
		"subscriberHelp":   true,
		"optoutKeywords":   "STOP",
		"helpKeywords":     "HELP",
		"optinKeywords":    "START",
		"autoRenewal":      true,
	}
	if hookURL != "" {
		request["webhookURL"] = hookURL
	}
	for i, sample := range useCase.MessageSamples[:min(len(useCase.MessageSamples), 5)] {
		request["sample"+strconv.Itoa(i+1)] = sample
	}
	var answered campaign
	raw, err := r.do(ctx, http.MethodPost, "/v2/10dlc/campaignBuilder", request, &answered)
	if err != nil {
		return dlc.Campaign{}, err
	}
	return campaignOf(answered, raw), nil
}

// Campaign reads where a registered campaign stands.
func (r *Registrar) Campaign(ctx context.Context, campaignID string) (dlc.Campaign, error) {
	var answered campaign
	raw, err := r.do(ctx, http.MethodGet, "/v2/10dlc/campaign/"+campaignID, nil, &answered)
	if err != nil {
		return dlc.Campaign{}, err
	}
	return campaignOf(answered, raw), nil
}

// campaignOf reads a campaign's outcome: usable once carriers provisioned it, failed when
// any step on the way refused it or it lapsed.
func campaignOf(answered campaign, raw []byte) dlc.Campaign {
	outcome := dlc.Pending
	switch status := answered.CampaignStatus; {
	case status == "MNO_PROVISIONED":
		outcome = dlc.Accepted
	case strings.HasSuffix(status, "_FAILED") || slices.Contains([]string{"MNO_REJECTED", "TCR_SUSPENDED", "TCR_EXPIRED"}, status):
		outcome = dlc.Failed
	}
	return dlc.Campaign{ID: answered.CampaignID, Status: answered.CampaignStatus, Outcome: outcome, Raw: raw}
}

// AssignNumber makes a number send as a campaign.
func (r *Registrar) AssignNumber(ctx context.Context, campaignID, e164 string) error {
	_, err := r.do(ctx, http.MethodPost, "/v2/10dlc/phone_number_campaigns",
		map[string]string{"phoneNumber": e164, "campaignId": campaignID}, nil)
	return err
}

// VerifyHook checks Telnyx's Ed25519 signature over the timestamp and the body.
func (r *Registrar) VerifyHook(header http.Header, body []byte, now time.Time) error {
	if r.publicKey == nil {
		return fmt.Errorf("%w: %s is not set, so no report can be believed", dlc.ErrRefused, publicKeyEnvVar)
	}
	seconds, err := strconv.ParseInt(header.Get(timestampHeader), 10, 64)
	if err != nil {
		return fmt.Errorf("%w: the report carries no timestamp", dlc.ErrRefused)
	}
	if age := now.Sub(time.Unix(seconds, 0)); age > hookTolerance || age < -hookTolerance {
		return fmt.Errorf("%w: the report is too old to act on", dlc.ErrRefused)
	}
	signature, err := base64.StdEncoding.DecodeString(header.Get(signatureHeader))
	if err != nil {
		return fmt.Errorf("%w: the report is not signed", dlc.ErrRefused)
	}
	signed := append([]byte(strconv.FormatInt(seconds, 10)+"|"), body...)
	if !ed25519.Verify(r.publicKey, signed, signature) {
		return fmt.Errorf("%w: the report is not signed by Telnyx", dlc.ErrRefused)
	}
	return nil
}

// report is where in a 10DLC report the campaign or brand is named: Telnyx has sent both
// the bare shape and the "data.payload" envelope its other webhooks use.
type report struct {
	CampaignID string `json:"campaignId"`
	BrandID    string `json:"brandId"`
	Data       struct {
		Payload struct {
			CampaignID string `json:"campaignId"`
			BrandID    string `json:"brandId"`
		} `json:"payload"`
	} `json:"data"`
}

// HookSubject is the campaign, or failing that the brand, a report is about. What it says
// about either is not read: Service asks Telnyx, which is the answer that cannot be stale.
func (r *Registrar) HookSubject(body []byte) (string, string) {
	var read report
	if json.Unmarshal(body, &read) != nil {
		return "", ""
	}
	return cmp.Or(read.CampaignID, read.Data.Payload.CampaignID), cmp.Or(read.BrandID, read.Data.Payload.BrandID)
}

func (r *Registrar) do(ctx context.Context, method, path string, body, into any) ([]byte, error) {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("telnyx: encode %s: %w", path, err)
		}
		payload = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, r.baseURL+path, payload)
	if err != nil {
		return nil, fmt.Errorf("telnyx: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+r.apiKey)
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := r.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("telnyx: %s: %w", path, err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return nil, fmt.Errorf("telnyx: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("telnyx: read %s: %w", path, err)
	}
	if into == nil {
		return raw, nil
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return nil, fmt.Errorf("telnyx: decode %s: %w", path, err)
	}
	return raw, nil
}

// entityType is Telnyx's name for what kind of organization a profile is.
func entityType(profile store.BusinessProfile) string {
	if profile.LegalEntityType == "sole_proprietor" {
		return "SOLE_PROPRIETOR"
	}
	switch profile.OrganizationType {
	case "public":
		return "PUBLIC_PROFIT"
	case "nonprofit":
		return "NON_PROFIT"
	case "government":
		return "GOVERNMENT"
	}
	return "PRIVATE_PROFIT"
}
