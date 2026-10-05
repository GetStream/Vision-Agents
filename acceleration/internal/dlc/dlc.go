// Package dlc is what an app has to prove before its numbers may text and call at volume:
// who it is, what it sends, and that people who said stop are not reached again.
//
// A use case is reviewed twice. Stream reads it first, so a vendor is only ever sent one
// that will pass, and the vendor's registry then approves it as a 10DLC campaign. Until one
// is approved a hosted app is sandboxed (see Gate).
package dlc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Where a use case stands.
const (
	Draft            = "draft"
	Submitted        = "submitted"
	ChangesRequested = "changes_requested"
	Rejected         = "rejected"
	VendorPending    = "vendor_pending"
	Approved         = "approved"
	VendorRejected   = "vendor_rejected"
)

// Statuses are every status, in the order a use case moves through them.
var Statuses = []string{Draft, Submitted, ChangesRequested, Rejected, VendorPending, Approved, VendorRejected}

// Who moved a use case.
const (
	ActorApp    = "app"
	ActorStaff  = "staff"
	ActorVendor = "vendor"
)

// What Stream's reviewer decided.
const (
	Approve        = "approve"
	Reject         = "reject"
	RequestChanges = "request_changes"
)

// The channels a recipient may opt out of, besides store.OptOutAll.
const (
	SMS      = "sms"
	Voice    = "voice"
	WhatsApp = "whatsapp"
	IMessage = "imessage"
)

// OptOutChannels are the channels an opt-out may name.
var OptOutChannels = []string{SMS, Voice, WhatsApp, IMessage, store.OptOutAll}

// ErrInvalid is a profile or use case that cannot be submitted or saved as it is.
var ErrInvalid = errors.New("dlc: invalid")

// ErrLocked is a use case whose status does not allow what was asked.
var ErrLocked = errors.New("dlc: the use case cannot do that in its status")

// ErrRefused is a text or call the gate would not let through.
var ErrRefused = errors.New("dlc: refused")

// Editable reports whether an app may still change a use case in status.
func Editable(status string) bool {
	return status == Draft || status == ChangesRequested || status == VendorRejected
}

// Deletable reports whether an app may delete a use case in status: not once the vendor
// holds it.
func Deletable(status string) bool {
	return status != VendorPending && status != Approved
}

// Registrar is a vendor that registers brands and 10DLC campaigns.
type Registrar interface {
	// Name is the vendor, as a number's vendor is named.
	Name() string
	RegisterBrand(ctx context.Context, profile store.BusinessProfile) (Brand, error)
	Brand(ctx context.Context, brandID string) (Brand, error)
	// RegisterCampaign registers a use case under a brand, reporting on hookURL.
	RegisterCampaign(ctx context.Context, brandID string, useCase store.UseCase, hookURL string) (Campaign, error)
	Campaign(ctx context.Context, campaignID string) (Campaign, error)
	// AssignNumber makes a number send as a campaign.
	AssignNumber(ctx context.Context, campaignID, e164 string) error
	// VerifyHook checks a delivery to the hook was signed by the vendor.
	VerifyHook(header http.Header, body []byte, now time.Time) error
	// HookSubject is the campaign, or failing that the brand, a delivery is about.
	HookSubject(body []byte) (campaignID, brandID string)
}

// Outcome is where a vendor's word for a brand or campaign leaves it.
type Outcome int

const (
	Pending Outcome = iota
	Accepted
	Failed
)

// Brand is a business as a vendor registered it.
type Brand struct {
	ID      string
	Status  string
	Outcome Outcome
	Raw     []byte
}

// Campaign is a use case as a vendor registered it.
type Campaign struct {
	ID      string
	Status  string
	Outcome Outcome
	Raw     []byte
}

// ValidateProfile reports what a profile is missing before a use case may be submitted on it.
func ValidateProfile(profile store.BusinessProfile) error {
	var missing []string
	for name, value := range map[string]string{
		"legal_business_name":           profile.LegalBusinessName,
		"legal_entity_type":             profile.LegalEntityType,
		"organization_type":             profile.OrganizationType,
		"business_registration_country": profile.RegistrationCountry,
		"website_url":                   profile.WebsiteURL,
		"industry":                      profile.Industry,
		"authorized_contact_email":      profile.ContactEmail,
		"authorized_contact_phone":      profile.ContactPhone,
		"registered_address.street":     profile.Address.Street,
		"registered_address.city":       profile.Address.City,
		"registered_address.country":    profile.Address.Country,
	} {
		if strings.TrimSpace(value) == "" {
			missing = append(missing, name)
		}
	}
	if profile.LegalEntityType != "sole_proprietor" && profile.TaxID == "" {
		missing = append(missing, "tax_id")
	}
	if profile.LegalEntityType == "sole_proprietor" && (profile.ContactFirstName == "" || profile.ContactLastName == "") {
		missing = append(missing, "authorized_contact_first_name and last_name")
	}
	if profile.OrganizationType == "public" && profile.StockSymbol == "" {
		missing = append(missing, "stock_symbol")
	}
	return missingFields("business profile", missing)
}

// ValidateUseCase reports what a use case is missing before it may be submitted.
func ValidateUseCase(useCase store.UseCase) error {
	var missing []string
	if useCase.UseCaseType == "" {
		missing = append(missing, "use_case_type")
	}
	if len(strings.TrimSpace(useCase.Description)) < 40 {
		missing = append(missing, "description of at least 40 characters")
	}
	if len(strings.TrimSpace(useCase.MessageFlow)) < 40 {
		missing = append(missing, "message_flow of at least 40 characters")
	}
	if len(useCase.MessageSamples) < 2 || slices.ContainsFunc(useCase.MessageSamples, func(sample string) bool {
		return len(strings.TrimSpace(sample)) < 20
	}) {
		missing = append(missing, "two or more message_samples of at least 20 characters")
	}
	if useCase.HelpMessage == "" {
		missing = append(missing, "help_message")
	}
	if useCase.OptOutMessage == "" {
		missing = append(missing, "opt_out_message")
	}
	return missingFields("use case", missing)
}

func missingFields(what string, missing []string) error {
	if len(missing) == 0 {
		return nil
	}
	slices.Sort(missing)
	return fmt.Errorf("%w: the %s needs %s", ErrInvalid, what, strings.Join(missing, ", "))
}
