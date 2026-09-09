package stream

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// NumberSearch narrows what the vendors are asked to offer.
type NumberSearch struct {
	// Vendor is who to buy from, e.g. "twilio". Empty asks every vendor that has its
	// credentials and buys from whichever offers the cheapest match.
	Vendor string
	// Country is an ISO 3166-1 alpha-2 code. Empty is "US".
	Country string
	// AreaCode narrows the search to one area.
	AreaCode string
	// Contains are digits the number must contain, anywhere in it.
	Contains string
	// Prefix are digits the number must start with, e.g. "719".
	Prefix string
	// Locality is a city, region or rate centre, e.g. "Denver".
	Locality string
	// AdministrativeArea is a US state or Canadian province, e.g. "CO".
	AdministrativeArea string
	// Features every number must have, e.g. "hd_voice". Voice is always required.
	Features []string
	// Tags are cost labels carried onto the purchase.
	Tags map[string]string
}

// Phone is the telephony half of the router: numbers to hold, and calls to place on them.
type Phone struct {
	client *acceleration.ClientWithResponses
}

// NewPhone reaches the telephony paths on a router.
func NewPhone(client *acceleration.ClientWithResponses) *Phone {
	return &Phone{client: client}
}

// PurchaseAnyNumber buys the cheapest number on offer that matches the search.
//
// Naming no vendor asks all of them at once, and the offer says which vendor is selling it,
// so there is no need to pick one first.
//
// It starts a monthly charge, so it is not something to call on every run: hold the number
// and attach it to whichever call needs answering.
func (p *Phone) PurchaseAnyNumber(ctx context.Context, search NumberSearch) (acceleration.PhoneNumber, error) {
	var bought acceleration.PhoneNumber

	country := search.Country
	if country == "" {
		country = "US"
	}

	one := 1
	params := &acceleration.SearchPhoneNumbersParams{Country: country, Limit: &one}
	if search.Vendor != "" {
		params.Vendor = &search.Vendor
	}
	if search.AreaCode != "" {
		params.AreaCode = &search.AreaCode
	}
	if search.Contains != "" {
		params.Contains = &search.Contains
	}
	if search.Prefix != "" {
		params.Prefix = &search.Prefix
	}
	if search.Locality != "" {
		params.Locality = &search.Locality
	}
	if search.AdministrativeArea != "" {
		params.AdministrativeArea = &search.AdministrativeArea
	}
	if len(search.Features) > 0 {
		features := make([]acceleration.PhoneCapability, 0, len(search.Features))
		for _, feature := range search.Features {
			features = append(features, acceleration.PhoneCapability(feature))
		}
		params.Features = &features
	}

	found, err := p.client.SearchPhoneNumbersWithResponse(ctx, params)
	if err != nil {
		return bought, fmt.Errorf("stream: searching for a number: %w", err)
	}
	offered, err := answer(found.JSON200, found.JSON400, found.JSON401, found.JSON404, found.Status())
	if err != nil {
		return bought, err
	}
	if len(offered.Numbers) == 0 {
		return bought, fmt.Errorf("stream: no vendor has a number in %s to sell%s",
			country, becauseOf(offered.Skipped))
	}

	offer := offered.Numbers[0]
	request := acceleration.BuyNumberRequest{
		Vendor:  offer.Vendor,
		E164:    offer.E164,
		Country: &offer.Country,
	}
	if len(search.Tags) > 0 {
		tags := search.Tags
		request.Tags = &tags
	}

	purchased, err := p.client.BuyPhoneNumberWithResponse(ctx, request)
	if err != nil {
		return bought, fmt.Errorf("stream: buying %s: %w", request.E164, err)
	}
	number, err := answer(purchased.JSON201, purchased.JSON400, purchased.JSON401, purchased.JSON404, purchased.Status())
	if err != nil {
		return bought, err
	}
	return *number, nil
}

// Numbers are the numbers the calling customer holds.
func (p *Phone) Numbers(ctx context.Context) ([]acceleration.PhoneNumber, error) {
	listed, err := p.client.ListPhoneNumbersWithResponse(ctx, &acceleration.ListPhoneNumbersParams{})
	if err != nil {
		return nil, fmt.Errorf("stream: listing numbers: %w", err)
	}
	held, err := answer(listed.JSON200, listed.JSON400, listed.JSON401, nil, listed.Status())
	if err != nil {
		return nil, err
	}
	return *held, nil
}

// ReadyVendor is a vendor this deployment is implemented for and holds credentials for.
func (p *Phone) ReadyVendor(ctx context.Context) (string, error) {
	listed, err := p.client.ListPhoneVendorsWithResponse(ctx)
	if err != nil {
		return "", fmt.Errorf("stream: listing vendors: %w", err)
	}
	vendors, err := answer(listed.JSON200, nil, listed.JSON401, nil, listed.Status())
	if err != nil {
		return "", err
	}
	for _, vendor := range *vendors {
		if vendor.Ready {
			return vendor.Vendor, nil
		}
	}
	return "", errors.New("stream: no telephony vendor is configured with the credentials it needs")
}

// Attach points a number at a call, which is what turns a bought number into one that
// reaches an agent.
func (p *Phone) Attach(ctx context.Context, number, callID, callType string) (acceleration.AttachedNumber, error) {
	var attached acceleration.AttachedNumber

	request := acceleration.AttachNumberRequest{}
	if callID != "" {
		request.CallId = &callID
	}
	if callType != "" {
		request.CallType = &callType
	}

	pointed, err := p.client.AttachPhoneNumberWithResponse(ctx, number, request)
	if err != nil {
		return attached, fmt.Errorf("stream: attaching %s: %w", number, err)
	}
	result, err := answer(pointed.JSON200, pointed.JSON400, pointed.JSON401, pointed.JSON404, pointed.Status())
	if err != nil {
		return attached, err
	}
	return *result, nil
}

// OutboundCall is the terms a call is placed on.
type OutboundCall struct {
	// From is one of the customer's own numbers, which is what the person sees.
	From string
	// To is who to call.
	To string
	// CallID is the Stream call the answered leg joins, and so the one the agent has to
	// be in. Empty has one named after this call.
	CallID string
	// CallType is the Stream call type. Empty means "default".
	CallType string
	// RingTimeout is how long to ring before giving up. Zero leaves the vendor's default,
	// which is long enough to reach voicemail.
	RingTimeout time.Duration
	// InitialDigits are pressed once the person answers, for reaching an extension behind
	// a menu, e.g. "ww1234#".
	InitialDigits string
	// Headers are carried to the person's leg as custom SIP headers. Only some vendors
	// can express these, and one that cannot refuses the call.
	Headers map[string]string
	// Custom is put on the Stream call, where the agent in it can read it.
	Custom map[string]string
	// Tags are cost labels carried onto the call.
	Tags map[string]string
}

// Place rings somebody and bridges the answered leg into a Stream call.
//
// Stream's SIP is inbound only, so the vendor originates the call rather than Stream
// dialling out. What comes back names the Stream call the answered leg is routed into, and
// an agent that is not in it hears nothing when the person picks up.
func (p *Phone) Place(ctx context.Context, call OutboundCall) (acceleration.PlacedCall, error) {
	var placed acceleration.PlacedCall

	request := acceleration.PlaceCallRequest{From: call.From, To: call.To}
	if call.CallID != "" {
		request.CallId = &call.CallID
	}
	if call.CallType != "" {
		request.CallType = &call.CallType
	}
	if call.RingTimeout > 0 {
		seconds := int(call.RingTimeout.Seconds())
		request.RingTimeoutSeconds = &seconds
	}
	if call.InitialDigits != "" {
		request.InitialDigits = &call.InitialDigits
	}
	if len(call.Headers) > 0 {
		headers := call.Headers
		request.Headers = &headers
	}
	if len(call.Custom) > 0 {
		custom := call.Custom
		request.Custom = &custom
	}
	if len(call.Tags) > 0 {
		labels := call.Tags
		request.Tags = &labels
	}

	dialled, err := p.client.PlacePhoneCallWithResponse(ctx, request)
	if err != nil {
		return placed, fmt.Errorf("stream: calling %s: %w", call.To, err)
	}
	result, err := answer(dialled.JSON202, dialled.JSON400, dialled.JSON401, dialled.JSON404, dialled.Status())
	if err != nil {
		return placed, err
	}
	return *result, nil
}

// becauseOf renders the vendors that were not part of an answer, since a search that found
// nothing because nobody could be asked is a different problem from one with no inventory.
func becauseOf(skipped []acceleration.SkippedVendor) string {
	if len(skipped) == 0 {
		return ""
	}
	reasons := make([]string, 0, len(skipped))
	for _, each := range skipped {
		reasons = append(reasons, each.Vendor+" "+each.Reason)
	}
	return ": " + strings.Join(reasons, "; ")
}

// answer returns what the router sent, raising what it said went wrong instead.
func answer[T any](ok *T, bad, unauthorized, missing *acceleration.Error, status string) (*T, error) {
	if ok != nil {
		return ok, nil
	}
	for _, failure := range []*acceleration.Error{bad, unauthorized, missing} {
		if failure != nil {
			return nil, fmt.Errorf("stream: %s", failure.Error)
		}
	}
	return nil, fmt.Errorf("stream: the router answered %s", status)
}
