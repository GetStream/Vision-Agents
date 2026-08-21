package stream

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/agents-core-go/acceleration"
)

// NumberSearch narrows what the vendor is asked to offer.
type NumberSearch struct {
	// Vendor is who to buy from, e.g. "twilio". Empty asks whichever vendor is ready.
	Vendor string
	// Country is an ISO 3166-1 alpha-2 code. Empty is "US".
	Country string
	// AreaCode narrows the search to one area.
	AreaCode string
	// Contains are digits the number must contain.
	Contains string
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

// PurchaseAnyNumber buys the first number a vendor offers that matches the search.
//
// It starts a monthly charge, so it is not something to call on every run: hold the number
// and attach it to whichever call needs answering.
func (p *Phone) PurchaseAnyNumber(ctx context.Context, search NumberSearch) (acceleration.PhoneNumber, error) {
	var bought acceleration.PhoneNumber

	vendor := search.Vendor
	if vendor == "" {
		ready, err := p.ReadyVendor(ctx)
		if err != nil {
			return bought, err
		}
		vendor = ready
	}
	country := search.Country
	if country == "" {
		country = "US"
	}

	one := 1
	params := &acceleration.SearchPhoneNumbersParams{Vendor: vendor, Country: country, Limit: &one}
	if search.AreaCode != "" {
		params.AreaCode = &search.AreaCode
	}
	if search.Contains != "" {
		params.Contains = &search.Contains
	}

	found, err := p.client.SearchPhoneNumbersWithResponse(ctx, params)
	if err != nil {
		return bought, fmt.Errorf("stream: searching for a number: %w", err)
	}
	offered, err := answer(found.JSON200, found.JSON400, found.JSON401, found.JSON404, found.Status())
	if err != nil {
		return bought, err
	}
	if len(*offered) == 0 {
		return bought, fmt.Errorf("stream: %s has no numbers in %s to sell", vendor, country)
	}

	request := acceleration.BuyNumberRequest{Vendor: vendor, E164: (*offered)[0].E164}
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

// Place rings somebody and bridges the answered leg into a Stream call.
//
// Stream's SIP is inbound only, so the vendor originates the call rather than Stream
// dialling out.
func (p *Phone) Place(ctx context.Context, from, to string, tags map[string]string) (acceleration.PlacedCall, error) {
	var placed acceleration.PlacedCall

	request := acceleration.PlaceCallRequest{From: from, To: to}
	if len(tags) > 0 {
		labels := tags
		request.Tags = &labels
	}

	dialled, err := p.client.PlacePhoneCallWithResponse(ctx, request)
	if err != nil {
		return placed, fmt.Errorf("stream: calling %s: %w", to, err)
	}
	result, err := answer(dialled.JSON202, dialled.JSON400, dialled.JSON401, dialled.JSON404, dialled.Status())
	if err != nil {
		return placed, err
	}
	return *result, nil
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
