package api

import (
	"context"
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The phone paths are served only when a deployment configured telephony. Without it they
// answer 400 with what is missing rather than 404, because the path exists and it is the
// deployment that is incomplete.

// ListPhoneVendors reports every vendor and whether it can be used.
func (s *Server) ListPhoneVendors(
	ctx context.Context,
	_ ListPhoneVendorsRequestObject,
) (ListPhoneVendorsResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return ListPhoneVendors401JSONResponse{missingCustomer()}, nil
	}
	if s.phone == nil {
		return ListPhoneVendors200JSONResponse{}, nil
	}

	registry := s.phone.Registry()
	ready := map[string]struct{}{}
	for _, name := range registry.Available() {
		ready[name] = struct{}{}
	}

	vendors := make([]PhoneVendor, 0, len(registry.Vendors()))
	for _, vendor := range registry.Vendors() {
		_, usable := ready[vendor.Vendor]
		listed := PhoneVendor{
			Vendor:       vendor.Vendor,
			Implemented:  vendor.Implemented,
			Ready:        usable,
			Capabilities: phoneCapabilities(vendor.Capabilities),
		}
		if missing := vendor.Missing(); len(missing) > 0 {
			listed.MissingCredentials = &missing
		}
		vendors = append(vendors, listed)
	}
	return ListPhoneVendors200JSONResponse(vendors), nil
}

// SearchPhoneNumbers asks a vendor what it has for sale.
func (s *Server) SearchPhoneNumbers(
	ctx context.Context,
	request SearchPhoneNumbersRequestObject,
) (SearchPhoneNumbersResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return SearchPhoneNumbers401JSONResponse{missingCustomer()}, nil
	}
	if s.phone == nil {
		return SearchPhoneNumbers400JSONResponse{noTelephony()}, nil
	}

	search := phone.Search{
		Country:      request.Params.Country,
		Capabilities: []phone.Capability{phone.Voice},
	}
	if request.Params.AreaCode != nil {
		search.AreaCode = *request.Params.AreaCode
	}
	if request.Params.Contains != nil {
		search.Contains = *request.Params.Contains
	}
	if request.Params.Limit != nil {
		search.Limit = *request.Params.Limit
	}

	offered, err := s.phone.Search(ctx, request.Params.Vendor, search)
	if errors.Is(err, phone.ErrNotImplemented) {
		return SearchPhoneNumbers404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return SearchPhoneNumbers400JSONResponse{badRequest(err.Error())}, nil
	}

	numbers := make([]AvailableNumber, 0, len(offered))
	for _, number := range offered {
		available := AvailableNumber{
			E164:         number.E164,
			Country:      number.Country,
			Capabilities: phoneCapabilities(number.Capabilities),
		}
		if number.Region != "" {
			available.Region = &number.Region
		}
		if number.Locality != "" {
			available.Locality = &number.Locality
		}
		if number.MonthlyCostMicros != 0 {
			cost := number.MonthlyCostMicros
			available.MonthlyCostMicros = &cost
		}
		numbers = append(numbers, available)
	}
	return SearchPhoneNumbers200JSONResponse(numbers), nil
}

// ListPhoneNumbers returns what the calling customer holds.
func (s *Server) ListPhoneNumbers(
	ctx context.Context,
	request ListPhoneNumbersRequestObject,
) (ListPhoneNumbersResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListPhoneNumbers401JSONResponse{missingCustomer()}, nil
	}
	if s.phone == nil {
		return ListPhoneNumbers400JSONResponse{noTelephony()}, nil
	}

	includeReleased := request.Params.IncludeReleased != nil && *request.Params.IncludeReleased
	held, err := s.phone.Numbers(ctx, customerID, includeReleased)
	if err != nil {
		return ListPhoneNumbers400JSONResponse{badRequest(err.Error())}, nil
	}

	numbers := make([]PhoneNumber, 0, len(held))
	for _, number := range held {
		numbers = append(numbers, phoneNumber(number))
	}
	return ListPhoneNumbers200JSONResponse(numbers), nil
}

// BuyPhoneNumber buys a number for the calling customer.
func (s *Server) BuyPhoneNumber(
	ctx context.Context,
	request BuyPhoneNumberRequestObject,
) (BuyPhoneNumberResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return BuyPhoneNumber401JSONResponse{missingCustomer()}, nil
	}
	if request.Body == nil {
		return BuyPhoneNumber400JSONResponse{badRequest("a request body is required")}, nil
	}
	if s.phone == nil {
		return BuyPhoneNumber400JSONResponse{noTelephony()}, nil
	}

	tags := phoneTags(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return BuyPhoneNumber400JSONResponse{badRequest(err.Error())}, nil
	}

	bought, err := s.phone.Buy(ctx, phone.Purchase{
		Vendor: request.Body.Vendor,
		E164:   request.Body.E164,
		Owner:  routing.Owner{CustomerID: customerID, Tags: tags},
	})
	if errors.Is(err, phone.ErrNotImplemented) {
		return BuyPhoneNumber404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return BuyPhoneNumber400JSONResponse{badRequest(err.Error())}, nil
	}
	return BuyPhoneNumber201JSONResponse(phoneNumber(bought)), nil
}

// ReleasePhoneNumber gives a number back.
func (s *Server) ReleasePhoneNumber(
	ctx context.Context,
	request ReleasePhoneNumberRequestObject,
) (ReleasePhoneNumberResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ReleasePhoneNumber401JSONResponse{missingCustomer()}, nil
	}
	if s.phone == nil {
		return ReleasePhoneNumber400JSONResponse{noTelephony()}, nil
	}

	err := s.phone.Release(ctx, customerID, request.E164)
	if err != nil && strings.Contains(err.Error(), "is not a number") {
		return ReleasePhoneNumber404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return ReleasePhoneNumber400JSONResponse{badRequest(err.Error())}, nil
	}
	return ReleasePhoneNumber204Response{}, nil
}

// AttachPhoneNumber points a number at a Stream call.
func (s *Server) AttachPhoneNumber(
	ctx context.Context,
	request AttachPhoneNumberRequestObject,
) (AttachPhoneNumberResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return AttachPhoneNumber401JSONResponse{missingCustomer()}, nil
	}
	if s.phone == nil {
		return AttachPhoneNumber400JSONResponse{noTelephony()}, nil
	}

	attachment := phone.Attachment{CustomerID: customerID, E164: request.E164}
	if request.Body != nil {
		if request.Body.CallId != nil {
			attachment.CallID = *request.Body.CallId
		}
		if request.Body.CallType != nil {
			attachment.CallType = *request.Body.CallType
		}
		if request.Body.AllowedIps != nil {
			attachment.AllowedIPs = *request.Body.AllowedIps
		}
	}

	attached, err := s.phone.Attach(ctx, attachment)
	if err != nil && strings.Contains(err.Error(), "is not a number") {
		return AttachPhoneNumber404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return AttachPhoneNumber400JSONResponse{badRequest(err.Error())}, nil
	}

	return AttachPhoneNumber200JSONResponse{
		TrunkId: attached.TrunkID,
		RouteId: attached.RouteID,
		SipUri:  attached.Bridge.URI,
	}, nil
}

// PlacePhoneCall dials out from one of the customer's numbers.
func (s *Server) PlacePhoneCall(
	ctx context.Context,
	request PlacePhoneCallRequestObject,
) (PlacePhoneCallResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return PlacePhoneCall401JSONResponse{missingCustomer()}, nil
	}
	if request.Body == nil {
		return PlacePhoneCall400JSONResponse{badRequest("a request body is required")}, nil
	}
	if s.phone == nil {
		return PlacePhoneCall400JSONResponse{noTelephony()}, nil
	}

	tags := phoneTags(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return PlacePhoneCall400JSONResponse{badRequest(err.Error())}, nil
	}

	call := phone.CallRequest{
		Owner: routing.Owner{CustomerID: customerID, Tags: tags},
		From:  request.Body.From,
		To:    request.Body.To,
	}
	if request.Body.SipUri != nil {
		call.Bridge = phone.Bridge{URI: *request.Body.SipUri}
	}

	placed, err := s.phone.Call(ctx, call)
	if err != nil && strings.Contains(err.Error(), "is not a number") {
		return PlacePhoneCall404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return PlacePhoneCall400JSONResponse{badRequest(err.Error())}, nil
	}

	return PlacePhoneCall202JSONResponse{
		VendorCallId: placed.VendorCallID,
		Status:       placed.Status,
	}, nil
}

func phoneNumber(held store.PhoneNumber) PhoneNumber {
	number := PhoneNumber{
		E164:              held.E164,
		Vendor:            held.Vendor,
		Country:           held.Country,
		Capabilities:      make([]PhoneCapability, 0, len(held.Capabilities)),
		MonthlyCostMicros: held.MonthlyCostMicros,
		PurchasedAt:       held.PurchasedAt,
		ReleasedAt:        held.ReleasedAt,
	}
	for _, capability := range held.Capabilities {
		number.Capabilities = append(number.Capabilities, PhoneCapability(capability))
	}
	if len(held.Tags) > 0 {
		tags := held.Tags
		number.Tags = &tags
	}
	if held.StreamTrunkID != "" {
		trunk := held.StreamTrunkID
		number.StreamTrunkId = &trunk
	}
	return number
}

func phoneCapabilities(capabilities []phone.Capability) []PhoneCapability {
	rendered := make([]PhoneCapability, 0, len(capabilities))
	for _, capability := range capabilities {
		rendered = append(rendered, PhoneCapability(capability))
	}
	return rendered
}

func phoneTags(tags *map[string]string) routing.Tags {
	if tags == nil {
		return nil
	}
	return routing.Tags(*tags)
}

func noTelephony() BadRequestJSONResponse {
	return badRequest("phone numbers are not available: no telephony configured")
}
