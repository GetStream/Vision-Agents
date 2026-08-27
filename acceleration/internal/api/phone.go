package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

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
		if operations := phoneOperations(vendor.Operations); len(operations) > 0 {
			listed.Operations = &operations
		}
		if missing := vendor.Missing(); len(missing) > 0 {
			listed.MissingCredentials = &missing
		}
		vendors = append(vendors, listed)
	}
	return ListPhoneVendors200JSONResponse(vendors), nil
}

// SearchPhoneNumbers asks what is for sale, at one vendor or at every usable one.
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

	// Voice is what an agent needs, so it is always required, on top of whatever else
	// was asked for.
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
	if request.Params.Prefix != nil {
		search.Prefix = *request.Params.Prefix
	}
	if request.Params.Locality != nil {
		search.Locality = *request.Params.Locality
	}
	if request.Params.AdministrativeArea != nil {
		search.AdministrativeArea = *request.Params.AdministrativeArea
	}
	if request.Params.NumberType != nil {
		search.Type = phone.NumberType(*request.Params.NumberType)
	}
	if request.Params.Features != nil {
		for _, feature := range *request.Params.Features {
			if capability := phone.Capability(feature); capability != phone.Voice {
				search.Capabilities = append(search.Capabilities, capability)
			}
		}
	}
	if request.Params.Limit != nil {
		search.Limit = *request.Params.Limit
	}

	offers, err := s.searchOffers(ctx, request.Params.Vendor, search)
	if errors.Is(err, phone.ErrNotImplemented) {
		return SearchPhoneNumbers404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return SearchPhoneNumbers400JSONResponse{badRequest(err.Error())}, nil
	}

	result := NumberSearchResult{
		Numbers: make([]AvailableNumber, 0, len(offers.Numbers)),
		Skipped: make([]SkippedVendor, 0, len(offers.Skipped)),
	}
	for _, number := range offers.Numbers {
		available := AvailableNumber{
			E164:         number.E164,
			Vendor:       number.Vendor,
			Country:      number.Country,
			Capabilities: phoneCapabilities(number.Capabilities),
		}
		if number.Region != "" {
			available.Region = &number.Region
		}
		if number.Locality != "" {
			available.Locality = &number.Locality
		}
		if number.Type != "" {
			kind := PhoneNumberType(number.Type)
			available.NumberType = &kind
		}
		if number.MonthlyCostMicros != 0 {
			cost := number.MonthlyCostMicros
			available.MonthlyCostMicros = &cost
		}
		result.Numbers = append(result.Numbers, available)
	}
	for _, skipped := range offers.Skipped {
		result.Skipped = append(result.Skipped, SkippedVendor{
			Vendor: skipped.Vendor,
			Reason: skipped.Reason,
		})
	}
	return SearchPhoneNumbers200JSONResponse(result), nil
}

// searchOffers asks one vendor when one is named and all of them when none is, so both
// answer in the same shape.
func (s *Server) searchOffers(
	ctx context.Context,
	vendor *string,
	search phone.Search,
) (phone.Offers, error) {
	if vendor == nil || *vendor == "" {
		return s.phone.SearchAll(ctx, search)
	}
	offered, err := s.phone.Search(ctx, *vendor, search)
	if err != nil {
		return phone.Offers{}, err
	}
	return phone.Offers{Numbers: offered}, nil
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

	purchase := phone.Purchase{
		Vendor: request.Body.Vendor,
		E164:   request.Body.E164,
		Owner:  routing.Owner{CustomerID: customerID, Tags: tags},
	}
	if request.Body.Country != nil {
		purchase.Country = *request.Body.Country
	}

	bought, err := s.phone.Buy(ctx, purchase)
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
	if request.Body.CallId != nil {
		call.CallID = *request.Body.CallId
	}
	if request.Body.CallType != nil {
		call.CallType = *request.Body.CallType
	}
	if request.Body.RingTimeoutSeconds != nil {
		if *request.Body.RingTimeoutSeconds < 0 {
			return PlacePhoneCall400JSONResponse{
				badRequest("a call cannot ring for less than no time"),
			}, nil
		}
		call.RingTimeout = time.Duration(*request.Body.RingTimeoutSeconds) * time.Second
	}
	if request.Body.InitialDigits != nil {
		call.InitialDigits = *request.Body.InitialDigits
	}
	if request.Body.Headers != nil {
		call.Headers = *request.Body.Headers
	}
	if request.Body.Custom != nil {
		call.Custom = *request.Body.Custom
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
		Vendor:       &placed.Vendor,
		CallId:       &placed.CallID,
		CallType:     &placed.CallType,
	}, nil
}

// answerPhoneCall serves the call plan a vendor fetches when the person it called picks up.
//
// This is the one path here a telephony vendor reaches rather than a customer, so it carries
// no customer header and is authenticated by the single-use token in its own path. It serves
// that vendor's XML rather than this API's JSON, which is why it is hand-written rather than
// generated. Vendors retry on a non-2xx and some of them use POST, so both verbs answer.
func (s *Server) answerPhoneCall(w http.ResponseWriter, r *http.Request) {
	token := r.PathValue("token")
	if s.phone == nil {
		http.Error(w, "telephony is not configured", http.StatusNotFound)
		return
	}

	plan, err := s.phone.Answer(r.Context(), token)
	if err != nil {
		// The vendor is about to bridge a live call to nowhere, so this is worth a log
		// line even though there is nobody to return the detail to.
		s.logger.Error("could not answer a placed call", "error", err)
		http.Error(w, "that call is not waiting to be answered", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", plan.ContentType)
	if _, err := w.Write(plan.Body); err != nil {
		s.logger.Error("could not serve a call plan", "error", err)
	}
}

// TransferPhoneCall brings a human onto a call that is already happening.
func (s *Server) TransferPhoneCall(
	ctx context.Context,
	request TransferPhoneCallRequestObject,
) (TransferPhoneCallResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return TransferPhoneCall401JSONResponse{missingCustomer()}, nil
	}
	if request.Body == nil {
		return TransferPhoneCall400JSONResponse{badRequest("a request body is required")}, nil
	}
	if s.phone == nil {
		return TransferPhoneCall400JSONResponse{noTelephony()}, nil
	}

	tags := phoneTags(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return TransferPhoneCall400JSONResponse{badRequest(err.Error())}, nil
	}

	transfer := phone.TransferRequest{
		Owner:  routing.Owner{CustomerID: customerID, Tags: tags},
		From:   request.Body.From,
		To:     request.Body.To,
		CallID: request.Body.CallId,
	}
	if request.Body.CallType != nil {
		transfer.CallType = *request.Body.CallType
	}

	placed, err := s.phone.Transfer(ctx, transfer)
	if err != nil && strings.Contains(err.Error(), "is not a number") {
		return TransferPhoneCall404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return TransferPhoneCall400JSONResponse{badRequest(err.Error())}, nil
	}

	return TransferPhoneCall202JSONResponse{
		VendorCallId: placed.VendorCallID,
		Status:       placed.Status,
	}, nil
}

// PressPhoneDigits presses digits on a call placed from here.
func (s *Server) PressPhoneDigits(
	ctx context.Context,
	request PressPhoneDigitsRequestObject,
) (PressPhoneDigitsResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return PressPhoneDigits401JSONResponse{missingCustomer()}, nil
	}
	if request.Body == nil {
		return PressPhoneDigits400JSONResponse{badRequest("a request body is required")}, nil
	}
	if s.phone == nil {
		return PressPhoneDigits400JSONResponse{noTelephony()}, nil
	}

	err := s.phone.SendDigits(ctx, request.Body.Vendor, request.VendorCallId, request.Body.Digits)
	if errors.Is(err, phone.ErrNotImplemented) {
		return PressPhoneDigits404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}
	if err != nil {
		return PressPhoneDigits400JSONResponse{badRequest(err.Error())}, nil
	}
	return PressPhoneDigits204Response{}, nil
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

func phoneOperations(operations []phone.Operation) []PhoneOperation {
	rendered := make([]PhoneOperation, 0, len(operations))
	for _, operation := range operations {
		rendered = append(rendered, PhoneOperation(operation))
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
