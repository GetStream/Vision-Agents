package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/danielgtaylor/huma/v2"
)

// The phone paths are served only when a deployment configured telephony. Without it they
// answer 400 with what is missing rather than 404, because the path exists and it is the
// deployment that is incomplete.

// listPhoneVendors reports every vendor and whether it can be used.
func (s *Server) listPhoneVendors(ctx context.Context, _ *listPhoneVendorsRequest) (*listPhoneVendorsResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if s.phone == nil {
		return &listPhoneVendorsResponse{Body: []PhoneVendor{}}, nil
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
	return &listPhoneVendorsResponse{Body: vendors}, nil
}

// searchPhoneNumbers asks what is for sale, at one vendor or at every usable one.
func (s *Server) searchPhoneNumbers(ctx context.Context, request *searchPhoneNumbersRequest) (*searchPhoneNumbersResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if s.phone == nil {
		return nil, errNoTelephony
	}

	// Voice is what an agent needs, so it is always required, on top of whatever else
	// was asked for.
	search := phone.Search{
		Country:      request.Country,
		Capabilities: []phone.Capability{phone.Voice},
	}
	if request.AreaCode.ptr() != nil {
		search.AreaCode = *request.AreaCode.ptr()
	}
	if request.Contains.ptr() != nil {
		search.Contains = *request.Contains.ptr()
	}
	if request.Prefix.ptr() != nil {
		search.Prefix = *request.Prefix.ptr()
	}
	if request.Locality.ptr() != nil {
		search.Locality = *request.Locality.ptr()
	}
	if request.AdministrativeArea.ptr() != nil {
		search.AdministrativeArea = *request.AdministrativeArea.ptr()
	}
	if request.NumberType.ptr() != nil {
		search.Type = phone.NumberType(*request.NumberType.ptr())
	}
	if request.Features.ptr() != nil {
		for _, feature := range *request.Features.ptr() {
			if capability := phone.Capability(feature); capability != phone.Voice {
				search.Capabilities = append(search.Capabilities, capability)
			}
		}
	}
	if request.Limit.ptr() != nil {
		search.Limit = *request.Limit.ptr()
	}

	offers, err := s.searchOffers(ctx, request.Vendor.ptr(), search)
	if errors.Is(err, phone.ErrNotImplemented) {
		return nil, notFound(err.Error())
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
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
	return &searchPhoneNumbersResponse{Body: result}, nil
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

// listPhoneNumbers returns what the calling customer holds.
func (s *Server) listPhoneNumbers(ctx context.Context, request *listPhoneNumbersRequest) (*listPhoneNumbersResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.phone == nil {
		return nil, errNoTelephony
	}

	includeReleased := request.IncludeReleased.ptr() != nil && *request.IncludeReleased.ptr()
	held, err := s.phone.Numbers(ctx, customerID, includeReleased)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	numbers := make([]PhoneNumber, 0, len(held))
	for _, number := range held {
		numbers = append(numbers, phoneNumber(number))
	}
	return &listPhoneNumbersResponse{Body: numbers}, nil
}

// buyPhoneNumber buys a number for the calling customer.
func (s *Server) buyPhoneNumber(ctx context.Context, request *buyPhoneNumberRequest) (*buyPhoneNumberResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if s.phone == nil {
		return nil, errNoTelephony
	}

	tags := phoneTags(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
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
		return nil, notFound(err.Error())
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &buyPhoneNumberResponse{Body: phoneNumber(bought)}, nil
}

// releasePhoneNumber gives a number back.
func (s *Server) releasePhoneNumber(ctx context.Context, request *releasePhoneNumberRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.phone == nil {
		return nil, errNoTelephony
	}

	err := s.phone.Release(ctx, customerID, request.E164)
	if err != nil && strings.Contains(err.Error(), "is not a number") {
		return nil, notFound(err.Error())
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return nil, nil
}

// attachPhoneNumber points a number at a Stream call.
func (s *Server) attachPhoneNumber(ctx context.Context, request *attachPhoneNumberRequest) (*attachPhoneNumberResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.phone == nil {
		return nil, errNoTelephony
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
		return nil, notFound(err.Error())
	}
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		return nil, err
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	return &attachPhoneNumberResponse{Body: AttachedNumber{TrunkId: attached.TrunkID,
		RouteId: attached.RouteID,
		SipUri:  attached.Bridge.URI}}, nil
}

// placePhoneCall dials out from one of the customer's numbers.
func (s *Server) placePhoneCall(ctx context.Context, request *placePhoneCallRequest) (*placePhoneCallResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if s.phone == nil {
		return nil, errNoTelephony
	}

	tags := phoneTags(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
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
			return nil, invalidRequest("a call cannot ring for less than no time")
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
	if errors.Is(err, dlc.ErrRefused) {
		return nil, forbidden(err.Error())
	}
	if err != nil && strings.Contains(err.Error(), "is not a number") {
		return nil, notFound(err.Error())
	}
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		return nil, err
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	return &placePhoneCallResponse{Body: PlacedCall{VendorCallId: placed.VendorCallID,
		Status:   placed.Status,
		Vendor:   &placed.Vendor,
		CallId:   &placed.CallID,
		CallType: &placed.CallType}}, nil
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
		writeError(w, notFound("telephony is not configured"))
		return
	}

	plan, err := s.phone.Answer(r.Context(), token)
	if err != nil {
		// The vendor is about to bridge a live call to nowhere, so this is worth a log
		// line even though there is nobody to return the detail to.
		s.logger.Error("could not answer a placed call", "error", err)
		writeError(w, notFound("that call is not waiting to be answered"))
		return
	}

	w.Header().Set("Content-Type", plan.ContentType)
	if _, err := w.Write(plan.Body); err != nil {
		s.logger.Error("could not serve a call plan", "error", err)
	}
}

// transferPhoneCall brings a human onto a call that is already happening.
func (s *Server) transferPhoneCall(ctx context.Context, request *transferPhoneCallRequest) (*transferPhoneCallResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if s.phone == nil {
		return nil, errNoTelephony
	}

	tags := phoneTags(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
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
	app, err := s.callApp(ctx, customerID, transfer.CallType, transfer.CallID)
	if err != nil {
		return nil, err
	}
	transfer.StreamApp = app

	placed, err := s.phone.Transfer(ctx, transfer)
	if err != nil && strings.Contains(err.Error(), "is not a number") {
		return nil, notFound(err.Error())
	}
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		return nil, err
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	return &transferPhoneCallResponse{Body: PlacedCall{VendorCallId: placed.VendorCallID,
		Status: placed.Status}}, nil
}

// pressPhoneDigits presses digits on a call placed from here.
func (s *Server) pressPhoneDigits(ctx context.Context, request *pressPhoneDigitsRequest) (*struct{}, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if s.phone == nil {
		return nil, errNoTelephony
	}

	err := s.phone.SendDigits(ctx, request.Body.Vendor, request.VendorCallId, request.Body.Digits)
	if errors.Is(err, phone.ErrNotImplemented) {
		return nil, notFound(err.Error())
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return nil, nil
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

var errNoTelephony = notConfigured("phone numbers are not available: no telephony configured")

// callApp is the Stream app a live call is in, which a human transferred into it has to be
// routed in too: the running session's, then the app its lines were made in, then the
// customer's own.
func (s *Server) callApp(ctx context.Context, customerID, callType, callID string) (int64, error) {
	if callType == "" {
		callType = defaultCallType
	}
	if s.sessions != nil {
		for _, running := range s.sessions.List(session.Owner{CustomerID: customerID, Kind: auth.KindServer}) {
			if spec := running.Spec(); spec.CallID == callID && spec.CallType == callType {
				return spec.StreamApp, nil
			}
		}
	}
	if s.store != nil {
		pin, found, err := s.store.CallPin(ctx, customerID, callType, callID)
		if err != nil || found {
			return pin, err
		}
	}
	if s.stream == nil {
		return 0, nil
	}
	return s.stream.Pin(ctx, customerID)
}

// registerPhone declares the operations served in phone.go.
func (s *Server) registerPhone(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listPhoneVendors",
		Method:      http.MethodGet,
		Path:        "/v1/phone/vendors",
		Summary:     "List the telephony vendors and whether they can be used",
		Description: "Every vendor this service knows about. A vendor that is declared but not implemented is " +
			"listed rather than hidden, so what is missing is visible before a number is bought.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The declared vendors"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden},
	}, s.listPhoneVendors)
	huma.Register(api, huma.Operation{
		OperationID: "searchPhoneNumbers",
		Method:      http.MethodGet,
		Path:        "/v1/phone/numbers/available",
		Summary:     "Search for numbers to buy, at one vendor or all of them",
		Description: "Naming a vendor searches only that one. Leaving it out asks every vendor that has its " +
			"credentials, at once, and merges what they offer cheapest first. Vendors do not agree " +
			"on how a search can be narrowed, so one whose API cannot express a filter is reported " +
			"in `skipped` rather than asked without it, which would answer a search for one place " +
			"with numbers from another.",
		// Declared rather than read off the input, so the default is documented without
		// being filled in: the handler tells a parameter left out from one sent.
		Parameters: []*huma.Param{
			{Name: "limit", In: "query", Schema: &huma.Schema{Type: huma.TypeInteger, Default: 10}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "What the vendors are offering, and which could not answer"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.searchPhoneNumbers)
	huma.Register(api, huma.Operation{
		OperationID: "listPhoneNumbers",
		Method:      http.MethodGet,
		Path:        "/v1/phone/numbers",
		Summary:     "The numbers the calling customer holds",
		// Declared rather than read off the input, so the default is documented without
		// being filled in: the handler tells a parameter left out from one sent.
		Parameters: []*huma.Param{
			{Name: "include_released", In: "query", Description: "Include numbers that have been given back. A released number keeps its row, because what it cost while it was held is still part of that month's bill.", Schema: &huma.Schema{Type: huma.TypeBoolean, Default: false}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's numbers, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listPhoneNumbers)
	huma.Register(api, huma.Operation{
		OperationID:   "buyPhoneNumber",
		Method:        http.MethodPost,
		Path:          "/v1/phone/numbers",
		Summary:       "Buy a number, which starts its monthly charge",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The number is bought and recorded"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.buyPhoneNumber)
	huma.Register(api, huma.Operation{
		OperationID:   "releasePhoneNumber",
		Method:        http.MethodDelete,
		Path:          "/v1/phone/numbers/{e164}",
		Summary:       "Give a number back, which stops its monthly charge",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The number was released"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.releasePhoneNumber)
	huma.Register(api, huma.Operation{
		OperationID: "attachPhoneNumber",
		Method:      http.MethodPost,
		Path:        "/v1/phone/numbers/{e164}/attach",
		Summary:     "Point a number at a Stream call",
		Description: "Creates the SIP inbound trunk and routing rule and tells the vendor to send calls " +
			"there. This is what turns a bought number into one that reaches an agent.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The number now reaches a call"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.attachPhoneNumber)
	huma.Register(api, huma.Operation{
		OperationID: "placePhoneCall",
		Method:      http.MethodPost,
		Path:        "/v1/phone/calls",
		Summary:     "Place an outbound call and bridge it into a Stream call",
		Description: "Stream's SIP is inbound only, so the vendor originates the call and connects it to a " +
			"trunk the agent is already on, rather than Stream dialling out.",
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The vendor is placing the call"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.placePhoneCall)
	huma.Register(api, huma.Operation{
		OperationID: "transferPhoneCall",
		Method:      http.MethodPost,
		Path:        "/v1/phone/calls/transfer",
		Summary:     "Bring a human onto a call that is already happening",
		Description: "Stream's SIP is inbound only, so a transfer is a second leg rather than a handover: the " +
			"vendor dials the human and the answered leg is routed into the same Stream call, after " +
			"which the agent can leave. The caller is never moved, so nothing is lost if nobody " +
			"answers.",
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The vendor is dialling the human"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.transferPhoneCall)
	huma.Register(api, huma.Operation{
		OperationID: "pressPhoneDigits",
		Method:      http.MethodPost,
		Path:        "/v1/phone/calls/{vendor_call_id}/digits",
		Summary:     "Press digits on a call placed from here",
		Description: "For getting past a menu on an outbound call. The call is named by the id its vendor " +
			"gave when it was dialled, so only calls placed from this service can be pressed at. Not " +
			"every vendor can do this without ending the call it is on, and one that cannot says so.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The digits were pressed"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.pressPhoneDigits)
}

type listPhoneVendorsRequest struct{}

type listPhoneVendorsResponse struct {
	Body []PhoneVendor `nullable:"false"`
}

type searchPhoneNumbersRequest struct {
	Vendor             optionalParam[string]            `query:"vendor" doc:"One vendor to search. Absent searches every usable vendor."`
	Country            string                           `query:"country" doc:"ISO 3166-1 alpha-2 country code." required:"true"`
	AreaCode           optionalParam[string]            `query:"area_code"`
	Contains           optionalParam[string]            `query:"contains" doc:"Digits the number must contain, anywhere in it."`
	Prefix             optionalParam[string]            "query:\"prefix\" doc:\"Digits the number must start with, matched after the country dial code. This differs from `contains` in where the digits have to fall.\""
	Locality           optionalParam[string]            `query:"locality" doc:"A city, region or rate centre."`
	AdministrativeArea optionalParam[string]            `query:"administrative_area" doc:"A US state or Canadian province."`
	NumberType         optionalParam[PhoneNumberType]   `query:"number_type"`
	Features           optionalParam[[]PhoneCapability] `query:"features,explode" doc:"Capabilities every number must have. Repeat the parameter to require several. A vendor that cannot filter on one still reports what its numbers carry, so these are checked on the results either way."`
	Limit              optionalParam[int]               `query:"limit"`
}

type searchPhoneNumbersResponse struct {
	Body NumberSearchResult
}

type listPhoneNumbersRequest struct {
	IncludeReleased optionalParam[bool] `query:"include_released" doc:"Include numbers that have been given back. A released number keeps its row, because what it cost while it was held is still part of that month's bill."`
}

type listPhoneNumbersResponse struct {
	Body []PhoneNumber `nullable:"false"`
}

type buyPhoneNumberRequest struct {
	Body *BuyNumberRequest `required:"true"`
}

type buyPhoneNumberResponse struct {
	Body PhoneNumber
}

type releasePhoneNumberRequest struct {
	E164 string `path:"e164" doc:"The number in +15551234567 form."`
}

type attachPhoneNumberRequest struct {
	E164 string `path:"e164"`
	Body *AttachNumberRequest
}

type attachPhoneNumberResponse struct {
	Body AttachedNumber
}

type placePhoneCallRequest struct {
	Body *PlaceCallRequest `required:"true"`
}

type placePhoneCallResponse struct {
	Body PlacedCall
}

type transferPhoneCallRequest struct {
	Body *TransferCallRequest `required:"true"`
}

type transferPhoneCallResponse struct {
	Body PlacedCall
}

type pressPhoneDigitsRequest struct {
	VendorCallId string              `path:"vendor_call_id"`
	Body         *PressDigitsRequest `required:"true"`
}

// AttachNumberRequest is the AttachNumberRequest schema.
type AttachNumberRequest struct {
	AllowedIps *[]string `json:"allowed_ips,omitempty" doc:"The vendor's signalling addresses, as IPs or CIDR blocks."`
	CallId     *string   `json:"call_id,omitempty" doc:"The call every caller joins. Omit to give each caller their own call, named after the number they rang."`
	CallType   *string   `json:"call_type,omitempty" doc:"The Stream call type. Omit for \"agent\"."`
}

// AttachedNumber is the AttachedNumber schema.
type AttachedNumber struct {
	RouteId string `json:"route_id"`
	SipUri  string `json:"sip_uri" doc:"Where the vendor sends calls, e.g. sip:trunk@sip.stream-io-api.com."`
	TrunkId string `json:"trunk_id"`
}

// AvailableNumber is the AvailableNumber schema.
type AvailableNumber struct {
	Capabilities      []PhoneCapability `json:"capabilities" nullable:"false"`
	Country           string            `json:"country"`
	E164              string            `json:"e164" example:"+15125551234"`
	Locality          *string           `json:"locality,omitempty"`
	MonthlyCostMicros *int64            `json:"monthly_cost_micros,omitempty" doc:"Millionths of a dollar per month, zero when the vendor does not quote one."`
	NumberType        *PhoneNumberType  `json:"number_type,omitempty"`
	Region            *string           `json:"region,omitempty"`
	Vendor            string            `json:"vendor" doc:"Who is offering it, which is also who to buy it from." example:"telnyx"`
}

// BuyNumberRequest is the BuyNumberRequest schema.
type BuyNumberRequest struct {
	Country *string            `json:"country,omitempty" doc:"The country the number was offered from, as the search reported it. Most vendors buy by number alone; the few that buy out of a country's inventory need this, and it cannot be guessed back out of the number." example:"US"`
	E164    string             `json:"e164" example:"+15125551234"`
	Tags    *map[string]string `json:"tags,omitempty" doc:"Cost labels carried onto the purchase's request row."`
	Vendor  string             `json:"vendor" example:"twilio"`
}

// NumberSearchResult is the NumberSearchResult schema.
type NumberSearchResult struct {
	Numbers []AvailableNumber `json:"numbers" doc:"What the vendors are offering, cheapest first." nullable:"false"`
	Skipped []SkippedVendor   `json:"skipped" doc:"Vendors that were not part of the answer. A search that reached two of eight vendors found what two vendors had, and deciding whether to buy needs to know which." nullable:"false"`
}

// PhoneCapability What a number can carry. The names are Telnyx's feature names, because they are the widest vocabulary any of these vendors offers.
type PhoneCapability string

// Defines values for PhoneCapability.
const (
	PhoneCapabilityEmergency        PhoneCapability = "emergency"
	PhoneCapabilityFax              PhoneCapability = "fax"
	PhoneCapabilityHdVoice          PhoneCapability = "hd_voice"
	PhoneCapabilityInternationalSms PhoneCapability = "international_sms"
	PhoneCapabilityLocalCalling     PhoneCapability = "local_calling"
	PhoneCapabilityMms              PhoneCapability = "mms"
	PhoneCapabilitySms              PhoneCapability = "sms"
	PhoneCapabilityVoice            PhoneCapability = "voice"
)

// Valid indicates whether the value is a known member of the PhoneCapability enum.
func (e PhoneCapability) Valid() bool {
	switch e {
	case PhoneCapabilityEmergency:
		return true
	case PhoneCapabilityFax:
		return true
	case PhoneCapabilityHdVoice:
		return true
	case PhoneCapabilityInternationalSms:
		return true
	case PhoneCapabilityLocalCalling:
		return true
	case PhoneCapabilityMms:
		return true
	case PhoneCapabilitySms:
		return true
	case PhoneCapabilityVoice:
		return true
	default:
		return false
	}
}

func (PhoneCapability) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "PhoneCapability", "What a number can carry. The names are Telnyx's feature names, because they are the widest vocabulary any of these vendors offers.", "voice", "sms", "mms", "fax", "emergency", "hd_voice", "international_sms", "local_calling")
}

// PhoneNumber is the PhoneNumber schema.
type PhoneNumber struct {
	Capabilities      []PhoneCapability  `json:"capabilities" nullable:"false"`
	Country           string             `json:"country"`
	E164              string             `json:"e164"`
	MonthlyCostMicros int64              `json:"monthly_cost_micros"`
	PurchasedAt       time.Time          `json:"purchased_at"`
	ReleasedAt        *time.Time         `json:"released_at,omitempty" nullable:"true"`
	StreamTrunkId     *string            `json:"stream_trunk_id,omitempty" doc:"The SIP trunk calls to this number arrive on. Absent until attached."`
	Tags              *map[string]string `json:"tags,omitempty" doc:"The customer's own cost labels."`
	Vendor            string             `json:"vendor"`
}

// PhoneNumberType What kind of number it is, which decides who pays for the call.
type PhoneNumberType string

// Defines values for PhoneNumberType.
const (
	Local    PhoneNumberType = "local"
	Mobile   PhoneNumberType = "mobile"
	TollFree PhoneNumberType = "toll_free"
)

// Valid indicates whether the value is a known member of the PhoneNumberType enum.
func (e PhoneNumberType) Valid() bool {
	switch e {
	case Local:
		return true
	case Mobile:
		return true
	case TollFree:
		return true
	default:
		return false
	}
}

func (PhoneNumberType) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "PhoneNumberType", "What kind of number it is, which decides who pays for the call.", "local", "toll_free", "mobile")
}

// PhoneOperation is the PhoneOperation schema.
type PhoneOperation string

// Defines values for PhoneOperation.
const (
	PhoneOperationAttach     PhoneOperation = "attach"
	PhoneOperationBuy        PhoneOperation = "buy"
	PhoneOperationDial       PhoneOperation = "dial"
	PhoneOperationRelease    PhoneOperation = "release"
	PhoneOperationSearch     PhoneOperation = "search"
	PhoneOperationSendDigits PhoneOperation = "send_digits"
)

// Valid indicates whether the value is a known member of the PhoneOperation enum.
func (e PhoneOperation) Valid() bool {
	switch e {
	case PhoneOperationAttach:
		return true
	case PhoneOperationBuy:
		return true
	case PhoneOperationDial:
		return true
	case PhoneOperationRelease:
		return true
	case PhoneOperationSearch:
		return true
	case PhoneOperationSendDigits:
		return true
	default:
		return false
	}
}

func (PhoneOperation) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "PhoneOperation", "", "search", "buy", "release", "attach", "dial", "send_digits")
	registry.Map()["PhoneOperation"].Extensions = map[string]any{"x-enum-varnames": []any{"PhoneOperationSearch", "PhoneOperationBuy", "PhoneOperationRelease", "PhoneOperationAttach", "PhoneOperationDial", "PhoneOperationSendDigits"}}
	return ref
}

// PhoneVendor is the PhoneVendor schema.
type PhoneVendor struct {
	Capabilities       []PhoneCapability `json:"capabilities" nullable:"false"`
	Implemented        bool              `json:"implemented" doc:"Whether this service can actually work with the vendor."`
	MissingCredentials *[]string         `json:"missing_credentials,omitempty" doc:"The environment variables the vendor needs and does not have."`
	Operations         *[]PhoneOperation `json:"operations,omitempty" doc:"What this service can do at the vendor. Eight vendors buy numbers and two of those also bridge calls, so a number is not bought from a vendor that cannot answer on it by accident."`
	Ready              bool              `json:"ready" doc:"Implemented and holding every credential it needs."`
	Vendor             string            `json:"vendor" example:"twilio"`
}

// PlaceCallRequest is the PlaceCallRequest schema.
type PlaceCallRequest struct {
	CallId             *string            `json:"call_id,omitempty" doc:"The Stream call the answered leg joins, and so the one the agent has to be in. Omit to have one named after this call, since two calls from the same number are two conversations."`
	CallType           *string            `json:"call_type,omitempty" doc:"The Stream call type. Omit for \"agent\"."`
	Custom             *map[string]string `json:"custom,omitempty" doc:"Put on the Stream call, where the agent in it can read it. It is set at Stream rather than at the vendor, so every vendor can carry it."`
	From               string             `json:"from" doc:"One of the customer's own numbers, which is what the person sees."`
	Headers            *map[string]string `json:"headers,omitempty" doc:"Carried to the person's leg as custom SIP headers. Only some vendors can express these, and one that cannot refuses the call."`
	InitialDigits      *string            `json:"initial_digits,omitempty" doc:"Digits pressed once the person answers, for reaching an extension behind a menu, e.g. \"ww1234#\". w is a short pause and W a long one."`
	RingTimeoutSeconds *int               `json:"ring_timeout_seconds,omitempty" doc:"How long to ring before giving up. Omit to leave the vendor's default, which is long enough to reach voicemail. A vendor whose call API cannot express it refuses the call rather than ringing for its own default."`
	Tags               *map[string]string `json:"tags,omitempty"`
	To                 string             `json:"to"`
}

func (*PlaceCallRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["ring_timeout_seconds"].Format = ""
	return schema
}

// PlacedCall is the PlacedCall schema.
type PlacedCall struct {
	CallId       *string `json:"call_id,omitempty" doc:"The Stream call the answered leg is routed into. An agent that is not in it hears nothing when the person picks up."`
	CallType     *string `json:"call_type,omitempty"`
	Status       string  `json:"status" doc:"The vendor's own word for where the call is, e.g. \"queued\"."`
	Vendor       *string `json:"vendor,omitempty" doc:"Who is placing the call."`
	VendorCallId string  `json:"vendor_call_id"`
}

// PressDigitsRequest is the PressDigitsRequest schema.
type PressDigitsRequest struct {
	Digits string `json:"digits" doc:"What to press. Only 0-9, * and # can be pressed, and w waits half a second between two of them."`
	Vendor string `json:"vendor" doc:"Who is carrying the call, e.g. \"telnyx\"."`
}

// SkippedVendor is the SkippedVendor schema.
type SkippedVendor struct {
	Reason string `json:"reason" example:"cannot search by administrative_area"`
	Vendor string `json:"vendor" example:"twilio"`
}

// TransferCallRequest is the TransferCallRequest schema.
type TransferCallRequest struct {
	CallId   string             `json:"call_id" doc:"The Stream call the caller and the agent are already on."`
	CallType *string            `json:"call_type,omitempty" doc:"The Stream call type. Omit for \"agent\"."`
	From     string             `json:"from" doc:"The customer's number the human is dialled from, which is what they see."`
	Tags     *map[string]string `json:"tags,omitempty"`
	To       string             `json:"to" doc:"The human being brought onto the call."`
}
