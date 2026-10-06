package api

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// errNoDLC is what the 10DLC paths say on a deployment with nowhere to keep a registration.
var errNoDLC = notConfigured("10DLC registration is not available: no database configured")

// opsKeyHeader carries the key Stream's own staff tools review use cases with.
const opsKeyHeader = "X-Ops-Key"

// opsSecurity marks an operation as Stream staff's, reached with the ops key and no
// customer at all.
var opsSecurity = []map[string][]string{{"OpsKey": {}}}

// maxReportBytes caps a registrar's report, which is a few hundred bytes of JSON.
const maxReportBytes = 64 << 10

// staffOperation reports whether an operation is reached with the ops key.
func staffOperation(operation *huma.Operation) bool {
	return slices.ContainsFunc(operation.Security, func(requirement map[string][]string) bool {
		_, ops := requirement["OpsKey"]
		return ops
	})
}

// requireOpsKey answers a staff operation called without this deployment's ops key with a
// 401, and one on a deployment with no ops key the same: nobody is staff there.
func (s *Server) requireOpsKey(api huma.API) func(huma.Context, func(huma.Context)) {
	return func(ctx huma.Context, next func(huma.Context)) {
		if !staffOperation(ctx.Operation()) {
			next(ctx)
			return
		}
		sent := ctx.Header(opsKeyHeader)
		if s.opsKey == "" || subtle.ConstantTimeCompare([]byte(sent), []byte(s.opsKey)) != 1 {
			writeOperationError(ctx, unauthenticated("this operation is Stream staff's: it needs "+opsKeyHeader))
			return
		}
		next(ctx)
	}
}

// dlcError answers a 10DLC failure with the status it deserves.
func dlcError(err error) error {
	switch {
	case errors.Is(err, dlc.ErrInvalid):
		return invalidRequest(err.Error())
	case errors.Is(err, dlc.ErrLocked), errors.Is(err, store.ErrUseCaseMoved):
		return conflict(err.Error())
	case errors.Is(err, store.ErrUnknownUseCase), errors.Is(err, store.ErrNoBusinessProfile),
		errors.Is(err, store.ErrUnknownOptOut):
		return notFound(err.Error())
	}
	return stack.Wrap(err)
}

// dlcCustomer is the customer a 10DLC operation acts for, or the error to answer with.
func (s *Server) dlcCustomer(ctx context.Context) (string, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return "", errMissingCustomer
	}
	if s.dlc == nil {
		return "", errNoDLC
	}
	return customerID, nil
}

func (s *Server) getBusinessProfile(ctx context.Context, _ *struct{}) (*businessProfileResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	profile, err := s.store.BusinessProfile(ctx, customerID)
	if err != nil {
		return nil, dlcError(err)
	}
	return &businessProfileResponse{Body: renderedProfile(profile)}, nil
}

func (s *Server) saveBusinessProfile(ctx context.Context, request *saveBusinessProfileRequest) (*businessProfileResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	profile := profileOf(customerID, *request.Body)
	if err := s.store.SaveBusinessProfile(ctx, &profile); err != nil {
		return nil, err
	}
	return &businessProfileResponse{Body: renderedProfile(profile)}, nil
}

func (s *Server) listUseCases(ctx context.Context, request *listUseCasesRequest) (*useCasePageResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	after, err := decodeCursor[store.CreatedPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.UseCases(ctx, customerID, request.Limit, after)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.DLCLimit(request.Limit))
	rendered := UseCasePage{HasMore: more, Items: make([]UseCase, 0, len(kept))}
	for _, useCase := range kept {
		item, err := s.renderedUseCase(ctx, useCase)
		if err != nil {
			return nil, err
		}
		rendered.Items = append(rendered.Items, item)
	}
	if more {
		last := kept[len(kept)-1]
		rendered.NextCursor = encodeCursor(store.CreatedPosition{At: last.CreatedAt, ID: last.ID})
	}
	return &useCasePageResponse{Body: rendered}, nil
}

func (s *Server) createUseCase(ctx context.Context, request *createUseCaseRequest) (*useCaseResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	useCase := store.UseCase{CustomerID: customerID}
	writeUseCase(&useCase, *request.Body)
	if err := s.dlc.Create(ctx, &useCase, request.Body.Numbers); err != nil {
		return nil, dlcError(err)
	}
	return s.useCaseResponse(ctx, useCase)
}

func (s *Server) getUseCase(ctx context.Context, request *useCaseIDRequest) (*useCaseResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	useCase, err := s.store.UseCase(ctx, customerID, request.Id)
	if err != nil {
		return nil, dlcError(err)
	}
	return s.useCaseResponse(ctx, useCase)
}

func (s *Server) updateUseCase(ctx context.Context, request *updateUseCaseRequest) (*useCaseResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	useCase, err := s.store.UseCase(ctx, customerID, request.Id)
	if err != nil {
		return nil, dlcError(err)
	}
	writeUseCase(&useCase, *request.Body)
	if err := s.dlc.Update(ctx, &useCase, request.Body.Numbers); err != nil {
		return nil, dlcError(err)
	}
	return s.useCaseResponse(ctx, useCase)
}

func (s *Server) deleteUseCase(ctx context.Context, request *useCaseIDRequest) (*struct{}, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if err := s.dlc.Delete(ctx, customerID, request.Id); err != nil {
		return nil, dlcError(err)
	}
	return nil, nil
}

func (s *Server) submitUseCase(ctx context.Context, request *useCaseIDRequest) (*useCaseResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	useCase, err := s.dlc.Submit(ctx, customerID, request.Id)
	if err != nil {
		return nil, dlcError(err)
	}
	return s.useCaseResponse(ctx, useCase)
}

func (s *Server) listUseCaseReviews(ctx context.Context, request *listUseCaseReviewsRequest) (*reviewPageResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if _, err := s.store.UseCase(ctx, customerID, request.Id); err != nil {
		return nil, dlcError(err)
	}
	after, err := decodeCursor[store.CreatedPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.ReviewLogs(ctx, customerID, request.Id, request.Limit, after)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.DLCLimit(request.Limit))
	rendered := ReviewPage{HasMore: more, Items: make([]UseCaseReview, 0, len(kept))}
	for _, log := range kept {
		rendered.Items = append(rendered.Items, renderedReview(log))
	}
	if more {
		last := kept[len(kept)-1]
		rendered.NextCursor = encodeCursor(store.CreatedPosition{At: last.CreatedAt, ID: last.ID})
	}
	return &reviewPageResponse{Body: rendered}, nil
}

func (s *Server) listOptOuts(ctx context.Context, request *listOptOutsRequest) (*optOutPageResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	after, err := decodeCursor[store.CreatedPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.OptOuts(ctx, customerID, request.Limit, after)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.DLCLimit(request.Limit))
	rendered := OptOutPage{HasMore: more, Items: make([]OptOut, 0, len(kept))}
	for _, optOut := range kept {
		rendered.Items = append(rendered.Items, renderedOptOut(optOut))
	}
	if more {
		last := kept[len(kept)-1]
		rendered.NextCursor = encodeCursor(store.CreatedPosition{At: last.CreatedAt, ID: last.ID})
	}
	return &optOutPageResponse{Body: rendered}, nil
}

func (s *Server) createOptOut(ctx context.Context, request *createOptOutRequest) (*optOutResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	optOut := store.OptOut{
		CustomerID: customerID,
		Recipient:  strings.TrimSpace(request.Body.Recipient),
		Channel:    string(request.Body.Channel),
		Source:     "api",
	}
	if request.Body.Source != nil {
		optOut.Source = *request.Body.Source
	}
	if err := s.store.OptOut(ctx, &optOut); err != nil {
		return nil, err
	}
	return &optOutResponse{Body: renderedOptOut(optOut)}, nil
}

func (s *Server) revokeOptOut(ctx context.Context, request *revokeOptOutRequest) (*struct{}, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if err := s.store.RevokeOptOut(ctx, customerID, request.Id); err != nil {
		return nil, dlcError(err)
	}
	return nil, nil
}

func (s *Server) getPhoneSandbox(ctx context.Context, _ *struct{}) (*phoneSandboxResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	usage, err := s.gate.Usage(ctx, customerID)
	if err != nil {
		return nil, err
	}
	return &phoneSandboxResponse{Body: PhoneSandbox{
		Enabled:            usage.Limits.Enabled,
		Sandboxed:          usage.Sandboxed,
		Recipients:         usage.Recipients,
		MaxRecipients:      usage.Limits.Recipients,
		MessagesToday:      usage.Messages,
		MessagesPerDay:     usage.Limits.MessagesPerDay,
		AudioSecondsToday:  usage.AudioSeconds,
		AudioMinutesPerDay: usage.Limits.AudioMinutesPerDay,
	}}, nil
}

func (s *Server) setSandboxRecipients(ctx context.Context, request *setSandboxRecipientsRequest) (*phoneSandboxResponse, error) {
	customerID, err := s.dlcCustomer(ctx)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if s.gate == nil {
		return nil, invalidRequest("this deployment has no sandbox")
	}
	recipients := make([]string, 0, len(request.Body.Recipients))
	for _, recipient := range request.Body.Recipients {
		recipients = append(recipients, strings.TrimSpace(recipient))
	}
	if err := s.gate.SetRecipients(ctx, customerID, recipients); err != nil {
		return nil, dlcError(err)
	}
	return s.getPhoneSandbox(ctx, nil)
}

func (s *Server) listUseCasesForReview(ctx context.Context, request *listUseCasesForReviewRequest) (*reviewQueueResponse, error) {
	if s.dlc == nil {
		return nil, errNoDLC
	}
	after, err := decodeCursor[store.CreatedPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	status := dlc.Submitted
	if request.Status != "" {
		status = string(request.Status)
	}
	found, err := s.store.UseCasesInStatus(ctx, status, request.Limit, after)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.DLCLimit(request.Limit))
	rendered := ReviewQueue{HasMore: more, Items: make([]UseCaseForReview, 0, len(kept))}
	for _, useCase := range kept {
		item, err := s.useCaseForReview(ctx, useCase, false)
		if err != nil {
			return nil, err
		}
		rendered.Items = append(rendered.Items, item)
	}
	if more {
		last := kept[len(kept)-1]
		rendered.NextCursor = encodeCursor(store.CreatedPosition{At: last.UpdatedAt, ID: last.ID})
	}
	return &reviewQueueResponse{Body: rendered}, nil
}

func (s *Server) getUseCaseForReview(ctx context.Context, request *useCaseIDRequest) (*useCaseForReviewResponse, error) {
	if s.dlc == nil {
		return nil, errNoDLC
	}
	useCase, err := s.store.UseCaseByID(ctx, request.Id)
	if err != nil {
		return nil, dlcError(err)
	}
	rendered, err := s.useCaseForReview(ctx, useCase, true)
	if err != nil {
		return nil, err
	}
	return &useCaseForReviewResponse{Body: rendered}, nil
}

func (s *Server) reviewUseCase(ctx context.Context, request *reviewUseCaseRequest) (*useCaseForReviewResponse, error) {
	if s.dlc == nil {
		return nil, errNoDLC
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	body := *request.Body
	if body.Decision != dlc.Approve && strings.TrimSpace(body.Notes) == "" {
		return nil, invalidRequest("say what is wrong in notes: the app reads them to fix it")
	}
	useCase, err := s.dlc.Review(ctx, request.Id, string(body.Decision), body.Notes, body.Reviewer)
	if err != nil {
		return nil, dlcError(err)
	}
	rendered, err := s.useCaseForReview(ctx, useCase, true)
	if err != nil {
		return nil, err
	}
	return &useCaseForReviewResponse{Body: rendered}, nil
}

// receiveDLCReport is the unauthenticated hook a registrar reports on brands and campaigns
// to. Each report is checked against the registrar's signing key, and then only says which
// use case to ask the registrar about.
func (s *Server) receiveDLCReport(w http.ResponseWriter, r *http.Request) {
	if s.dlc == nil {
		writeError(w, errNoDLC)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxReportBytes+1))
	if err != nil {
		writeError(w, invalidRequest(err.Error()))
		return
	}
	if len(body) > maxReportBytes {
		writeError(w, payloadTooLarge("a report is at most 64 KiB"))
		return
	}
	if err := s.dlc.Hook(r.Context(), r.Header, body); errors.Is(err, dlc.ErrRefused) {
		writeError(w, unauthenticated(err.Error()))
		return
	} else if err != nil {
		writeFailure(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) useCaseResponse(ctx context.Context, useCase store.UseCase) (*useCaseResponse, error) {
	rendered, err := s.renderedUseCase(ctx, useCase)
	if err != nil {
		return nil, err
	}
	return &useCaseResponse{Body: rendered}, nil
}

func (s *Server) renderedUseCase(ctx context.Context, useCase store.UseCase) (UseCase, error) {
	numbers, err := s.store.AssignedNumbers(ctx, useCase.CustomerID, useCase.ID)
	if err != nil {
		return UseCase{}, err
	}
	rendered := UseCase{
		Id: useCase.ID,
		UseCaseRequest: UseCaseRequest{
			Name:           useCase.Name,
			IsDefault:      useCase.IsDefault,
			UseCaseType:    useCase.UseCaseType,
			Description:    useCase.Description,
			MessageFlow:    useCase.MessageFlow,
			MessageSamples: useCase.MessageSamples,
			HelpMessage:    useCase.HelpMessage,
			OptOutMessage:  useCase.OptOutMessage,
			OptInMessage:   useCase.OptInMessage,
			EmbeddedLinks:  useCase.EmbeddedLinks,
			EmbeddedPhone:  useCase.EmbeddedPhone,
			AgeGated:       useCase.AgeGated,
			DirectLending:  useCase.DirectLending,
			Channels:       &useCase.Channels,
			Numbers:        numbers,
		},
		Status:           UseCaseStatus(useCase.Status),
		Vendor:           optional(useCase.Vendor),
		VendorCampaignId: optional(useCase.VendorCampaignID),
		VendorStatus:     optional(useCase.VendorStatus),
		CreatedAt:        useCase.CreatedAt,
		UpdatedAt:        useCase.UpdatedAt,
		SubmittedAt:      useCase.SubmittedAt,
		ApprovedAt:       useCase.ApprovedAt,
	}
	return rendered, nil
}

// useCaseForReview is a use case with what a reviewer reads it against: whose it is, the
// profile it was submitted on and, when asked, what happened to it so far.
func (s *Server) useCaseForReview(ctx context.Context, useCase store.UseCase, withReviews bool) (UseCaseForReview, error) {
	rendered, err := s.renderedUseCase(ctx, useCase)
	if err != nil {
		return UseCaseForReview{}, err
	}
	forReview := UseCaseForReview{UseCase: rendered, CustomerId: useCase.CustomerID}
	profile, err := s.store.BusinessProfile(ctx, useCase.CustomerID)
	if err == nil {
		shown := renderedProfile(profile)
		forReview.BusinessProfile = &shown
	} else if !errors.Is(err, store.ErrNoBusinessProfile) {
		return UseCaseForReview{}, err
	}
	if !withReviews {
		return forReview, nil
	}
	logs, err := s.store.ReviewLogs(ctx, useCase.CustomerID, useCase.ID, store.DLCLimit(maxReviewsShown), nil)
	if err != nil {
		return UseCaseForReview{}, err
	}
	forReview.Reviews = make([]UseCaseReview, 0, len(logs))
	for _, log := range logs[:min(len(logs), maxReviewsShown)] {
		forReview.Reviews = append(forReview.Reviews, renderedReview(log))
	}
	return forReview, nil
}

// maxReviewsShown is as much of a use case's history as a reviewer is shown with it.
const maxReviewsShown = 200

func writeUseCase(useCase *store.UseCase, body UseCaseRequest) {
	useCase.Name = strings.TrimSpace(body.Name)
	useCase.IsDefault = body.IsDefault
	useCase.UseCaseType = body.UseCaseType
	useCase.Description = body.Description
	useCase.MessageFlow = body.MessageFlow
	useCase.MessageSamples = body.MessageSamples
	useCase.HelpMessage = body.HelpMessage
	useCase.OptOutMessage = body.OptOutMessage
	useCase.OptInMessage = body.OptInMessage
	useCase.EmbeddedLinks = body.EmbeddedLinks
	useCase.EmbeddedPhone = body.EmbeddedPhone
	useCase.AgeGated = body.AgeGated
	useCase.DirectLending = body.DirectLending
	useCase.Channels = store.UseCaseChannels{}
	if body.Channels != nil {
		useCase.Channels = *body.Channels
	}
}

func profileOf(customerID string, body BusinessProfileRequest) store.BusinessProfile {
	profile := store.BusinessProfile{
		CustomerID:            customerID,
		LegalBusinessName:     body.LegalBusinessName,
		BrandName:             body.BrandName,
		LegalEntityType:       body.LegalEntityType,
		OrganizationType:      body.OrganizationType,
		RegistrationCountry:   body.BusinessRegistrationCountry,
		TaxID:                 body.TaxId,
		TaxIDCountry:          body.TaxIdIssuingCountry,
		WebsiteURL:            body.WebsiteUrl,
		Industry:              body.Industry,
		ContactFirstName:      body.AuthorizedContactFirstName,
		ContactLastName:       body.AuthorizedContactLastName,
		ContactTitle:          body.AuthorizedContactTitle,
		ContactEmail:          body.AuthorizedContactEmail,
		ContactPhone:          body.AuthorizedContactPhone,
		PrivacyPolicyURL:      body.PrivacyPolicyUrl,
		TermsURL:              body.TermsAndConditionsUrl,
		StockSymbol:           body.StockSymbol,
		StockExchange:         body.StockExchange,
		VerificationDocuments: body.BusinessVerificationDocuments,
	}
	if body.RegisteredAddress != nil {
		profile.Address = *body.RegisteredAddress
	}
	return profile
}

func renderedProfile(profile store.BusinessProfile) BusinessProfile {
	return BusinessProfile{
		BusinessProfileRequest: BusinessProfileRequest{
			LegalBusinessName:             profile.LegalBusinessName,
			BrandName:                     profile.BrandName,
			LegalEntityType:               profile.LegalEntityType,
			OrganizationType:              profile.OrganizationType,
			BusinessRegistrationCountry:   profile.RegistrationCountry,
			TaxId:                         profile.TaxID,
			TaxIdIssuingCountry:           profile.TaxIDCountry,
			RegisteredAddress:             &profile.Address,
			WebsiteUrl:                    profile.WebsiteURL,
			Industry:                      profile.Industry,
			AuthorizedContactFirstName:    profile.ContactFirstName,
			AuthorizedContactLastName:     profile.ContactLastName,
			AuthorizedContactTitle:        profile.ContactTitle,
			AuthorizedContactEmail:        profile.ContactEmail,
			AuthorizedContactPhone:        profile.ContactPhone,
			PrivacyPolicyUrl:              profile.PrivacyPolicyURL,
			TermsAndConditionsUrl:         profile.TermsURL,
			StockSymbol:                   profile.StockSymbol,
			StockExchange:                 profile.StockExchange,
			BusinessVerificationDocuments: profile.VerificationDocuments,
		},
		BrandId:     optional(profile.VendorBrandID),
		BrandStatus: optional(profile.BrandStatus),
		CreatedAt:   profile.CreatedAt,
		UpdatedAt:   profile.UpdatedAt,
	}
}

func renderedReview(log store.ReviewLog) UseCaseReview {
	review := UseCaseReview{
		Id:          log.ID,
		Actor:       ReviewActor(log.Actor),
		ActorName:   optional(log.ActorName),
		FromStatus:  UseCaseStatus(log.FromStatus),
		ToStatus:    UseCaseStatus(log.ToStatus),
		Notes:       optional(log.Notes),
		CreatedAt:   log.CreatedAt,
		SubmittedAt: log.SubmittedAt,
		ApprovedAt:  log.ApprovedAt,
	}
	if len(log.VendorPayload) > 0 {
		var payload map[string]any
		if json.Unmarshal(log.VendorPayload, &payload) == nil {
			review.VendorPayload = payload
		}
	}
	return review
}

func renderedOptOut(optOut store.OptOut) OptOut {
	return OptOut{
		Id:        optOut.ID,
		Recipient: optOut.Recipient,
		Channel:   OptOutChannel(optOut.Channel),
		Source:    optOut.Source,
		CreatedAt: optOut.CreatedAt,
	}
}

// BusinessProfileRequest is who an app is, as it tells it.
type BusinessProfileRequest struct {
	LegalBusinessName             string               `json:"legal_business_name,omitempty" maxLength:"255" doc:"The name the business is registered under."`
	BrandName                     string               `json:"brand_name,omitempty" maxLength:"255" doc:"What people know it as, shown to recipients. Omitted is the legal name."`
	LegalEntityType               string               `json:"legal_entity_type,omitempty" enum:"corporation,llc,partnership,sole_proprietor,other"`
	OrganizationType              string               `json:"organization_type,omitempty" enum:"private,public,nonprofit,government"`
	BusinessRegistrationCountry   string               `json:"business_registration_country,omitempty" maxLength:"2" doc:"ISO 3166-1 alpha-2."`
	TaxId                         string               `json:"tax_id,omitempty" maxLength:"64" doc:"The EIN in the US, or the country's business number. A sole proprietor has none."`
	TaxIdIssuingCountry           string               `json:"tax_id_issuing_country,omitempty" maxLength:"2" doc:"ISO 3166-1 alpha-2. Omitted is the registration country."`
	RegisteredAddress             *store.PostalAddress `json:"registered_address,omitempty"`
	WebsiteUrl                    string               `json:"website_url,omitempty" maxLength:"2048"`
	Industry                      string               `json:"industry,omitempty" maxLength:"64" doc:"The registry's vertical, such as technology, healthcare or retail."`
	AuthorizedContactFirstName    string               `json:"authorized_contact_first_name,omitempty" maxLength:"100"`
	AuthorizedContactLastName     string               `json:"authorized_contact_last_name,omitempty" maxLength:"100"`
	AuthorizedContactTitle        string               `json:"authorized_contact_title,omitempty" maxLength:"100"`
	AuthorizedContactEmail        string               `json:"authorized_contact_email,omitempty" maxLength:"255"`
	AuthorizedContactPhone        string               `json:"authorized_contact_phone,omitempty" maxLength:"20" doc:"E.164."`
	PrivacyPolicyUrl              string               `json:"privacy_policy_url,omitempty" maxLength:"2048"`
	TermsAndConditionsUrl         string               `json:"terms_and_conditions_url,omitempty" maxLength:"2048"`
	StockSymbol                   string               `json:"stock_symbol,omitempty" maxLength:"10" doc:"A public company's ticker."`
	StockExchange                 string               `json:"stock_exchange,omitempty" maxLength:"20" doc:"Where it is listed, such as NASDAQ or NYSE."`
	BusinessVerificationDocuments []string             `json:"business_verification_documents,omitempty" maxItems:"10" doc:"URLs of documents a reviewer may ask for, such as articles of incorporation."`
}

func (*BusinessProfileRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who the app is: written once and reused by every channel it registers " +
		"for. Nothing is required to save it; submitting a use case says what is missing."
	return schema
}

// BusinessProfile is an app's profile as it was saved.
type BusinessProfile struct {
	BusinessProfileRequest
	BrandId     *string   `json:"brand_id,omitempty" doc:"The brand the vendor registered the app as, once a use case was sent."`
	BrandStatus *string   `json:"brand_status,omitempty" doc:"The vendor's word for where the brand stands, such as VERIFIED."`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

func (*BusinessProfile) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who the app is, and the brand a vendor registered it as."
	return schema
}

type saveBusinessProfileRequest struct {
	Body *BusinessProfileRequest
}

type businessProfileResponse struct {
	Body BusinessProfile
}

// UseCaseStatus is where a use case stands.
type UseCaseStatus string

func (UseCaseStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "UseCaseStatus",
		"Where a use case stands. draft, changes_requested and vendor_rejected can be edited "+
			"and submitted; submitted waits on Stream's review; vendor_pending on the vendor's; "+
			"approved numbers may send. rejected is final.",
		dlc.Statuses...)
}

// ReviewActor is who moved a use case.
type ReviewActor string

func (ReviewActor) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ReviewActor", "Who moved a use case: the app, Stream staff or the vendor.",
		dlc.ActorApp, dlc.ActorStaff, dlc.ActorVendor)
}

// UseCaseRequest is what an app writes on a use case.
type UseCaseRequest struct {
	Name           string                 `json:"name" minLength:"1" maxLength:"120"`
	IsDefault      bool                   `json:"is_default,omitempty" doc:"Send as this use case from every number assigned to no other one. An app's first use case is its default."`
	UseCaseType    string                 `json:"use_case_type,omitempty" maxLength:"64" doc:"The campaign registry's use case, such as CUSTOMER_CARE, ACCOUNT_NOTIFICATION, 2FA, MARKETING or MIXED."`
	Description    string                 `json:"description,omitempty" maxLength:"4096" doc:"What the messages are for, in at least 40 characters."`
	MessageFlow    string                 `json:"message_flow,omitempty" maxLength:"4096" doc:"How a recipient opts in, and where a reviewer can see it, in at least 40 characters."`
	MessageSamples []string               `json:"message_samples,omitempty" maxItems:"5" doc:"Two to five messages as they will be sent."`
	HelpMessage    string                 `json:"help_message,omitempty" maxLength:"320" doc:"The answer to HELP: who you are and how to reach support."`
	OptOutMessage  string                 `json:"opt_out_message,omitempty" maxLength:"320" doc:"The answer to STOP."`
	OptInMessage   string                 `json:"opt_in_message,omitempty" maxLength:"320" doc:"The answer to START, and the first message after opting in."`
	EmbeddedLinks  bool                   `json:"embedded_links,omitempty"`
	EmbeddedPhone  bool                   `json:"embedded_phone,omitempty"`
	AgeGated       bool                   `json:"age_gated,omitempty"`
	DirectLending  bool                   `json:"direct_lending,omitempty"`
	Channels       *store.UseCaseChannels `json:"channels,omitempty" doc:"What RCS, WhatsApp, iMessage and voice ask for. Reviewed with the rest; not sent to a vendor yet."`
	Numbers        []string               `json:"numbers,omitempty" maxItems:"500" doc:"The app's numbers that send as this use case, in E.164. Omitted on an update leaves them as they are; empty unassigns them all."`
}

func (*UseCaseRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What an app sends texts and makes calls for. Saved as a draft and " +
		"submitted for review once it and the business profile are complete."
	return schema
}

// UseCase is a use case and where it stands.
type UseCase struct {
	Id string `json:"id"`
	UseCaseRequest
	Status           UseCaseStatus `json:"status"`
	Vendor           *string       `json:"vendor,omitempty" doc:"Who registers the campaign, once Stream approved it."`
	VendorCampaignId *string       `json:"vendor_campaign_id,omitempty"`
	VendorStatus     *string       `json:"vendor_status,omitempty" doc:"The vendor's word for where the campaign stands, such as TCR_ACCEPTED."`
	CreatedAt        time.Time     `json:"created_at"`
	UpdatedAt        time.Time     `json:"updated_at"`
	SubmittedAt      *time.Time    `json:"submitted_at,omitempty"`
	ApprovedAt       *time.Time    `json:"approved_at,omitempty"`
}

func (*UseCase) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A 10DLC use case: what an app sends, reviewed by Stream and then " +
		"registered as a campaign with the vendor its numbers come from."
	return schema
}

// UseCasePage is one page of an app's use cases.
type UseCasePage struct {
	HasMore    bool      `json:"has_more"`
	Items      []UseCase `json:"items" nullable:"false"`
	NextCursor *string   "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// UseCaseReview is one move a use case made.
type UseCaseReview struct {
	Id            string         `json:"id"`
	Actor         ReviewActor    `json:"actor"`
	ActorName     *string        `json:"actor_name,omitempty" doc:"Who exactly: the reviewer, or the vendor."`
	FromStatus    UseCaseStatus  `json:"from_status"`
	ToStatus      UseCaseStatus  `json:"to_status"`
	Notes         *string        `json:"notes,omitempty" doc:"What the reviewer or vendor said."`
	VendorPayload map[string]any `json:"vendor_payload,omitempty" doc:"What the vendor answered, as it came."`
	CreatedAt     time.Time      `json:"created_at"`
	SubmittedAt   *time.Time     `json:"submitted_at,omitempty"`
	ApprovedAt    *time.Time     `json:"approved_at,omitempty"`
}

func (*UseCaseReview) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One move a use case made, and who made it. Nothing here is ever changed."
	return schema
}

// ReviewPage is one page of a use case's history, oldest first.
type ReviewPage struct {
	HasMore    bool            `json:"has_more"`
	Items      []UseCaseReview `json:"items" nullable:"false"`
	NextCursor *string         "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// UseCaseForReview is a use case as Stream's reviewer reads it.
type UseCaseForReview struct {
	UseCase
	CustomerId      string           `json:"customer_id" doc:"The app the use case is for."`
	BusinessProfile *BusinessProfile `json:"business_profile,omitempty"`
	Reviews         []UseCaseReview  `json:"reviews,omitempty" doc:"What happened to it so far, oldest first. Only on a single use case."`
}

func (*UseCaseForReview) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A use case with the app it is for and the profile it was submitted on."
	return schema
}

// ReviewQueue is one page of use cases waiting on Stream.
type ReviewQueue struct {
	HasMore    bool               `json:"has_more"`
	Items      []UseCaseForReview `json:"items" nullable:"false"`
	NextCursor *string            "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// ReviewDecision is what Stream's reviewer decided.
type ReviewDecision string

func (ReviewDecision) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ReviewDecision",
		"approve sends the use case to the vendor; request_changes hands it back to the app to "+
			"edit; reject ends it.",
		dlc.Approve, dlc.RequestChanges, dlc.Reject)
}

// ReviewUseCaseRequest is a reviewer's decision.
type ReviewUseCaseRequest struct {
	Decision ReviewDecision `json:"decision"`
	Notes    string         `json:"notes,omitempty" maxLength:"4096" doc:"What the app should change, or why it was rejected. Required unless approving."`
	Reviewer string         `json:"reviewer,omitempty" maxLength:"255" doc:"Who decided, as the app's timeline shows it."`
}

// OptOutChannel is what a recipient opted out of.
type OptOutChannel string

func (OptOutChannel) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "OptOutChannel", "The channel a recipient opted out of, or all of them.",
		dlc.OptOutChannels...)
}

// OptOut is somebody who asked not to be reached.
type OptOut struct {
	Id        string        `json:"id"`
	Recipient string        `json:"recipient" doc:"The number, in E.164."`
	Channel   OptOutChannel `json:"channel"`
	Source    string        `json:"source" doc:"keyword when they texted STOP, otherwise api or dashboard."`
	CreatedAt time.Time     `json:"created_at"`
}

func (*OptOut) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Somebody who asked not to be reached. Nothing is texted or dialled to " +
		"them on the channel, or on any for all, until the opt-out is revoked or they text START."
	return schema
}

// OptOutPage is one page of an app's opt-outs, newest first.
type OptOutPage struct {
	HasMore    bool     `json:"has_more"`
	Items      []OptOut `json:"items" nullable:"false"`
	NextCursor *string  "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// CreateOptOutRequest records an opt-out the app heard about another way.
type CreateOptOutRequest struct {
	Recipient string        `json:"recipient" minLength:"2" maxLength:"20" doc:"The number, in E.164."`
	Channel   OptOutChannel `json:"channel"`
	Source    *string       `json:"source,omitempty" enum:"api,dashboard" doc:"Omitted is api."`
}

// PhoneSandbox is where an app stands against the sandbox today.
type PhoneSandbox struct {
	Enabled            bool     `json:"enabled" doc:"Whether this deployment sandboxes apps with no approved use case. Off on a self-hosted router."`
	Sandboxed          bool     `json:"sandboxed" doc:"Whether this app is held to the limits below: true until one of its use cases is approved."`
	Recipients         []string `json:"recipients" nullable:"false" doc:"The only numbers a sandboxed app may text and call."`
	MaxRecipients      int      `json:"max_recipients"`
	MessagesToday      int64    `json:"messages_today" doc:"Messages sent since midnight UTC."`
	MessagesPerDay     int64    `json:"messages_per_day"`
	AudioSecondsToday  int64    `json:"audio_seconds_today" doc:"Seconds of outbound calls since midnight UTC."`
	AudioMinutesPerDay int64    `json:"audio_minutes_per_day"`
}

func (*PhoneSandbox) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What an app may text and call before a 10DLC use case of its is approved."
	return schema
}

// SetSandboxRecipientsRequest replaces the numbers a sandboxed app may reach.
type SetSandboxRecipientsRequest struct {
	Recipients []string `json:"recipients" nullable:"false" maxItems:"10" doc:"Numbers in E.164, at most max_recipients of them."`
}

type listUseCasesRequest struct {
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type useCasePageResponse struct {
	Body UseCasePage
}

type createUseCaseRequest struct {
	Body *UseCaseRequest
}

type updateUseCaseRequest struct {
	Id   string `path:"id"`
	Body *UseCaseRequest
}

type useCaseIDRequest struct {
	Id string `path:"id"`
}

type useCaseResponse struct {
	Body UseCase
}

type listUseCaseReviewsRequest struct {
	Id     string `path:"id"`
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type reviewPageResponse struct {
	Body ReviewPage
}

type listOptOutsRequest struct {
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type optOutPageResponse struct {
	Body OptOutPage
}

type createOptOutRequest struct {
	Body *CreateOptOutRequest
}

type optOutResponse struct {
	Body OptOut
}

type revokeOptOutRequest struct {
	Id string `path:"id"`
}

type phoneSandboxResponse struct {
	Body PhoneSandbox
}

type setSandboxRecipientsRequest struct {
	Body *SetSandboxRecipientsRequest
}

type listUseCasesForReviewRequest struct {
	Status UseCaseStatus `query:"status" doc:"Omitted is submitted: what waits on Stream."`
	Limit  int           `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor string        `query:"cursor" doc:"The next_cursor of the previous page, sent with the same status."`
}

type reviewQueueResponse struct {
	Body ReviewQueue
}

type useCaseForReviewResponse struct {
	Body UseCaseForReview
}

type reviewUseCaseRequest struct {
	Id   string `path:"id"`
	Body *ReviewUseCaseRequest
}

// registerDLC declares the 10DLC operations: an app's profile, use cases, opt-outs and
// sandbox, which are server-side only, and Stream staff's review, which takes the ops key.
func (s *Server) registerDLC(api huma.API) {
	customerErrors := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden}
	withNotFound := append(slices.Clone(customerErrors), http.StatusNotFound)
	withConflict := append(slices.Clone(withNotFound), http.StatusConflict)

	huma.Register(api, huma.Operation{
		OperationID: "getBusinessProfile",
		Method:      http.MethodGet,
		Path:        "/v1/phone/business-profile",
		Summary:     "Get the business profile",
		Description: "Who the app said it is, and the brand a vendor registered it as.\n\nServer-side only.",
		Responses:   map[string]*huma.Response{"200": {Description: "The profile"}},
		Errors:      withNotFound,
	}, s.getBusinessProfile)
	huma.Register(api, huma.Operation{
		OperationID: "saveBusinessProfile",
		Method:      http.MethodPut,
		Path:        "/v1/phone/business-profile",
		Summary:     "Save the business profile",
		Description: "Replaces who the app says it is. Every use case is registered under it, so " +
			"submitting one checks it is complete.\n\nServer-side only.",
		Responses: map[string]*huma.Response{"200": {Description: "The profile as saved"}},
		Errors:    customerErrors,
	}, s.saveBusinessProfile)
	huma.Register(api, huma.Operation{
		OperationID: "listUseCases",
		Method:      http.MethodGet,
		Path:        "/v1/phone/use-cases",
		Summary:     "List 10DLC use cases",
		Description: "The app's use cases, newest first.\n\nServer-side only.",
		Responses:   map[string]*huma.Response{"200": {Description: "A page of use cases"}},
		Errors:      customerErrors,
	}, s.listUseCases)
	huma.Register(api, huma.Operation{
		OperationID: "createUseCase",
		Method:      http.MethodPost,
		Path:        "/v1/phone/use-cases",
		Summary:     "Create a 10DLC use case",
		Description: "Saves a draft use case. Nothing is checked beyond its shape until it is " +
			"submitted.\n\nServer-side only.",
		DefaultStatus: http.StatusCreated,
		Responses:     map[string]*huma.Response{"201": {Description: "The draft"}},
		Errors:        customerErrors,
	}, s.createUseCase)
	huma.Register(api, huma.Operation{
		OperationID: "getUseCase",
		Method:      http.MethodGet,
		Path:        "/v1/phone/use-cases/{id}",
		Summary:     "Get a 10DLC use case",
		Description: "Server-side only.",
		Responses:   map[string]*huma.Response{"200": {Description: "The use case"}},
		Errors:      withNotFound,
	}, s.getUseCase)
	huma.Register(api, huma.Operation{
		OperationID: "updateUseCase",
		Method:      http.MethodPut,
		Path:        "/v1/phone/use-cases/{id}",
		Summary:     "Update a 10DLC use case",
		Description: "Replaces what the app wrote. Only a draft, or one handed back by Stream or " +
			"the vendor, can be edited.\n\nServer-side only.",
		Responses: map[string]*huma.Response{"200": {Description: "The use case"}},
		Errors:    withConflict,
	}, s.updateUseCase)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteUseCase",
		Method:        http.MethodDelete,
		Path:          "/v1/phone/use-cases/{id}",
		Summary:       "Delete a 10DLC use case",
		Description:   "Refused once the vendor holds it.\n\nServer-side only.",
		DefaultStatus: http.StatusNoContent,
		Responses:     map[string]*huma.Response{"204": {Description: "Deleted"}},
		Errors:        withConflict,
	}, s.deleteUseCase)
	huma.Register(api, huma.Operation{
		OperationID: "submitUseCase",
		Method:      http.MethodPost,
		Path:        "/v1/phone/use-cases/{id}/submit",
		Summary:     "Submit a 10DLC use case for review",
		Description: "Sends a use case to Stream's review, once it and the business profile have " +
			"everything the registry asks for; a 400 says what is missing. Stream approving it " +
			"registers it with the vendor.\n\nServer-side only.",
		Responses: map[string]*huma.Response{"200": {Description: "The submitted use case"}},
		Errors:    withConflict,
	}, s.submitUseCase)
	huma.Register(api, huma.Operation{
		OperationID: "listUseCaseReviews",
		Method:      http.MethodGet,
		Path:        "/v1/phone/use-cases/{id}/reviews",
		Summary:     "List a use case's review history",
		Description: "Every move the use case made and who made it, oldest first.\n\nServer-side only.",
		Responses:   map[string]*huma.Response{"200": {Description: "A page of moves"}},
		Errors:      withNotFound,
	}, s.listUseCaseReviews)
	huma.Register(api, huma.Operation{
		OperationID: "listOptOuts",
		Method:      http.MethodGet,
		Path:        "/v1/phone/opt-outs",
		Summary:     "List opt-outs",
		Description: "The people who asked not to be reached, newest first.\n\nServer-side only.",
		Responses:   map[string]*huma.Response{"200": {Description: "A page of opt-outs"}},
		Errors:      customerErrors,
	}, s.listOptOuts)
	huma.Register(api, huma.Operation{
		OperationID: "createOptOut",
		Method:      http.MethodPost,
		Path:        "/v1/phone/opt-outs",
		Summary:     "Record an opt-out",
		Description: "Stops every text and call to a recipient on a channel, or on all of them. " +
			"Somebody texting STOP is recorded without this.\n\nServer-side only.",
		DefaultStatus: http.StatusCreated,
		Responses:     map[string]*huma.Response{"201": {Description: "The opt-out, or the one already recorded"}},
		Errors:        customerErrors,
	}, s.createOptOut)
	huma.Register(api, huma.Operation{
		OperationID:   "revokeOptOut",
		Method:        http.MethodDelete,
		Path:          "/v1/phone/opt-outs/{id}",
		Summary:       "Revoke an opt-out",
		Description:   "Lifts an opt-out. The record of it is kept.\n\nServer-side only.",
		DefaultStatus: http.StatusNoContent,
		Responses:     map[string]*huma.Response{"204": {Description: "Revoked"}},
		Errors:        withNotFound,
	}, s.revokeOptOut)
	huma.Register(api, huma.Operation{
		OperationID: "getPhoneSandbox",
		Method:      http.MethodGet,
		Path:        "/v1/phone/sandbox",
		Summary:     "Get the sandbox",
		Description: "What the app may text and call before a use case is approved, and how much " +
			"of today's allowance it used.\n\nServer-side only.",
		Responses: map[string]*huma.Response{"200": {Description: "The sandbox"}},
		Errors:    customerErrors,
	}, s.getPhoneSandbox)
	huma.Register(api, huma.Operation{
		OperationID: "setSandboxRecipients",
		Method:      http.MethodPut,
		Path:        "/v1/phone/sandbox/recipients",
		Summary:     "Set the sandbox recipients",
		Description: "Replaces the numbers a sandboxed app may text and call.\n\nServer-side only.",
		Responses:   map[string]*huma.Response{"200": {Description: "The sandbox"}},
		Errors:      customerErrors,
	}, s.setSandboxRecipients)

	staffErrors := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusConflict}
	huma.Register(api, huma.Operation{
		OperationID: "listUseCasesForReview",
		Method:      http.MethodGet,
		Path:        "/v1/ops/use-cases",
		Summary:     "List use cases waiting on Stream",
		Description: "Every app's use cases in one status, longest waiting first, with the profile " +
			"each was submitted on.\n\nStream staff only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "A page of use cases"}},
		Errors:    staffErrors,
	}, s.listUseCasesForReview)
	huma.Register(api, huma.Operation{
		OperationID: "getUseCaseForReview",
		Method:      http.MethodGet,
		Path:        "/v1/ops/use-cases/{id}",
		Summary:     "Get a use case to review",
		Description: "A use case with its app, profile and history.\n\nStream staff only: it needs the ops key.",
		Security:    opsSecurity,
		Responses:   map[string]*huma.Response{"200": {Description: "The use case"}},
		Errors:      staffErrors,
	}, s.getUseCaseForReview)
	huma.Register(api, huma.Operation{
		OperationID: "reviewUseCase",
		Method:      http.MethodPost,
		Path:        "/v1/ops/use-cases/{id}/review",
		Summary:     "Review a submitted use case",
		Description: "Approves, rejects or hands back a submitted use case. Approving registers the " +
			"brand and the campaign with the vendor, which approves it in turn.\n\nStream staff " +
			"only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The use case as reviewed"}},
		Errors:    staffErrors,
	}, s.reviewUseCase)
}
