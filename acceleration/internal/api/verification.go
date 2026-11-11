package api

import (
	"cmp"
	"context"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const defaultVerificationPage = 50

var errNoVerification = notConfigured("verification requests need a database: this deployment has none")

// VerificationRequestInput is what an organization tells Stream about itself to be verified.
type VerificationRequestInput struct {
	LegalName     string `json:"legal_name" minLength:"1" maxLength:"200" doc:"The organization's registered legal name."`
	Country       string `json:"country" enum:"US,CA" doc:"Where the organization is registered: US or CA."`
	Website       string `json:"website" format:"uri" pattern:"^https?://\\S+$" maxLength:"300" doc:"The organization's website, an http or https URL."`
	TaxId         string `json:"tax_id" minLength:"1" maxLength:"50" doc:"The organization's tax id: an EIN in the US, a BN in Canada. Not checked further."`
	UseCase       string `json:"use_case" minLength:"1" maxLength:"2000" doc:"What the organization's agents will do on calls."`
	CallsPerMonth string `json:"calls_per_month" enum:"under_1k,1k_10k,10k_100k,over_100k" doc:"How many calls a month the organization expects."`
	Email         string `json:"email" format:"email" maxLength:"254" doc:"Where Stream answers."`
}

// VerificationRequest is an organization's request to be verified.
type VerificationRequest struct {
	VerificationRequestInput
	SubmittedAt time.Time `json:"submitted_at" readOnly:"true" doc:"When the router took the request."`
	SubmittedBy string    `json:"submitted_by" readOnly:"true" doc:"The user the request was made as, when the caller said, else empty."`
}

func (*VerificationRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An organization's request to be verified: who it is, what its agents will do and where " +
		"Stream answers. An organization has one, replaced whenever it sends again."
	return schema
}

// VerificationRequestEntry is one organization's request, with the tier Stream last knew it on.
type VerificationRequestEntry struct {
	OrganizationId string              `json:"organization_id"`
	Request        VerificationRequest `json:"request"`
	Tier           string              `json:"tier" doc:"The organization's last tier the router knows: verified or not-verified, empty when unknown."`
}

// VerificationRequestPage is one page of requests, newest first.
type VerificationRequestPage struct {
	Items      []VerificationRequestEntry `json:"items" nullable:"false"`
	HasMore    bool                       `json:"has_more"`
	NextCursor *string                    `json:"next_cursor,omitempty" doc:"Where the next page starts, for cursor; absent on the last page."`
}

// OrganizationVerificationRequest is what the organization sent, absent when it sent nothing.
type OrganizationVerificationRequest struct {
	Request *VerificationRequest `json:"request" doc:"The organization's request, null when it never sent one."`
}

// ManagedVerificationRequest is an organization's request with its last tier, for staff.
type ManagedVerificationRequest struct {
	Request *VerificationRequest `json:"request" doc:"The organization's request, null when it never sent one."`
	Tier    string               `json:"tier" doc:"The organization's last tier the router knows: verified or not-verified, empty when unknown."`
}

type putVerificationRequest struct{ Body VerificationRequestInput }

type organizationVerificationResponse struct {
	Body OrganizationVerificationRequest
}

type manageListVerificationRequestsRequest struct {
	Limit  int    `query:"limit" minimum:"1" maximum:"100" default:"50" doc:"The most requests one page holds."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page."`
}

type verificationRequestPageResponse struct{ Body VerificationRequestPage }

type manageGetVerificationRequestRequest struct {
	OrganizationId string `path:"organization_id"`
}

type managedVerificationResponse struct{ Body ManagedVerificationRequest }

func (s *Server) registerVerification(api huma.API) {
	serverSide := "\n\nServer-side only: a request that is not server-side answers 403 server_side_only."
	staffOnly := "\n\nStream staff only, for support: it needs the manage key."
	errs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound}
	huma.Register(api, huma.Operation{
		OperationID: "putVerification", Method: http.MethodPut, Path: "/v1/verification",
		Summary: "Send the organization's verification request",
		Description: "Sends the calling organization's request to be verified, replacing the one it sent before, " +
			"and tells Stream's staff. Whether the organization is verified is the tier getAccount reads." + serverSide,
		Errors: errs,
	}, s.putVerification)
	huma.Register(api, huma.Operation{
		OperationID: "getVerification", Method: http.MethodGet, Path: "/v1/verification",
		Summary:     "Read the organization's verification request",
		Description: "The request the calling organization sent, or null when it never sent one." + serverSide,
		Errors:      errs,
	}, s.getVerification)
	huma.Register(api, huma.Operation{
		OperationID: "manageListVerificationRequests", Method: http.MethodGet, Path: "/v1/manage/verification-requests",
		Summary: "List the organizations' verification requests",
		Description: "Every organization's request, newest first, each with the tier the router last knew the " +
			"organization on." + staffOnly,
		Tags: []string{staffTag}, Security: manageSecurity,
		Errors: errs,
	}, s.manageListVerificationRequests)
	huma.Register(api, huma.Operation{
		OperationID: "manageGetVerificationRequest", Method: http.MethodGet,
		Path:    "/v1/manage/organizations/{organization_id}/verification-request",
		Summary: "Read one organization's verification request",
		Description: "The request the organization sent, null when it never sent one, with the tier the router " +
			"last knew it on." + staffOnly,
		Tags: []string{staffTag}, Security: manageSecurity,
		Errors: errs,
	}, s.manageGetVerificationRequest)
}

func (s *Server) putVerification(ctx context.Context, request *putVerificationRequest) (*organizationVerificationResponse, error) {
	organization, err := s.accountCaller(ctx)
	if err != nil {
		return nil, err
	}
	if s.store == nil {
		return nil, errNoVerification
	}
	saved, err := s.store.SaveVerificationRequest(ctx, store.VerificationRequest{
		OrganizationID: organization,
		LegalName:      request.Body.LegalName,
		Country:        request.Body.Country,
		Website:        request.Body.Website,
		TaxID:          request.Body.TaxId,
		UseCase:        request.Body.UseCase,
		CallsPerMonth:  request.Body.CallsPerMonth,
		Email:          request.Body.Email,
		SubmittedBy:    ActorFrom(ctx).ID,
		SubmittedAt:    time.Now(),
	})
	if err != nil {
		return nil, err
	}
	sent := verificationRequestOf(saved)
	return &organizationVerificationResponse{Body: OrganizationVerificationRequest{Request: &sent}}, nil
}

func (s *Server) getVerification(ctx context.Context, _ *struct{}) (*organizationVerificationResponse, error) {
	organization, err := s.accountCaller(ctx)
	if err != nil {
		return nil, err
	}
	if s.store == nil {
		return nil, errNoVerification
	}
	entry, err := s.store.VerificationRequestOf(ctx, organization)
	if err != nil {
		return nil, err
	}
	answer := OrganizationVerificationRequest{}
	if entry != nil {
		sent := verificationRequestOf(entry.VerificationRequest)
		answer.Request = &sent
	}
	return &organizationVerificationResponse{Body: answer}, nil
}

func (s *Server) manageListVerificationRequests(ctx context.Context, request *manageListVerificationRequestsRequest) (*verificationRequestPageResponse, error) {
	if s.store == nil {
		return nil, errNoVerification
	}
	after, err := decodeCursor[store.VerificationPosition](&request.Cursor)
	if err != nil {
		return nil, errPageCursor
	}
	limit := cmp.Or(request.Limit, defaultVerificationPage)
	rows, err := s.store.ListVerificationRequests(ctx, limit, after)
	if err != nil {
		return nil, err
	}
	rows, more := page(rows, limit)
	listed := VerificationRequestPage{Items: make([]VerificationRequestEntry, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		listed.Items = append(listed.Items, VerificationRequestEntry{
			OrganizationId: row.OrganizationID, Request: verificationRequestOf(row.VerificationRequest), Tier: row.Tier,
		})
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.VerificationPosition{SubmittedAt: last.SubmittedAt, OrganizationID: last.OrganizationID})
	}
	return &verificationRequestPageResponse{Body: listed}, nil
}

func (s *Server) manageGetVerificationRequest(ctx context.Context, request *manageGetVerificationRequestRequest) (*managedVerificationResponse, error) {
	if s.store == nil {
		return nil, errNoVerification
	}
	entry, err := s.store.VerificationRequestOf(ctx, request.OrganizationId)
	if err != nil {
		return nil, err
	}
	answer := ManagedVerificationRequest{}
	if entry != nil {
		sent := verificationRequestOf(entry.VerificationRequest)
		answer.Request, answer.Tier = &sent, entry.Tier
	} else if answer.Tier, err = s.store.TierOf(ctx, request.OrganizationId); err != nil {
		return nil, err
	}
	return &managedVerificationResponse{Body: answer}, nil
}

func verificationRequestOf(row store.VerificationRequest) VerificationRequest {
	return VerificationRequest{
		VerificationRequestInput: VerificationRequestInput{
			LegalName: row.LegalName, Country: row.Country, Website: row.Website, TaxId: row.TaxID,
			UseCase: row.UseCase, CallsPerMonth: row.CallsPerMonth, Email: row.Email,
		},
		SubmittedAt: row.SubmittedAt, SubmittedBy: row.SubmittedBy,
	}
}

func (*OrganizationVerificationRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	nullableRequest(schema)
	return schema
}

func (*ManagedVerificationRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	nullableRequest(schema)
	return schema
}

// nullableRequest makes the request property say it may be null: Huma refuses a nullable
// object, and a client reading the spec would otherwise take the null for a broken answer.
func nullableRequest(schema *huma.Schema) {
	request := schema.Properties["request"]
	schema.Properties["request"] = &huma.Schema{
		Description: request.Description,
		OneOf:       []*huma.Schema{{Ref: request.Ref}, {Type: "null"}},
	}
}
