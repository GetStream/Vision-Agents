package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// noCampaigns is what the campaign paths say on a deployment that cannot run one. A
// campaign is a phone call, a conversation and a row, so it needs all three.
const (
	noCampaigns     = "campaigns are not available: this deployment has no database, telephony or sessions"
	unknownCampaign = "no such campaign"
)

type CampaignState string

const (
	CampaignStateDraft    CampaignState = "draft"
	CampaignStateFinished CampaignState = "finished"
	CampaignStatePaused   CampaignState = "paused"
	CampaignStateRunning  CampaignState = "running"
)

// Valid indicates whether the value is a known member of the CampaignState enum.
func (e CampaignState) Valid() bool {
	switch e {
	case CampaignStateDraft:
		return true
	case CampaignStateFinished:
		return true
	case CampaignStatePaused:
		return true
	case CampaignStateRunning:
		return true
	default:
		return false
	}
}

type Campaign struct {
	Id          string             `json:"id"`
	Name        string             `json:"name"`
	ConfigId    string             `json:"config_id"`
	FromNumber  string             `json:"from_number"`
	Concurrency int                `json:"concurrency"`
	State       CampaignState      `json:"state" enum:"draft,running,paused,finished"`
	Tags        *map[string]string `json:"tags,omitempty"`
	CreatedAt   time.Time          `json:"created_at"`
	StartedAt   *time.Time         `json:"started_at,omitempty"`
	FinishedAt  *time.Time         `json:"finished_at,omitempty"`
}

type CampaignRequest struct {
	Name        string             `json:"name"`
	ConfigId    string             `json:"config_id" doc:"The agent config the calls are made with."`
	FromNumber  string             `json:"from_number" doc:"One of your own numbers, which is what the person sees."`
	Concurrency *int               `json:"concurrency,omitempty" doc:"How many of these calls may be happening at once." default:"1"`
	Tags        *map[string]string `json:"tags,omitempty"`
}

type ContactState string

const (
	ContactStateCalling ContactState = "calling"
	ContactStateDone    ContactState = "done"
	ContactStateFailed  ContactState = "failed"
	ContactStatePending ContactState = "pending"
)

// Valid indicates whether the value is a known member of the ContactState enum.
func (e ContactState) Valid() bool {
	switch e {
	case ContactStateCalling:
		return true
	case ContactStateDone:
		return true
	case ContactStateFailed:
		return true
	case ContactStatePending:
		return true
	default:
		return false
	}
}

type Contact struct {
	Id           string       `json:"id"`
	ToNumber     string       `json:"to_number"`
	Instructions *string      `json:"instructions,omitempty"`
	State        ContactState `json:"state" enum:"pending,calling,done,failed"`
	Attempts     int          `json:"attempts"`
	CallId       *string      `json:"call_id,omitempty" doc:"The call this contact became, which is what the call paths take."`
	VendorCallId *string      `json:"vendor_call_id,omitempty"`
	Error        *string      `json:"error,omitempty" doc:"Why they could not be rung, when they could not be."`
}

type ContactsRequest struct {
	Contacts []struct {
		ToNumber     string  `json:"to_number"`
		Instructions *string `json:"instructions,omitempty" doc:"What to say to this person, added to whatever the config already says."`
	} `json:"contacts"`
}

// TransformSchema declares the contacts inline, where Huma names an anonymous struct Item.
func (*ContactsRequest) TransformSchema(registry huma.Registry, schema *huma.Schema) *huma.Schema {
	items := schema.Properties["contacts"].Items
	*items = *registry.SchemaFromRef(items.Ref)
	delete(registry.Map(), "Item")
	return schema
}

type campaignListResponse struct {
	Body []Campaign
}

type createCampaignRequest struct {
	Body CampaignRequest
}

type campaignResponse struct {
	Body Campaign
}

type getCampaignRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type listCampaignContactsRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type contactListResponse struct {
	Body []Contact
}

type addCampaignContactsRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body ContactsRequest
}

type startCampaignRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type pauseCampaignRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

func (s *Server) registerCampaigns(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listCampaigns",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns",
		Summary:     "The campaigns the calling customer has",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's campaigns, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listCampaigns)
	huma.Register(api, huma.Operation{
		OperationID: "createCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns",
		Summary:     "Define a list of people to ring",
		Description: "A campaign is created stopped. Add the people to ring, then start it: " +
			"concurrency is how many of these calls may be happening at once, and every call " +
			"is a conversation this service is running.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The campaign was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "getCampaign",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns/{id}",
		Summary:     "One campaign",
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "listCampaignContacts",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns/{id}/contacts",
		Summary:     "Who a campaign is ringing, and how far it has got",
		Responses: map[string]*huma.Response{
			"200": {Description: "The contacts, in the order they are rung"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listCampaignContacts)
	huma.Register(api, huma.Operation{
		OperationID: "addCampaignContacts",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/contacts",
		Summary:     "Add people to ring",
		Description: "Contacts are added rather than replaced, so a campaign can be topped up while " +
			"it is running. Each may carry instructions of their own, which the agent is " +
			"told along with whatever its config already says.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The contacts, in the order they are rung"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.addCampaignContacts)
	huma.Register(api, huma.Operation{
		OperationID: "startCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/start",
		Summary:     "Start ringing",
		Description: "Returns once the campaign is running rather than once it is over. A campaign " +
			"that was paused carries on from whoever is left.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The campaign is running"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.startCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "pauseCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/pause",
		Summary:     "Stop ringing anybody new",
		Description: "The calls already happening are left alone: a campaign is paused to stop " +
			"ringing people, not to hang up on the ones who answered.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign is paused"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.pauseCampaign)
}

// listCampaigns returns the calling customer's campaigns, newest first.
func (s *Server) listCampaigns(ctx context.Context, _ *struct{}) (*campaignListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCampaigns)
	}

	stored, err := s.store.CustomerCampaigns(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]Campaign, 0, len(stored))
	for _, campaign := range stored {
		listed = append(listed, campaignOf(campaign))
	}
	return &campaignListResponse{Body: listed}, nil
}

// createCampaign defines a list of people to ring. It is created stopped.
func (s *Server) createCampaign(ctx context.Context, request *createCampaignRequest) (*campaignResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCampaigns)
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return nil, huma.Error400BadRequest("a campaign needs a name")
	}
	if request.Body.ConfigId == "" {
		return nil, huma.Error400BadRequest("a campaign needs an agent config to make its calls with")
	}
	if request.Body.FromNumber == "" {
		return nil, huma.Error400BadRequest("a campaign needs one of your numbers to call from")
	}

	// A campaign that names a config nobody has would fail one call at a time, at
	// whatever hour it was started.
	if _, err := s.store.AgentConfig(ctx, customerID, request.Body.ConfigId); err != nil {
		return nil, huma.Error400BadRequest(unknownConfig)
	}

	campaign := store.Campaign{
		CustomerID:  customerID,
		Name:        strings.TrimSpace(request.Body.Name),
		ConfigID:    request.Body.ConfigId,
		FromNumber:  request.Body.FromNumber,
		Concurrency: value(request.Body.Concurrency),
	}
	if request.Body.Tags != nil {
		campaign.Tags = *request.Body.Tags
	}
	if err := routing.Tags(campaign.Tags).Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if err := s.store.CreateCampaign(ctx, &campaign); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &campaignResponse{Body: campaignOf(campaign)}, nil
}

// getCampaign returns one campaign.
func (s *Server) getCampaign(ctx context.Context, request *getCampaignRequest) (*campaignResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCampaigns)
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCampaign)
	}
	return &campaignResponse{Body: campaignOf(campaign)}, nil
}

// listCampaignContacts returns who a campaign is ringing and how far it has got.
func (s *Server) listCampaignContacts(ctx context.Context, request *listCampaignContactsRequest) (*contactListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCampaigns)
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCampaign)
	}

	stored, err := s.store.CampaignContacts(ctx, campaign.ID)
	if err != nil {
		return nil, err
	}
	return &contactListResponse{Body: contactsOf(stored)}, nil
}

// addCampaignContacts adds people to ring.
func (s *Server) addCampaignContacts(ctx context.Context, request *addCampaignContactsRequest) (*contactListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCampaigns)
	}
	if len(request.Body.Contacts) == 0 {
		return nil, huma.Error400BadRequest("there is nobody to add")
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCampaign)
	}

	contacts := make([]store.Contact, 0, len(request.Body.Contacts))
	for _, wanted := range request.Body.Contacts {
		if strings.TrimSpace(wanted.ToNumber) == "" {
			return nil, huma.Error400BadRequest("a contact needs a number to ring")
		}
		contacts = append(contacts, store.Contact{
			CampaignID:   campaign.ID,
			ToNumber:     strings.TrimSpace(wanted.ToNumber),
			Instructions: value(wanted.Instructions),
		})
	}

	if err := s.store.AddContacts(ctx, contacts); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &contactListResponse{Body: contactsOf(contacts)}, nil
}

// startCampaign starts ringing. It returns once the campaign is running rather than once
// it is over.
func (s *Server) startCampaign(ctx context.Context, request *startCampaignRequest) (*campaignResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil || s.campaigns == nil {
		return nil, huma.Error400BadRequest(noCampaigns)
	}

	if _, err := s.store.Campaign(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownCampaign)
	}
	if err := s.campaigns.Start(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.ID)
	if err != nil {
		return nil, err
	}
	return &campaignResponse{Body: campaignOf(campaign)}, nil
}

// pauseCampaign stops ringing anybody new.
func (s *Server) pauseCampaign(ctx context.Context, request *pauseCampaignRequest) (*campaignResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil || s.campaigns == nil {
		return nil, huma.Error400BadRequest(noCampaigns)
	}

	if _, err := s.store.Campaign(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownCampaign)
	}
	if err := s.campaigns.Pause(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.ID)
	if err != nil {
		return nil, err
	}
	return &campaignResponse{Body: campaignOf(campaign)}, nil
}

// campaignOf renders a campaign for the wire.
func campaignOf(campaign store.Campaign) Campaign {
	rendered := Campaign{
		Id:          campaign.ID,
		Name:        campaign.Name,
		ConfigId:    campaign.ConfigID,
		FromNumber:  campaign.FromNumber,
		Concurrency: campaign.Concurrency,
		State:       CampaignState(campaign.State),
		CreatedAt:   campaign.CreatedAt,
		StartedAt:   campaign.StartedAt,
		FinishedAt:  campaign.FinishedAt,
	}
	if len(campaign.Tags) > 0 {
		tags := campaign.Tags
		rendered.Tags = &tags
	}
	return rendered
}

// contactsOf renders contacts for the wire.
func contactsOf(contacts []store.Contact) []Contact {
	rendered := make([]Contact, 0, len(contacts))
	for _, contact := range contacts {
		rendered = append(rendered, Contact{
			Id:           contact.ID,
			ToNumber:     contact.ToNumber,
			Instructions: optional(contact.Instructions),
			State:        ContactState(contact.State),
			Attempts:     contact.Attempts,
			CallId:       optional(contact.CallID),
			VendorCallId: optional(contact.VendorCallID),
			Error:        optional(contact.Error),
		})
	}
	return rendered
}
