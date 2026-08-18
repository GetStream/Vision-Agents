package api

import (
	"context"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noCampaigns is what the campaign paths say on a deployment that cannot run one. A
// campaign is a phone call, a conversation and a row, so it needs all three.
const (
	noCampaigns     = "campaigns are not available: this deployment has no database, telephony or sessions"
	unknownCampaign = "no such campaign"
)

// ListCampaigns returns the calling customer's campaigns, newest first.
func (s *Server) ListCampaigns(ctx context.Context, _ ListCampaignsRequestObject) (ListCampaignsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListCampaigns401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListCampaigns400JSONResponse{badRequest(noCampaigns)}, nil
	}

	stored, err := s.store.CustomerCampaigns(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]Campaign, 0, len(stored))
	for _, campaign := range stored {
		listed = append(listed, campaignOf(campaign))
	}
	return ListCampaigns200JSONResponse(listed), nil
}

// CreateCampaign defines a list of people to ring. It is created stopped.
func (s *Server) CreateCampaign(ctx context.Context, request CreateCampaignRequestObject) (CreateCampaignResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateCampaign401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateCampaign400JSONResponse{badRequest(noCampaigns)}, nil
	}
	if request.Body == nil {
		return CreateCampaign400JSONResponse{badRequest("a request body is required")}, nil
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return CreateCampaign400JSONResponse{badRequest("a campaign needs a name")}, nil
	}
	if request.Body.ConfigId == "" {
		return CreateCampaign400JSONResponse{badRequest("a campaign needs an agent config to make its calls with")}, nil
	}
	if request.Body.FromNumber == "" {
		return CreateCampaign400JSONResponse{badRequest("a campaign needs one of your numbers to call from")}, nil
	}

	// A campaign that names a config nobody has would fail one call at a time, at
	// whatever hour it was started.
	if _, err := s.store.AgentConfig(ctx, customerID, request.Body.ConfigId); err != nil {
		return CreateCampaign400JSONResponse{badRequest(unknownConfig)}, nil
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
		return CreateCampaign400JSONResponse{badRequest(err.Error())}, nil
	}
	if err := s.store.CreateCampaign(ctx, &campaign); err != nil {
		return CreateCampaign400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateCampaign201JSONResponse(campaignOf(campaign)), nil
}

// GetCampaign returns one campaign.
func (s *Server) GetCampaign(ctx context.Context, request GetCampaignRequestObject) (GetCampaignResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetCampaign401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetCampaign400JSONResponse{badRequest(noCampaigns)}, nil
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.Id)
	if err != nil {
		return GetCampaign404JSONResponse{NotFoundJSONResponse{Error: unknownCampaign}}, nil
	}
	return GetCampaign200JSONResponse(campaignOf(campaign)), nil
}

// ListCampaignContacts returns who a campaign is ringing and how far it has got.
func (s *Server) ListCampaignContacts(ctx context.Context, request ListCampaignContactsRequestObject) (ListCampaignContactsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListCampaignContacts401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListCampaignContacts400JSONResponse{badRequest(noCampaigns)}, nil
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.Id)
	if err != nil {
		return ListCampaignContacts404JSONResponse{NotFoundJSONResponse{Error: unknownCampaign}}, nil
	}

	stored, err := s.store.CampaignContacts(ctx, campaign.ID)
	if err != nil {
		return nil, err
	}
	return ListCampaignContacts200JSONResponse(contactsOf(stored)), nil
}

// AddCampaignContacts adds people to ring.
func (s *Server) AddCampaignContacts(ctx context.Context, request AddCampaignContactsRequestObject) (AddCampaignContactsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return AddCampaignContacts401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return AddCampaignContacts400JSONResponse{badRequest(noCampaigns)}, nil
	}
	if request.Body == nil || len(request.Body.Contacts) == 0 {
		return AddCampaignContacts400JSONResponse{badRequest("there is nobody to add")}, nil
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.Id)
	if err != nil {
		return AddCampaignContacts404JSONResponse{NotFoundJSONResponse{Error: unknownCampaign}}, nil
	}

	contacts := make([]store.Contact, 0, len(request.Body.Contacts))
	for _, wanted := range request.Body.Contacts {
		if strings.TrimSpace(wanted.ToNumber) == "" {
			return AddCampaignContacts400JSONResponse{badRequest("a contact needs a number to ring")}, nil
		}
		contacts = append(contacts, store.Contact{
			CampaignID:   campaign.ID,
			ToNumber:     strings.TrimSpace(wanted.ToNumber),
			Instructions: value(wanted.Instructions),
		})
	}

	if err := s.store.AddContacts(ctx, contacts); err != nil {
		return AddCampaignContacts400JSONResponse{badRequest(err.Error())}, nil
	}
	return AddCampaignContacts201JSONResponse(contactsOf(contacts)), nil
}

// StartCampaign starts ringing. It returns once the campaign is running rather than once
// it is over.
func (s *Server) StartCampaign(ctx context.Context, request StartCampaignRequestObject) (StartCampaignResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return StartCampaign401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil || s.campaigns == nil {
		return StartCampaign400JSONResponse{badRequest(noCampaigns)}, nil
	}

	if _, err := s.store.Campaign(ctx, customerID, request.Id); err != nil {
		return StartCampaign404JSONResponse{NotFoundJSONResponse{Error: unknownCampaign}}, nil
	}
	if err := s.campaigns.Start(ctx, customerID, request.Id); err != nil {
		return StartCampaign400JSONResponse{badRequest(err.Error())}, nil
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	return StartCampaign202JSONResponse(campaignOf(campaign)), nil
}

// PauseCampaign stops ringing anybody new.
func (s *Server) PauseCampaign(ctx context.Context, request PauseCampaignRequestObject) (PauseCampaignResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return PauseCampaign401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil || s.campaigns == nil {
		return PauseCampaign400JSONResponse{badRequest(noCampaigns)}, nil
	}

	if _, err := s.store.Campaign(ctx, customerID, request.Id); err != nil {
		return PauseCampaign404JSONResponse{NotFoundJSONResponse{Error: unknownCampaign}}, nil
	}
	if err := s.campaigns.Pause(ctx, customerID, request.Id); err != nil {
		return PauseCampaign400JSONResponse{badRequest(err.Error())}, nil
	}

	campaign, err := s.store.Campaign(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	return PauseCampaign200JSONResponse(campaignOf(campaign)), nil
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
