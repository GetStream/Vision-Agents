package client

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// Campaigns are the app's outbound dialer: an agent config ringing a list of contacts from one
// of the app's numbers, within calling hours. Only a backend may ask.
type Campaigns struct {
	client *Client
}

// Campaigns is this app's outbound calling campaigns.
func (c *Client) Campaigns() *Campaigns { return &Campaigns{client: c} }

// CampaignListOptions narrows and pages Campaigns.List.
type CampaignListOptions struct {
	// State is the one state to list. Empty is every state.
	State acceleration.CampaignState
	// Limit is up to 200. Zero is 50.
	Limit int
	// Cursor is the NextCursor of the page before, with the same state. Empty is the first page.
	Cursor string
}

// ContactListOptions narrows and pages Campaigns.ListContacts.
type ContactListOptions struct {
	// Stage is the one stage to list. Empty is every stage.
	Stage acceleration.ContactStage
	// Number keeps the contacts whose number contains it.
	Number string
	// Limit is up to 200. Zero is 50.
	Limit int
	// Cursor is the NextCursor of the page before, with the same filters. Empty is the first page.
	Cursor string
}

// LaunchOptions is what launching a campaign asks.
type LaunchOptions struct {
	// Consent must be true: the people on the campaign agreed to be called by it. The router
	// records it with the time.
	Consent bool
	// StartAt is when to start ringing. Zero is now.
	StartAt time.Time
}

// List returns a page of the app's campaigns.
func (c *Campaigns) List(ctx context.Context, options CampaignListOptions) (*acceleration.CampaignPage, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	listed, err := api.ListCampaignsWithResponse(ctx, &acceleration.ListCampaignsParams{
		State:  pointer(options.State),
		Limit:  pointer(int64(options.Limit)),
		Cursor: pointer(options.Cursor),
	})
	if err != nil {
		return nil, fmt.Errorf("client: listing the campaigns: %w", err)
	}
	return answered("listing the campaigns", listed.JSON200, listed.HTTPResponse, listed.Body)
}

// Create stores a draft campaign. It rings nobody until it is launched.
func (c *Campaigns) Create(ctx context.Context, campaign acceleration.CampaignRequest) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	created, err := api.CreateCampaignWithResponse(ctx, campaign)
	if err != nil {
		return nil, fmt.Errorf("client: creating a campaign: %w", err)
	}
	return answered("creating a campaign", created.JSON201, created.HTTPResponse, created.Body)
}

// Get returns one campaign.
func (c *Campaigns) Get(ctx context.Context, id string) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	got, err := api.GetCampaignWithResponse(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("client: getting the campaign %s: %w", id, err)
	}
	return answered("getting the campaign "+id, got.JSON200, got.HTTPResponse, got.Body)
}

// Update changes the fields set in campaign and keeps the rest.
func (c *Campaigns) Update(ctx context.Context, id string, campaign acceleration.CampaignRequest) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	updated, err := api.UpdateCampaignWithResponse(ctx, id, campaign)
	if err != nil {
		return nil, fmt.Errorf("client: updating the campaign %s: %w", id, err)
	}
	return answered("updating the campaign "+id, updated.JSON200, updated.HTTPResponse, updated.Body)
}

// Delete removes a draft, stopped or completed campaign. Its calls stay.
func (c *Campaigns) Delete(ctx context.Context, id string) error {
	api, err := c.client.api()
	if err != nil {
		return err
	}
	deleted, err := api.DeleteCampaignWithResponse(ctx, id)
	if err != nil {
		return fmt.Errorf("client: deleting the campaign %s: %w", id, err)
	}
	if deleted.StatusCode() != http.StatusNoContent {
		return failure("deleting the campaign "+id, deleted.HTTPResponse, deleted.Body)
	}
	return nil
}

// Launch starts ringing a draft campaign's contacts, now or at options.StartAt.
func (c *Campaigns) Launch(ctx context.Context, id string, options LaunchOptions) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	request := acceleration.LaunchCampaignRequest{Consent: options.Consent}
	if !options.StartAt.IsZero() {
		request.StartAt = &options.StartAt
	}
	launched, err := api.LaunchCampaignWithResponse(ctx, id, request)
	if err != nil {
		return nil, fmt.Errorf("client: launching the campaign %s: %w", id, err)
	}
	return answered("launching the campaign "+id, launched.JSON200, launched.HTTPResponse, launched.Body)
}

// Pause stops placing new calls. The calls already ringing finish.
func (c *Campaigns) Pause(ctx context.Context, id string) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	paused, err := api.PauseCampaignWithResponse(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("client: pausing the campaign %s: %w", id, err)
	}
	return answered("pausing the campaign "+id, paused.JSON200, paused.HTTPResponse, paused.Body)
}

// Resume carries on a paused campaign.
func (c *Campaigns) Resume(ctx context.Context, id string) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	resumed, err := api.ResumeCampaignWithResponse(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("client: resuming the campaign %s: %w", id, err)
	}
	return answered("resuming the campaign "+id, resumed.JSON200, resumed.HTTPResponse, resumed.Body)
}

// Stop ends a campaign for good.
func (c *Campaigns) Stop(ctx context.Context, id string) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	stopped, err := api.StopCampaignWithResponse(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("client: stopping the campaign %s: %w", id, err)
	}
	return answered("stopping the campaign "+id, stopped.JSON200, stopped.HTTPResponse, stopped.Body)
}

// Duplicate copies a campaign into a new draft, with none, all or the never rung of its
// contacts.
func (c *Campaigns) Duplicate(ctx context.Context, id string, contacts acceleration.DuplicateCampaignRequestContacts) (*acceleration.Campaign, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	copied, err := api.DuplicateCampaignWithResponse(ctx, id, acceleration.DuplicateCampaignRequest{Contacts: contacts})
	if err != nil {
		return nil, fmt.Errorf("client: duplicating the campaign %s: %w", id, err)
	}
	return answered("duplicating the campaign "+id, copied.JSON201, copied.HTTPResponse, copied.Body)
}

// Analytics returns how a campaign's calls went.
func (c *Campaigns) Analytics(ctx context.Context, id string) (*acceleration.CampaignAnalytics, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	read, err := api.GetCampaignAnalyticsWithResponse(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("client: reading the analytics of the campaign %s: %w", id, err)
	}
	return answered("reading the analytics of the campaign "+id, read.JSON200, read.HTTPResponse, read.Body)
}

// AddContacts adds up to 1,000 contacts to a draft campaign. A row the router could not take
// is not an error: it is in the answer's Refused, at its index in the request.
func (c *Campaigns) AddContacts(ctx context.Context, id string, contacts acceleration.AddContactsRequest) (*acceleration.AddContactsResponse, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	added, err := api.AddCampaignContactsWithResponse(ctx, id, contacts)
	if err != nil {
		return nil, fmt.Errorf("client: adding contacts to the campaign %s: %w", id, err)
	}
	return answered("adding contacts to the campaign "+id, added.JSON201, added.HTTPResponse, added.Body)
}

// DeleteContacts removes every contact from a draft campaign.
func (c *Campaigns) DeleteContacts(ctx context.Context, id string) error {
	api, err := c.client.api()
	if err != nil {
		return err
	}
	deleted, err := api.DeleteCampaignContactsWithResponse(ctx, id)
	if err != nil {
		return fmt.Errorf("client: deleting the contacts of the campaign %s: %w", id, err)
	}
	if deleted.StatusCode() != http.StatusNoContent {
		return failure("deleting the contacts of the campaign "+id, deleted.HTTPResponse, deleted.Body)
	}
	return nil
}

// ListContacts returns a page of a campaign's contacts.
func (c *Campaigns) ListContacts(ctx context.Context, id string, options ContactListOptions) (*acceleration.ContactPage, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	listed, err := api.ListCampaignContactsWithResponse(ctx, id, &acceleration.ListCampaignContactsParams{
		Stage:  pointer(options.Stage),
		Number: pointer(options.Number),
		Limit:  pointer(int64(options.Limit)),
		Cursor: pointer(options.Cursor),
	})
	if err != nil {
		return nil, fmt.Errorf("client: listing the contacts of the campaign %s: %w", id, err)
	}
	return answered("listing the contacts of the campaign "+id, listed.JSON200, listed.HTTPResponse, listed.Body)
}

// GetContact returns one of a campaign's contacts, with its attempts.
func (c *Campaigns) GetContact(ctx context.Context, id, contactID string) (*acceleration.Contact, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	got, err := api.GetCampaignContactWithResponse(ctx, id, contactID)
	if err != nil {
		return nil, fmt.Errorf("client: getting the contact %s of the campaign %s: %w", contactID, id, err)
	}
	return answered("getting the contact "+contactID+" of the campaign "+id, got.JSON200, got.HTTPResponse, got.Body)
}

// StopContact rings a contact no more. One on a call is stopped when the call ends.
func (c *Campaigns) StopContact(ctx context.Context, id, contactID string) (*acceleration.Contact, error) {
	api, err := c.client.api()
	if err != nil {
		return nil, err
	}
	stopped, err := api.StopCampaignContactWithResponse(ctx, id, contactID)
	if err != nil {
		return nil, fmt.Errorf("client: stopping the contact %s of the campaign %s: %w", contactID, id, err)
	}
	return answered("stopping the contact "+contactID+" of the campaign "+id, stopped.JSON200, stopped.HTTPResponse, stopped.Body)
}

// answered is the typed answer the generated client decoded, which it does only for the
// success status, or what the router said went wrong instead.
func answered[T any](what string, typed *T, response *http.Response, body []byte) (*T, error) {
	if typed == nil {
		return nil, failure(what, response, body)
	}
	return typed, nil
}
