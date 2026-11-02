package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// defaultConcurrency is how many calls a campaign that did not say may have at once. One
// is the safe answer: a campaign that rings a thousand people the moment it starts is a
// mistake nobody meant to make.
const defaultConcurrency = 1

// maxConcurrency bounds what a customer may ask for, since every call in flight is a
// conversation this process is running.
const maxConcurrency = 50

// CreateCampaign stores a campaign and fills in its id and timestamps.
func (s *Store) CreateCampaign(ctx context.Context, campaign *Campaign) error {
	if campaign.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if campaign.Name == "" {
		return stack.Wrap(errors.New("store: a campaign needs a name"))
	}
	if campaign.Concurrency <= 0 {
		campaign.Concurrency = defaultConcurrency
	}
	if campaign.Concurrency > maxConcurrency {
		return stack.Wrap(fmt.Errorf("store: a campaign may run at most %d calls at once", maxConcurrency))
	}

	campaign.ID = newID()
	campaign.State = Draft
	campaign.Type = Standard
	campaign.CreatedAt = time.Now().UTC()
	if campaign.Tags == nil {
		campaign.Tags = map[string]string{}
	}

	if _, err := s.db.NewInsert().Model(campaign).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: create campaign: %w", err))
	}
	return nil
}

// Campaign returns one campaign a customer holds.
func (s *Store) Campaign(ctx context.Context, customerID, id string) (Campaign, error) {
	if customerID == "" || id == "" {
		return Campaign{}, stack.Wrap(errors.New("store: a customer and a campaign id are required"))
	}

	var campaign Campaign
	err := s.db.NewSelect().Model(&campaign).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Campaign{}, unknownCampaign(id)
	}
	if err != nil {
		return Campaign{}, stack.Wrap(fmt.Errorf("store: campaign: %w", err))
	}
	return campaign, nil
}

// CustomerCampaigns returns a customer's campaigns, newest first.
func (s *Store) CustomerCampaigns(ctx context.Context, customerID string) ([]Campaign, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	var campaigns []Campaign
	err := s.db.NewSelect().Model(&campaigns).
		Where("customer_id = ?", customerID).
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer campaigns: %w", err))
	}
	return campaigns, nil
}

// SetCampaignState records what a campaign is now doing. Starting one stamps when it
// first started, and finishing one stamps when it stopped having anybody left to ring. Either
// clears why the router last paused it, and that a data move did (ResumeMovedRows): the
// customer has decided what it does here.
func (s *Store) SetCampaignState(ctx context.Context, customerID, id, state string) error {
	return s.setCampaignState(ctx, customerID, id, state, "")
}

// PauseCampaign pauses a campaign the router stopped itself, saying why with the code of what
// refused its next call.
func (s *Store) PauseCampaign(ctx context.Context, customerID, id, reason string) error {
	return s.setCampaignState(ctx, customerID, id, Paused, reason)
}

func (s *Store) setCampaignState(ctx context.Context, customerID, id, state, reason string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a campaign id are required"))
	}

	now := time.Now().UTC()
	query := s.db.NewUpdate().Model((*Campaign)(nil)).
		Set("state = ?", state).
		Set("paused_reason = NULLIF(?, '')", reason).
		Set("move_paused = false").
		Where("id = ?", id).
		Where("customer_id = ?", customerID)
	switch state {
	case Running:
		query = query.Set("started_at = COALESCE(started_at, ?)", now).
			Set("finished_at = NULL")
	case Completed:
		query = query.Set("finished_at = ?", now)
	}

	result, err := query.Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set campaign state: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set campaign state: %w", err))
	}
	if affected == 0 {
		return unknownCampaign(id)
	}
	return nil
}

// AddContacts stores people to ring. They are added rather than replaced, so a campaign
// can be topped up while it is running.
func (s *Store) AddContacts(ctx context.Context, contacts []Contact) error {
	if len(contacts) == 0 {
		return stack.Wrap(errors.New("store: there is nobody to add"))
	}

	now := time.Now().UTC()
	for index := range contacts {
		if contacts[index].CampaignID == "" {
			return stack.Wrap(errors.New("store: a contact belongs to a campaign"))
		}
		if contacts[index].ToNumber == "" {
			return stack.Wrap(errors.New("store: a contact needs a number to ring"))
		}
		contacts[index].ID = newID()
		contacts[index].Stage = Enrolled
		contacts[index].CreatedAt = now
	}

	if _, err := s.db.NewInsert().Model(&contacts).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: add contacts: %w", err))
	}
	return nil
}

// CampaignContacts returns a campaign's contacts, oldest first, which is the order they
// are rung in.
func (s *Store) CampaignContacts(ctx context.Context, campaignID string) ([]Contact, error) {
	if campaignID == "" {
		return nil, stack.Wrap(errors.New("store: a campaign id is required"))
	}

	var contacts []Contact
	err := s.db.NewSelect().Model(&contacts).
		Where("campaign_id = ?", campaignID).
		Order("seq ASC").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: campaign contacts: %w", err))
	}
	return contacts, nil
}

// nullable keeps an empty string out of a column that means "nothing happened" by being
// null rather than blank.
func nullable(text string) any {
	if text == "" {
		return nil
	}
	return text
}

func unknownCampaign(id string) error {
	return stack.Wrap(fmt.Errorf("store: there is no campaign %s", id))
}
