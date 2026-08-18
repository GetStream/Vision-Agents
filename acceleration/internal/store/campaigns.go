package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
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
		return errors.New("store: customer id is required")
	}
	if campaign.Name == "" {
		return errors.New("store: a campaign needs a name")
	}
	if campaign.Concurrency <= 0 {
		campaign.Concurrency = defaultConcurrency
	}
	if campaign.Concurrency > maxConcurrency {
		return fmt.Errorf("store: a campaign may run at most %d calls at once", maxConcurrency)
	}

	campaign.ID = newID()
	campaign.State = Draft
	campaign.CreatedAt = time.Now().UTC()
	if campaign.Tags == nil {
		campaign.Tags = map[string]string{}
	}

	if _, err := s.db.NewInsert().Model(campaign).Exec(ctx); err != nil {
		return fmt.Errorf("store: create campaign: %w", err)
	}
	return nil
}

// Campaign returns one campaign a customer holds.
func (s *Store) Campaign(ctx context.Context, customerID, id string) (Campaign, error) {
	if customerID == "" || id == "" {
		return Campaign{}, errors.New("store: a customer and a campaign id are required")
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
		return Campaign{}, fmt.Errorf("store: campaign: %w", err)
	}
	return campaign, nil
}

// CustomerCampaigns returns a customer's campaigns, newest first.
func (s *Store) CustomerCampaigns(ctx context.Context, customerID string) ([]Campaign, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var campaigns []Campaign
	err := s.db.NewSelect().Model(&campaigns).
		Where("customer_id = ?", customerID).
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: customer campaigns: %w", err)
	}
	return campaigns, nil
}

// SetCampaignState records what a campaign is now doing. Starting one stamps when it
// first started, and finishing one stamps when it stopped having anybody left to ring.
func (s *Store) SetCampaignState(ctx context.Context, customerID, id, state string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a campaign id are required")
	}

	now := time.Now().UTC()
	query := s.db.NewUpdate().Model((*Campaign)(nil)).
		Set("state = ?", state).
		Where("id = ?", id).
		Where("customer_id = ?", customerID)
	switch state {
	case Running:
		query = query.Set("started_at = COALESCE(started_at, ?)", now).
			Set("finished_at = NULL")
	case Finished:
		query = query.Set("finished_at = ?", now)
	}

	result, err := query.Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: set campaign state: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: set campaign state: %w", err)
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
		return errors.New("store: there is nobody to add")
	}

	now := time.Now().UTC()
	for index := range contacts {
		if contacts[index].CampaignID == "" {
			return errors.New("store: a contact belongs to a campaign")
		}
		if contacts[index].ToNumber == "" {
			return errors.New("store: a contact needs a number to ring")
		}
		contacts[index].ID = newID()
		contacts[index].State = Pending
		contacts[index].CreatedAt = now
	}

	if _, err := s.db.NewInsert().Model(&contacts).Exec(ctx); err != nil {
		return fmt.Errorf("store: add contacts: %w", err)
	}
	return nil
}

// CampaignContacts returns a campaign's contacts, oldest first, which is the order they
// are rung in.
func (s *Store) CampaignContacts(ctx context.Context, campaignID string) ([]Contact, error) {
	if campaignID == "" {
		return nil, errors.New("store: a campaign id is required")
	}

	var contacts []Contact
	err := s.db.NewSelect().Model(&contacts).
		Where("campaign_id = ?", campaignID).
		Order("seq ASC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: campaign contacts: %w", err)
	}
	return contacts, nil
}

// ClaimContact takes the next person to ring, marking them as being called so nobody
// else takes them.
//
// The claim is the update rather than the select: two runners over one campaign is not
// something this process arranges, but a restart mid-call is, and a contact left as
// calling forever is worse than one rung twice.
func (s *Store) ClaimContact(ctx context.Context, campaignID string) (Contact, bool, error) {
	if campaignID == "" {
		return Contact{}, false, errors.New("store: a campaign id is required")
	}

	var claimed Contact
	err := s.db.NewUpdate().Model(&claimed).
		Set("state = ?", Calling).
		Set("attempts = attempts + 1").
		Where("id = (?)",
			s.db.NewSelect().Model((*Contact)(nil)).
				Column("id").
				Where("campaign_id = ?", campaignID).
				Where("state = ?", Pending).
				Order("seq ASC").
				Limit(1).
				For("UPDATE SKIP LOCKED")).
		Returning("*").
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Contact{}, false, nil
	}
	if err != nil {
		return Contact{}, false, fmt.Errorf("store: claim contact: %w", err)
	}
	return claimed, true, nil
}

// FinishContact records what became of ringing somebody. A contact that could not be
// rung keeps the reason, since that is the only place it is ever explained.
func (s *Store) FinishContact(ctx context.Context, contact Contact) error {
	if contact.ID == "" {
		return errors.New("store: a contact id is required")
	}

	_, err := s.db.NewUpdate().Model((*Contact)(nil)).
		Set("state = ?", contact.State).
		Set("call_id = ?", nullable(contact.CallID)).
		Set("vendor_call_id = ?", nullable(contact.VendorCallID)).
		Set("error = ?", nullable(contact.Error)).
		Where("id = ?", contact.ID).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish contact: %w", err)
	}
	return nil
}

// ReleaseContacts puts calls that never finished back in the queue, which is what a
// paused campaign and a restarted process both leave behind.
func (s *Store) ReleaseContacts(ctx context.Context, campaignID string) error {
	if campaignID == "" {
		return errors.New("store: a campaign id is required")
	}

	_, err := s.db.NewUpdate().Model((*Contact)(nil)).
		Set("state = ?", Pending).
		Where("campaign_id = ?", campaignID).
		Where("state = ?", Calling).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: release contacts: %w", err)
	}
	return nil
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
	return fmt.Errorf("store: there is no campaign %s", id)
}
