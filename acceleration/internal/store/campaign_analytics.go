package store

import (
	"context"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// CampaignAnalytics is how a campaign's calls went. Dialed, Connected and Converted count
// contacts, not dials: one contact rung three times is one.
type CampaignAnalytics struct {
	Dialed    int
	Connected int
	Converted int
	// Outcomes counts the contacts that are done, by outcome.
	Outcomes map[string]int
	Days     []DialDay
}

// DialDay is the dials placed on one UTC day, and how many were answered.
type DialDay struct {
	Day      time.Time
	Dials    int
	Answered int
}

// CampaignTotal is what the campaign list shows of each campaign.
type CampaignTotal struct {
	Dialed    int
	Converted int
}

// CampaignAnalytics counts a campaign's contacts and dials. Days run oldest first and hold
// only the days with a dial.
func (s *Store) CampaignAnalytics(ctx context.Context, campaignID string) (CampaignAnalytics, error) {
	totals, err := s.CampaignTotals(ctx, []string{campaignID})
	if err != nil {
		return CampaignAnalytics{}, err
	}
	analytics := CampaignAnalytics{
		Dialed:    totals[campaignID].Dialed,
		Converted: totals[campaignID].Converted,
		Outcomes:  map[string]int{},
	}

	err = s.db.NewRaw(`SELECT count(*) FROM campaign_contacts cc
WHERE cc.campaign_id = ? AND EXISTS (SELECT 1 FROM campaign_attempts ca
	WHERE ca.contact_id = cc.id AND ca.result = 'answered')`, campaignID).Scan(ctx, &analytics.Connected)
	if err != nil {
		return CampaignAnalytics{}, stack.Wrap(fmt.Errorf("store: count connected contacts: %w", err))
	}

	var outcomes []struct {
		Outcome string
		Count   int
	}
	err = s.db.NewRaw(`SELECT outcome, count(*) AS count FROM campaign_contacts
WHERE campaign_id = ? AND stage = ? GROUP BY outcome`, campaignID, ContactCompleted).Scan(ctx, &outcomes)
	if err != nil {
		return CampaignAnalytics{}, stack.Wrap(fmt.Errorf("store: count outcomes: %w", err))
	}
	for _, row := range outcomes {
		analytics.Outcomes[row.Outcome] = row.Count
	}

	err = s.db.NewRaw(`SELECT date_trunc('day', started_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' AS day,
	count(*) AS dials, count(*) FILTER (WHERE result = 'answered') AS answered
FROM campaign_attempts WHERE campaign_id = ? GROUP BY 1 ORDER BY 1`, campaignID).Scan(ctx, &analytics.Days)
	if err != nil {
		return CampaignAnalytics{}, stack.Wrap(fmt.Errorf("store: count dials per day: %w", err))
	}
	for i := range analytics.Days {
		analytics.Days[i].Day = analytics.Days[i].Day.UTC()
	}
	return analytics, nil
}

// ContactStages counts each campaign's contacts by stage. A call awaiting its outcome counts
// as in progress, which is how the API shows it.
func (s *Store) ContactStages(ctx context.Context, campaignIDs []string) (map[string]map[string]int, error) {
	stages := map[string]map[string]int{}
	if len(campaignIDs) == 0 {
		return stages, nil
	}
	var rows []struct {
		CampaignID string
		Stage      string
		Count      int
	}
	err := s.db.NewRaw(`SELECT campaign_id, CASE WHEN stage = ? THEN ? ELSE stage END AS stage, count(*) AS count
FROM campaign_contacts WHERE campaign_id IN (?) GROUP BY 1, 2`,
		AwaitingOutcome, InProgress, bun.In(campaignIDs)).Scan(ctx, &rows)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: count contact stages: %w", err))
	}
	for _, row := range rows {
		if stages[row.CampaignID] == nil {
			stages[row.CampaignID] = map[string]int{}
		}
		stages[row.CampaignID][row.Stage] = row.Count
	}
	return stages, nil
}

// CampaignTotals counts, per campaign, the contacts dialed and converted, as CampaignAnalytics does.
func (s *Store) CampaignTotals(ctx context.Context, campaignIDs []string) (map[string]CampaignTotal, error) {
	totals := map[string]CampaignTotal{}
	if len(campaignIDs) == 0 {
		return totals, nil
	}
	var rows []struct {
		CampaignID string
		Dialed     int
		Converted  int
	}
	err := s.db.NewRaw(`SELECT cc.campaign_id,
	count(*) FILTER (WHERE EXISTS (SELECT 1 FROM campaign_attempts ca WHERE ca.contact_id = cc.id)) AS dialed,
	count(*) FILTER (WHERE cc.converted) AS converted
FROM campaign_contacts cc WHERE cc.campaign_id IN (?) GROUP BY cc.campaign_id`, bun.In(campaignIDs)).Scan(ctx, &rows)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: count campaign totals: %w", err))
	}
	for _, row := range rows {
		totals[row.CampaignID] = CampaignTotal{Dialed: row.Dialed, Converted: row.Converted}
	}
	return totals, nil
}
