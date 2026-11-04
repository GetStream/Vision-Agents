//go:build integration

package store

import "time"

// dialed records an ended dial with result and the phone outcome of its call.
func (s *StoreSuite) dialed(contact Contact, at time.Time, result, callOutcome string) {
	attempt := CampaignAttempt{CampaignID: contact.CampaignID, ContactID: contact.ID, StartedAt: at, Result: result, CallOutcome: callOutcome}
	s.Require().NoError(s.store.StartDial(s.ctx, &attempt))
}

func (s *StoreSuite) settled(contact Contact, stage, outcome string, converted bool) {
	_, err := s.store.DB().NewUpdate().Model((*Contact)(nil)).
		Set("stage = ?", stage).Set("outcome = ?", outcome).Set("converted = ?", converted).
		Where("id = ?", contact.ID).Exec(s.ctx)
	s.Require().NoError(err)
}

func (s *StoreSuite) TestAnalyticsCountContactsNotDials() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002")
	contact := s.contactByNumber(campaign, "+15550001")
	s.dialed(contact, s.base, "", "no_answer")
	s.dialed(contact, s.base.Add(time.Minute), "", "completed")
	s.dialed(contact, s.base.Add(26*time.Hour), "", "busy")
	s.settled(contact, ContactCompleted, "interested", true)

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal(1, got.Dialed)
	s.Equal(1, got.Connected)
	s.Equal(1, got.Converted)
	s.Equal([]DialDay{
		{Day: s.base.Truncate(24 * time.Hour), Dials: 2, Answered: 1},
		{Day: s.base.Truncate(24 * time.Hour).Add(24 * time.Hour), Dials: 1},
	}, got.Days)

	totals, err := s.store.CampaignTotals(s.ctx, []string{campaign.ID})
	s.Require().NoError(err)
	s.Equal(CampaignTotal{Dialed: 1, Converted: 1}, totals[campaign.ID])
}

func (s *StoreSuite) TestOutcomesCountOnlyFinishedContacts() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002", "+15550003", "+15550004")
	s.settled(s.contactByNumber(campaign, "+15550001"), ContactCompleted, "interested", false)
	s.settled(s.contactByNumber(campaign, "+15550002"), ContactCompleted, "interested", false)
	s.settled(s.contactByNumber(campaign, "+15550003"), ContactCompleted, "no_response", false)
	s.settled(s.contactByNumber(campaign, "+15550004"), Enrolled, "interested", false)

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal(map[string]int{"interested": 2, "no_response": 1}, got.Outcomes)
	s.Empty(got.Days)
}

func (s *StoreSuite) TestStagesFoldAwaitingOutcomeIntoInProgress() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002", "+15550003")
	other := s.draftCampaign(newID(), "+15550009")
	s.settled(s.contactByNumber(campaign, "+15550001"), InProgress, "", false)
	s.settled(s.contactByNumber(campaign, "+15550002"), AwaitingOutcome, "", false)

	got, err := s.store.ContactStages(s.ctx, []string{campaign.ID, other.ID})
	s.Require().NoError(err)

	s.Equal(map[string]int{InProgress: 2, Enrolled: 1}, got[campaign.ID])
	s.Equal(map[string]int{Enrolled: 1}, got[other.ID])
}

func (s *StoreSuite) TestOnlyAttemptsThatPlacedACallAreDials() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002", "+15550003")
	mixed := s.contactByNumber(campaign, "+15550001")
	for _, result := range []string{"deferred", "paused", "refused", "invalid_number", "not_placed"} {
		s.dialed(mixed, s.base.Add(-48*time.Hour), result, "")
	}
	s.dialed(mixed, s.base, "", "")
	s.dialed(s.contactByNumber(campaign, "+15550002"), s.base, "refused", "")

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal(1, got.Dialed)
	s.Equal(0, got.Connected)
	s.Equal([]DialDay{{Day: s.base.Truncate(24 * time.Hour), Dials: 1}}, got.Days)
}

func (s *StoreSuite) TestAnEmptyCampaignHasZeroAnalytics() {
	campaign := s.draftCampaign(newID())

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal(CampaignAnalytics{Outcomes: map[string]int{}, CallOutcomes: map[string]int{}, Days: []DialDay{}}, got)
}

func (s *StoreSuite) TestDialsFallOnUTCDaysAtMidnight() {
	campaign := s.draftCampaign(newID(), "+15550001")
	contact := s.contactByNumber(campaign, "+15550001")
	day := s.base.Truncate(24 * time.Hour)
	s.dialed(contact, day.Add(-time.Minute), "", "no_answer")
	s.dialed(contact, day.Add(time.Minute), "", "completed")

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal([]DialDay{{Day: day.Add(-24 * time.Hour), Dials: 1}, {Day: day, Dials: 1, Answered: 1}}, got.Days)
}

func (s *StoreSuite) TestVoicemailDialsAreDialsAndTheirOwnSeries() {
	campaign := s.draftCampaign(newID(), "+15550001")
	contact := s.contactByNumber(campaign, "+15550001")
	s.dialed(contact, s.base, "voicemail", "completed")
	s.dialed(contact, s.base.Add(time.Minute), "", "no_answer")

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal(1, got.Dialed)
	s.Equal(0, got.Connected, "a machine is not a person who answered")
	s.Equal([]DialDay{{Day: s.base.Truncate(24 * time.Hour), Dials: 2, Voicemail: 1}}, got.Days)
	made, err := s.store.AttemptsMade(s.ctx, []string{contact.ID})
	s.Require().NoError(err)
	s.Equal(1, made[contact.ID])
}

func (s *StoreSuite) TestACallNobodySpokeOnIsNotAnAnswer() {
	campaign := s.draftCampaign(newID(), "+15550001")
	contact := s.contactByNumber(campaign, "+15550001")
	silent := CampaignAttempt{CampaignID: campaign.ID, ContactID: contact.ID, StartedAt: s.base, CallOutcome: "completed", Outcome: "no_response"}
	s.Require().NoError(s.store.StartDial(s.ctx, &silent))

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal(1, got.Dialed)
	s.Equal(0, got.Connected, "the phone joined, but no person spoke")
	s.Equal([]DialDay{{Day: s.base.Truncate(24 * time.Hour), Dials: 1}}, got.Days)
}

func (s *StoreSuite) TestCallsAreCountedByHowThePhoneSaysTheyEnded() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002")
	first, second := s.contactByNumber(campaign, "+15550001"), s.contactByNumber(campaign, "+15550002")
	s.dialed(first, s.base, "", "busy")
	s.dialed(first, s.base.Add(time.Minute), "", "busy")
	s.dialed(second, s.base, "", "no_answer")
	s.dialed(second, s.base.Add(time.Hour), "voicemail", "completed")
	s.dialed(second, s.base.Add(2*time.Hour), "paused", "")
	s.dialed(second, s.base.Add(3*time.Hour), "", "")

	got, err := s.store.CampaignAnalytics(s.ctx, campaign.ID)
	s.Require().NoError(err)

	s.Equal(map[string]int{"busy": 2, "no_answer": 1, "completed": 1}, got.CallOutcomes, "a call still happening and a dial that placed none are left out")
}
