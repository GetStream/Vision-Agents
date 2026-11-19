//go:build integration

package store

import (
	"time"
)

// autoEnrollCampaign is a draft auto-enroll campaign for a fresh customer, launched to start at
// startAt.
func (s *StoreSuite) autoEnrollCampaign(startAt time.Time) Campaign {
	campaign := Campaign{CustomerID: newID(), Name: "Inbound leads", ConfigID: "config-1", FromNumber: "+15550100", Type: AutoEnroll}
	s.Require().NoError(s.store.CreateCampaign(s.ctx, &campaign))
	return s.launch(campaign, startAt)
}

func (s *StoreSuite) addOne(campaign Campaign, number string) (Contact, error) {
	added, _, err := s.store.AddContacts(s.ctx, campaign.ID, []Contact{{CampaignID: campaign.ID, ToNumber: number}})
	if err != nil {
		return Contact{}, err
	}
	s.Require().Len(added, 1)
	return s.contactByNumber(campaign, number), nil
}

func (s *StoreSuite) claimedNumbers(now time.Time) []string {
	var numbers []string
	for _, claimed := range s.claimDue("a", now, 10) {
		numbers = append(numbers, claimed.Contact.ToNumber)
	}
	return numbers
}

func (s *StoreSuite) TestARunningAutoEnrollCampaignRingsAContactAddedToItFromTheFirstStep() {
	campaign := s.autoEnrollCampaign(s.base)
	before := time.Now().UTC()

	added, err := s.addOne(campaign, "+15550001")

	s.Require().NoError(err)
	s.Equal(Enrolled, added.Stage)
	s.Equal("step_1", added.StepID)
	s.Equal(0, added.Attempt)
	s.Require().NotNil(added.NextAttemptAt, "a contact with no next attempt is never rung")
	s.False(added.NextAttemptAt.Before(before.Truncate(time.Microsecond)), "due now, not at the start long past")
	s.Equal([]string{"+15550001"}, s.claimedNumbers(time.Now().Add(time.Minute)))
}

func (s *StoreSuite) TestAContactAddedToAPausedAutoEnrollCampaignWaitsForTheResume() {
	campaign := s.autoEnrollCampaign(s.base)
	s.Require().NoError(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, ""))

	_, err := s.addOne(campaign, "+15550001")

	s.Require().NoError(err)
	s.Empty(s.claimedNumbers(time.Now().Add(time.Minute)))
	s.Require().NoError(s.store.ResumeCampaign(s.ctx, campaign.CustomerID, campaign.ID))
	s.Equal([]string{"+15550001"}, s.claimedNumbers(time.Now().Add(time.Minute)))
}

func (s *StoreSuite) TestAContactAddedBeforeAScheduledStartIsDueAtTheStart() {
	start := time.Now().UTC().Add(24 * time.Hour).Truncate(time.Second)
	campaign := s.autoEnrollCampaign(start)
	s.Require().Equal(Scheduled, campaign.State)

	added, err := s.addOne(campaign, "+15550001")

	s.Require().NoError(err)
	s.Require().NotNil(added.NextAttemptAt)
	s.Equal(start, added.NextAttemptAt.UTC())
}

func (s *StoreSuite) TestAnOptedOutContactAddedAfterLaunchIsRejectedWithoutACall() {
	campaign := s.autoEnrollCampaign(s.base)
	s.Require().NoError(s.store.OptOut(s.ctx, &OptOut{CustomerID: campaign.CustomerID, Recipient: "+15550001", Channel: "voice", Source: "test"}))

	added, err := s.addOne(campaign, "+15550001")

	s.Require().NoError(err)
	s.Equal(Rejected, added.Stage)
	s.Equal("opted_out", added.Reason)
	s.Nil(added.NextAttemptAt)
	s.Empty(s.claimedNumbers(time.Now().Add(time.Minute)))
}

func (s *StoreSuite) TestAStoppedAutoEnrollCampaignTakesNoMoreContacts() {
	campaign := s.autoEnrollCampaign(s.base)
	s.Require().NoError(s.store.StopCampaign(s.ctx, campaign.CustomerID, campaign.ID, s.base))

	_, err := s.addOne(campaign, "+15550001")

	s.ErrorIs(err, ErrCampaignLocked)
	s.Empty(s.contactsOf(campaign))
}

func (s *StoreSuite) TestADraftsContactCanBeChanged() {
	campaign := s.draftCampaign(newID(), "+15550001")
	contact := s.contactByNumber(campaign, "+15550001")

	changed, err := s.store.UpdateContact(s.ctx, campaign.ID, contact.ID, Contact{
		ToNumber: "+15550009", Variables: map[string]string{"first_name": "Anna"}, TimeZone: "Europe/Amsterdam", Instructions: "Speak Dutch",
	})

	s.Require().NoError(err)
	s.Equal(contact.ID, changed.ID)
	s.Equal(contact.Seq, changed.Seq, "a change keeps the contact's place in the queue")
	stored := s.contactsOf(campaign)
	s.Require().Len(stored, 1)
	s.Equal("+15550009", stored[0].ToNumber)
	s.Equal(map[string]string{"first_name": "Anna"}, stored[0].Variables)
	s.Equal("Europe/Amsterdam", stored[0].TimeZone)
	s.Equal("Speak Dutch", stored[0].Instructions)
}

func (s *StoreSuite) TestAContactCannotTakeANumberAnotherContactHolds() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002")
	contact := s.contactByNumber(campaign, "+15550001")

	_, err := s.store.UpdateContact(s.ctx, campaign.ID, contact.ID, Contact{ToNumber: "+15550002"})

	s.ErrorIs(err, ErrDuplicateContact)
	s.Equal("+15550001", s.contactByNumber(campaign, "+15550001").ToNumber)
}

func (s *StoreSuite) TestOneContactCanBeRemovedFromADraft() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002")

	s.Require().NoError(s.store.DeleteContact(s.ctx, campaign.ID, s.contactByNumber(campaign, "+15550001").ID))

	remaining := s.contactsOf(campaign)
	s.Require().Len(remaining, 1)
	s.Equal("+15550002", remaining[0].ToNumber)
}

func (s *StoreSuite) TestALaunchedCampaignsContactsCannotBeChangedOrRemoved() {
	campaign := s.autoEnrollCampaign(s.base)
	contact, err := s.addOne(campaign, "+15550001")
	s.Require().NoError(err)

	_, err = s.store.UpdateContact(s.ctx, campaign.ID, contact.ID, Contact{ToNumber: "+15550002"})
	s.ErrorIs(err, ErrCampaignLocked)
	s.ErrorIs(s.store.DeleteContact(s.ctx, campaign.ID, contact.ID), ErrCampaignLocked)
	s.Equal([]string{"+15550001"}, []string{s.contactsOf(campaign)[0].ToNumber})
}

func (s *StoreSuite) TestChangingOrRemovingAContactOfAnotherCampaignFindsNone() {
	campaign := s.draftCampaign(newID(), "+15550001")
	other := s.draftCampaign(newID(), "+15550001")
	contact := s.contactByNumber(other, "+15550001")

	_, err := s.store.UpdateContact(s.ctx, campaign.ID, contact.ID, Contact{ToNumber: "+15550002"})
	s.ErrorIs(err, ErrNoCampaignContact)
	s.ErrorIs(s.store.DeleteContact(s.ctx, campaign.ID, contact.ID), ErrNoCampaignContact)
	s.Equal("+15550001", s.contactByNumber(other, "+15550001").ToNumber)
}
