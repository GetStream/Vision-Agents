//go:build integration

package store

import (
	"fmt"
	"sync"
	"time"
)

// runningCampaign launches a campaign that may hold concurrency calls at once, with one contact
// per number, all due at s.base.
func (s *StoreSuite) runningCampaign(concurrency int, numbers ...string) Campaign {
	campaign := s.draftCampaign(newID(), numbers...)
	_, err := s.store.UpdateCampaign(s.ctx, campaign.CustomerID, campaign.ID, CampaignChanges{Concurrency: &concurrency})
	s.Require().NoError(err)
	return s.launch(campaign, s.base)
}

func (s *StoreSuite) claimDue(owner string, now time.Time, max int) []Claimed {
	claimed, err := s.store.ClaimDueContacts(s.ctx, owner, now, time.Minute, max)
	s.Require().NoError(err)
	return claimed
}

func (s *StoreSuite) contactByNumber(campaign Campaign, number string) Contact {
	for _, contact := range s.contactsOf(campaign) {
		if contact.ToNumber == number {
			return contact
		}
	}
	s.FailNow("no contact " + number)
	return Contact{}
}

func (s *StoreSuite) dueAt(contact Contact, at time.Time) {
	_, err := s.store.DB().NewUpdate().Model((*Contact)(nil)).Set("next_attempt_at = ?", at).Where("id = ?", contact.ID).Exec(s.ctx)
	s.Require().NoError(err)
}

func (s *StoreSuite) campaignNow(campaign Campaign) Campaign {
	found, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	return found
}

func (s *StoreSuite) TestAClaimNeverExceedsTheCampaignsConcurrency() {
	// Several campaigns raced at once widen the window a check-then-claim would slip through.
	campaigns := make([]Campaign, 6)
	for i := range campaigns {
		campaigns[i] = s.runningCampaign(2, "+15550001", "+15550002", "+15550003", "+15550004", "+15550005")
	}

	var wg sync.WaitGroup
	start := make(chan struct{})
	results := map[string][]Claimed{}
	var mu sync.Mutex
	for _, owner := range []string{"a", "b"} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			claimed, err := s.store.ClaimDueContacts(s.ctx, owner, s.base, time.Minute, 100)
			s.NoError(err)
			mu.Lock()
			results[owner] = claimed
			mu.Unlock()
		}()
	}
	close(start)
	wg.Wait()

	perCampaign := map[string]int{}
	seen := map[string]bool{}
	for owner, claimed := range results {
		for _, one := range claimed {
			s.False(seen[one.Contact.ID], "contact %s claimed twice", one.Contact.ID)
			seen[one.Contact.ID] = true
			perCampaign[one.Campaign.ID]++
			s.Equal(owner, one.Contact.LeasedBy)
			s.Equal(InProgress, one.Contact.Stage)
		}
	}
	for _, campaign := range campaigns {
		s.Equal(2, perCampaign[campaign.ID], "campaign %s", campaign.ID)
		inProgress := 0
		for _, contact := range s.contactsOf(campaign) {
			if contact.Stage == InProgress {
				inProgress++
			}
		}
		s.Equal(2, inProgress, "campaign %s", campaign.ID)
	}
	s.Empty(s.claimDue("c", s.base, 100), "every campaign is full")
}

func (s *StoreSuite) TestAClaimTakesTheEarliestDueFirstAndStopsAtMax() {
	campaign := s.runningCampaign(5, "+15550001", "+15550002", "+15550003")
	s.dueAt(s.contactByNumber(campaign, "+15550001"), s.base.Add(-time.Minute))
	s.dueAt(s.contactByNumber(campaign, "+15550002"), s.base.Add(-2*time.Minute))

	claimed := s.claimDue("a", s.base, 2)

	s.Require().Len(claimed, 2)
	numbers := []string{claimed[0].Contact.ToNumber, claimed[1].Contact.ToNumber}
	s.ElementsMatch([]string{"+15550002", "+15550001"}, numbers)
	s.Equal(campaign.ID, claimed[0].Campaign.ID)
	s.Equal(s.base.Add(time.Minute), claimed[0].Contact.LeasedUntil.UTC())
}

func (s *StoreSuite) TestAContactNotYetDueIsNotClaimed() {
	campaign := s.runningCampaign(5, "+15550001", "+15550002")
	s.dueAt(s.contactByNumber(campaign, "+15550002"), s.base.Add(time.Hour))

	claimed := s.claimDue("a", s.base, 10)

	s.Require().Len(claimed, 1)
	s.Equal("+15550001", claimed[0].Contact.ToNumber)
	later := s.claimDue("a", s.base.Add(time.Hour), 10)
	s.Require().Len(later, 1)
	s.Equal("+15550002", later[0].Contact.ToNumber)
}

func (s *StoreSuite) TestAPausedCampaignsContactsAreNotClaimed() {
	campaign := s.runningCampaign(5, "+15550001")
	s.Require().NoError(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, ""))

	s.Empty(s.claimDue("a", s.base, 10))

	s.Require().NoError(s.store.ResumeCampaign(s.ctx, campaign.CustomerID, campaign.ID))
	s.Len(s.claimDue("a", s.base, 10), 1)
}

func (s *StoreSuite) TestAScheduledCampaignStartsWhenItsTimeComes() {
	campaign := s.draftCampaign(newID(), "+15550001")
	start := s.base.Add(time.Hour)
	s.launch(campaign, start)

	s.Require().NoError(s.store.StartDueCampaigns(s.ctx, start.Add(-time.Second)))
	s.Equal(Scheduled, s.campaignNow(campaign).State)
	s.Empty(s.claimDue("a", start, 10), "a scheduled campaign rings nobody")

	s.Require().NoError(s.store.StartDueCampaigns(s.ctx, start))
	started := s.campaignNow(campaign)
	s.Equal(Running, started.State)
	s.Require().NotNil(started.StartedAt)
	s.Equal(start, started.StartedAt.UTC())
	s.Len(s.claimDue("a", start, 10), 1)
}

func (s *StoreSuite) TestAnExpiredLeaseIsTakenOverAndALiveOneIsNot() {
	campaign := s.runningCampaign(5, "+15550001", "+15550002")
	s.dueAt(s.contactByNumber(campaign, "+15550002"), s.base.Add(5*time.Minute))
	_, err := s.store.ClaimDueContacts(s.ctx, "a", s.base, 10*time.Minute, 10)
	s.Require().NoError(err)
	_, err = s.store.ClaimDueContacts(s.ctx, "a", s.base.Add(5*time.Minute), time.Minute, 10)
	s.Require().NoError(err)

	taken, err := s.store.TakeOverExpired(s.ctx, "b", s.base.Add(7*time.Minute), time.Minute, 10)

	s.Require().NoError(err)
	s.Require().Len(taken, 1)
	s.Equal("+15550002", taken[0].Contact.ToNumber)
	s.Equal("b", taken[0].Contact.LeasedBy)
	s.Equal(s.base.Add(8*time.Minute), taken[0].Contact.LeasedUntil.UTC())
	s.Equal(campaign.ID, taken[0].Campaign.ID)
	s.Equal("a", s.contactByNumber(campaign, "+15550001").LeasedBy, "a live lease stays")
	again, err := s.store.TakeOverExpired(s.ctx, "c", s.base.Add(7*time.Minute), time.Minute, 10)
	s.Require().NoError(err)
	s.Empty(again, "b's new lease is live")
}

func (s *StoreSuite) TestOnlyTheLeaseHolderCanSettle() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	next := s.base.Add(time.Hour)

	err := s.store.SettleContact(s.ctx, "b", contact.ID, Settle{Stage: Enrolled, Attempt: 1, NextAttemptAt: &next})
	s.ErrorIs(err, ErrNotLeaseHolder)
	s.Equal(InProgress, s.contactByNumber(campaign, "+15550001").Stage)

	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: Enrolled, Attempt: 1, NextAttemptAt: &next, CallID: "call-1"}))
	settled := s.contactByNumber(campaign, "+15550001")
	s.Equal(Enrolled, settled.Stage)
	s.Equal(1, settled.Attempt)
	s.Equal(next, settled.NextAttemptAt.UTC())
	s.Equal("call-1", settled.CallID)
	s.Empty(settled.LeasedBy)
	s.Nil(settled.LeasedUntil)

	err = s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: ContactCompleted})
	s.ErrorIs(err, ErrNotLeaseHolder, "a released lease settles nothing")
	s.NoError(s.store.SettleContact(s.ctx, "a", "deleted-contact", Settle{Stage: ContactCompleted}), "a deleted contact is nothing to settle")
}

func (s *StoreSuite) TestSettlingAContactOfAStoppedCampaignEndsItStopped() {
	campaign := s.runningCampaign(5, "+15550001", "+15550002")
	claimed := s.claimDue("a", s.base, 10)
	s.Require().Len(claimed, 2)
	s.Require().NoError(s.store.StopCampaign(s.ctx, campaign.CustomerID, campaign.ID, s.base))
	next := s.base.Add(time.Hour)

	for _, one := range claimed {
		settle := Settle{Stage: Enrolled, Attempt: 1, NextAttemptAt: &next}
		if one.Contact.ToNumber == "+15550002" {
			settle = Settle{Stage: ContactCompleted, Attempt: 1, Outcome: "success", Converted: true}
		}
		s.Require().NoError(s.store.SettleContact(s.ctx, "a", one.Contact.ID, settle))
	}

	waiting := s.contactByNumber(campaign, "+15550001")
	s.Equal(ContactStopped, waiting.Stage)
	s.Nil(waiting.NextAttemptAt)
	done := s.contactByNumber(campaign, "+15550002")
	s.Equal(ContactCompleted, done.Stage)
	s.Equal("success", done.Outcome)
	s.True(done.Converted)
}

func (s *StoreSuite) TestSettlingAContactStoppedMidCallEndsItStopped() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	s.Require().NoError(s.store.StopContact(s.ctx, campaign.ID, contact.ID))
	next := s.base.Add(time.Hour)

	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: Enrolled, Attempt: 1, NextAttemptAt: &next}))

	settled := s.contactByNumber(campaign, "+15550001")
	s.Equal(ContactStopped, settled.Stage)
	s.Equal("stopped", settled.Reason)
	s.Nil(settled.NextAttemptAt)
	s.Equal(Running, s.campaignNow(campaign).State, "stopping one contact leaves the campaign running")
}

func (s *StoreSuite) TestAnAnsweredCallOfAStoppedCampaignStillAwaitsItsOutcome() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	s.Require().NoError(s.store.StopCampaign(s.ctx, campaign.CustomerID, campaign.ID, s.base))

	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))

	s.Equal(AwaitingOutcome, s.contactByNumber(campaign, "+15550001").Stage)
}

func (s *StoreSuite) TestSettlingAContactOfAPausedCampaignKeepsItEnrolled() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	s.Require().NoError(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, ""))
	next := s.base.Add(time.Hour)

	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: Enrolled, Attempt: 1, NextAttemptAt: &next}))

	settled := s.contactByNumber(campaign, "+15550001")
	s.Equal(Enrolled, settled.Stage)
	s.Equal(next, settled.NextAttemptAt.UTC())
}

func (s *StoreSuite) TestAnAnsweredDialWaitsToBeClassifiedAndIsClaimedOnce() {
	campaign := s.runningCampaign(5, "+15550001")
	claimed := s.claimDue("a", s.base, 10)[0]
	attempt := CampaignAttempt{CampaignID: campaign.ID, ContactID: claimed.Contact.ID, StepID: "step_1", Attempt: 1, Dial: 1, StartedAt: s.base}
	s.Require().NoError(s.store.StartDial(s.ctx, &attempt))
	s.NotEmpty(attempt.ID)
	classifyAfter := s.base.Add(2 * time.Minute)
	s.Require().NoError(s.store.EndDial(s.ctx, attempt.ID, "", "completed", "call-1", "vendor-1", s.base.Add(time.Minute), &classifyAfter))
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", claimed.Contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1, CallID: "call-1"}))

	latest, found, err := s.store.LatestDial(s.ctx, claimed.Contact.ID)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Empty(latest.Result)
	s.Equal("completed", latest.CallOutcome, "the phone's outcome is kept as is")
	s.Equal("call-1", latest.CallID)
	s.Equal("vendor-1", latest.VendorCallID)
	s.Equal("pending", latest.ClassifyState)
	early, err := s.store.ClaimClassifications(s.ctx, "a", s.base.Add(time.Minute), time.Minute, 10)
	s.Require().NoError(err)
	s.Empty(early, "the transcript is not written yet")

	var wg sync.WaitGroup
	start := make(chan struct{})
	results := make([][]Classifying, 2)
	for i, owner := range []string{"a", "b"} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			found, err := s.store.ClaimClassifications(s.ctx, owner, classifyAfter, time.Minute, 10)
			s.NoError(err)
			results[i] = found
		}()
	}
	close(start)
	wg.Wait()

	s.Equal(1, len(results[0])+len(results[1]), "one router classifies it")
	winner, loser := "a", "b"
	won := results[0]
	if len(won) == 0 {
		winner, loser, won = "b", "a", results[1]
	}
	s.Require().Len(won, 1)
	s.Equal(attempt.ID, won[0].Attempt.ID)
	s.Equal(1, won[0].Attempt.ClassifyTries)
	s.Equal(classifyAfter.Add(time.Minute), won[0].Attempt.ClassifyAfter.UTC())
	s.Equal(winner, won[0].Contact.LeasedBy)
	s.Equal(AwaitingOutcome, won[0].Contact.Stage)
	s.Equal(campaign.ID, won[0].Campaign.ID)

	s.Require().NoError(s.store.Classified(s.ctx, attempt.ID, "success", true))
	s.ErrorIs(s.store.SettleContact(s.ctx, loser, claimed.Contact.ID, Settle{Stage: ContactCompleted}), ErrNotLeaseHolder)
	s.Require().NoError(s.store.SettleContact(s.ctx, winner, claimed.Contact.ID, Settle{Stage: ContactCompleted, Attempt: 1, Outcome: "success", Converted: true}))
	latest, _, err = s.store.LatestDial(s.ctx, claimed.Contact.ID)
	s.Require().NoError(err)
	s.Equal("done", latest.ClassifyState)
	s.Equal("success", latest.Outcome)
	s.True(latest.Converted)
	s.Equal(ContactCompleted, s.contactByNumber(campaign, "+15550001").Stage)
	later, err := s.store.ClaimClassifications(s.ctx, "a", classifyAfter.Add(time.Hour), time.Minute, 10)
	s.Require().NoError(err)
	s.Empty(later, "a classified dial is not classified again")
}

func (s *StoreSuite) TestTheLatestDialIsTheSecondOfADoubleDial() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	_, found, err := s.store.LatestDial(s.ctx, contact.ID)
	s.Require().NoError(err)
	s.False(found)

	for dial := 1; dial <= 2; dial++ {
		attempt := CampaignAttempt{CampaignID: campaign.ID, ContactID: contact.ID, StepID: "step_1", Attempt: 1, Dial: dial, StartedAt: s.base}
		s.Require().NoError(s.store.StartDial(s.ctx, &attempt))
		s.Require().NoError(s.store.EndDial(s.ctx, attempt.ID, "", "no_answer", fmt.Sprintf("call-%d", dial), "", s.base, nil))
	}

	latest, found, err := s.store.LatestDial(s.ctx, contact.ID)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(2, latest.Dial)
	s.Equal("call-2", latest.CallID)
	s.Empty(latest.ClassifyState, "an unanswered dial has nothing to classify")
}

func (s *StoreSuite) TestEndingADialOfADeletedCampaignIsNothing() {
	s.NoError(s.store.EndDial(s.ctx, "deleted-attempt", "", "completed", "call-1", "", s.base, nil))
	s.NoError(s.store.Classified(s.ctx, "deleted-attempt", "success", true))
}

func (s *StoreSuite) TestACampaignCompletesWhenNobodyIsLeft() {
	campaign := s.runningCampaign(5, "+15550001", "+15550002")
	autoEnroll := Campaign{CustomerID: newID(), Name: "Inbound leads", ConfigID: "config-1", FromNumber: "+15550100", Type: AutoEnroll}
	s.Require().NoError(s.store.CreateCampaign(s.ctx, &autoEnroll))
	s.launch(autoEnroll, s.base)
	for _, one := range s.claimDue("a", s.base, 10) {
		s.Require().NoError(s.store.SettleContact(s.ctx, "a", one.Contact.ID, Settle{Stage: ContactCompleted, Attempt: 1}))
	}

	s.Require().NoError(s.store.CompleteCampaigns(s.ctx, s.base.Add(time.Hour)))

	completed := s.campaignNow(campaign)
	s.Equal(Completed, completed.State)
	s.Require().NotNil(completed.FinishedAt)
	s.Equal(s.base.Add(time.Hour), completed.FinishedAt.UTC())
	s.Equal(Running, s.campaignNow(autoEnroll).State, "an auto-enroll campaign waits for more people")
}

func (s *StoreSuite) TestACampaignWithACallAwaitingAnOutcomeIsNotComplete() {
	campaign := s.runningCampaign(5, "+15550001", "+15550002")
	waiting := s.runningCampaign(5, "+15550003")
	s.dueAt(s.contactByNumber(waiting, "+15550003"), s.base.Add(time.Hour))
	inCall := s.runningCampaign(5, "+15550004")
	for _, one := range s.claimDue("a", s.base, 10) {
		stage := ContactCompleted
		if one.Contact.ToNumber == "+15550002" {
			stage = AwaitingOutcome
		}
		if one.Campaign.ID == inCall.ID {
			continue
		}
		s.Require().NoError(s.store.SettleContact(s.ctx, "a", one.Contact.ID, Settle{Stage: stage, Attempt: 1}))
	}

	s.Require().NoError(s.store.CompleteCampaigns(s.ctx, s.base))

	s.Equal(Running, s.campaignNow(campaign).State)
	s.Equal(Running, s.campaignNow(waiting).State, "somebody is still to be rung")
	s.Equal(Running, s.campaignNow(inCall).State, "a call is in progress")
}

func (s *StoreSuite) TestOneClaimServesEveryCampaignWithContactsDue() {
	numbers := []string{"+15550001", "+15550002", "+15550003", "+15550004", "+15550005", "+15550006"}
	first := s.runningCampaign(5, numbers...)
	second := s.runningCampaign(5, numbers...)

	perCampaign := map[string]int{}
	for _, one := range s.claimDue("a", s.base, 4) {
		perCampaign[one.Campaign.ID]++
	}

	s.Equal(2, perCampaign[first.ID])
	s.Equal(2, perCampaign[second.ID])
}

func (s *StoreSuite) TestADialMustSayWhenItStarted() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact

	err := s.store.StartDial(s.ctx, &CampaignAttempt{CampaignID: campaign.ID, ContactID: contact.ID, StepID: "step_1", Attempt: 1, Dial: 1})

	s.Error(err)
}

// answered dials a contact, ends the dial as answered with its outcome to read at classifyAfter,
// and settles the contact to wait for it.
func (s *StoreSuite) answered(owner string, campaign Campaign, contact Contact, startedAt, classifyAfter time.Time) CampaignAttempt {
	attempt := CampaignAttempt{CampaignID: campaign.ID, ContactID: contact.ID, StepID: "step_1", Attempt: 1, Dial: 1, StartedAt: startedAt}
	s.Require().NoError(s.store.StartDial(s.ctx, &attempt))
	s.Require().NoError(s.store.EndDial(s.ctx, attempt.ID, "", "completed", "call-"+attempt.ID, "", startedAt, &classifyAfter))
	return attempt
}

func (s *StoreSuite) TestAContactStoppedMidCallStaysStoppedOnceItsOutcomeIsKnown() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	s.answered("a", campaign, contact, s.base, s.base.Add(time.Minute))
	s.Require().NoError(s.store.StopContact(s.ctx, campaign.ID, contact.ID))
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	s.Equal("stopped", s.contactByNumber(campaign, "+15550001").Reason, "waiting for the outcome keeps the stop")
	claimed, err := s.store.ClaimClassifications(s.ctx, "b", s.base.Add(time.Minute), time.Minute, 10)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	next := s.base.Add(time.Hour)

	s.Require().NoError(s.store.SettleContact(s.ctx, "b", contact.ID, Settle{Stage: Enrolled, StepID: "step_2", Attempt: 0, NextAttemptAt: &next, Outcome: "callback"}))

	settled := s.contactByNumber(campaign, "+15550001")
	s.Equal(ContactStopped, settled.Stage)
	s.Equal("stopped", settled.Reason)
	s.Nil(settled.NextAttemptAt)
}

func (s *StoreSuite) TestOnlyTheLatestAnsweredDialOfAContactIsClassified() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	classifyAfter := s.base.Add(5 * time.Minute)
	older := s.answered("a", campaign, contact, s.base, classifyAfter)
	latest := s.answered("a", campaign, contact, s.base.Add(time.Minute), classifyAfter)
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))

	claimed, err := s.store.ClaimClassifications(s.ctx, "a", classifyAfter, time.Minute, 10)

	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Equal(latest.ID, claimed[0].Attempt.ID)
	s.Require().NoError(s.store.Classified(s.ctx, latest.ID, "success", true))
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: ContactCompleted, Attempt: 1, Outcome: "success"}))
	_, attempts, err := s.store.CampaignContact(s.ctx, campaign.ID, contact.ID)
	s.Require().NoError(err)
	for _, attempt := range attempts {
		if attempt.ID == older.ID {
			s.NotEqual("pending", attempt.ClassifyState, "settling the contact leaves no dial of it to classify")
		}
	}
	later, err := s.store.ClaimClassifications(s.ctx, "b", classifyAfter.Add(time.Hour), time.Minute, 10)
	s.Require().NoError(err)
	s.Empty(later)
}

func (s *StoreSuite) TestADialWhoseContactAnotherRunnerHoldsIsNotClassified() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	attempt := s.answered("a", campaign, contact, s.base, s.base.Add(time.Minute))
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	held, err := s.store.ClaimClassifications(s.ctx, "a", s.base.Add(time.Minute), 10*time.Minute, 10)
	s.Require().NoError(err)
	s.Require().Len(held, 1)
	_, err = s.store.DB().NewUpdate().Model((*CampaignAttempt)(nil)).Set("classify_after = ?", s.base.Add(time.Minute)).Where("id = ?", attempt.ID).Exec(s.ctx)
	s.Require().NoError(err)

	taken, err := s.store.ClaimClassifications(s.ctx, "b", s.base.Add(2*time.Minute), time.Minute, 10)
	s.Require().NoError(err)
	s.Empty(taken, "a still holds the contact")

	taken, err = s.store.ClaimClassifications(s.ctx, "b", s.base.Add(12*time.Minute), time.Minute, 10)
	s.Require().NoError(err)
	s.Len(taken, 1, "a's lease ran out")
}

func (s *StoreSuite) TestAContactStoppedMidCallThatFailsKeepsWhyItFailed() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	s.answered("a", campaign, contact, s.base, s.base.Add(time.Minute))
	s.Require().NoError(s.store.StopContact(s.ctx, campaign.ID, contact.ID))
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	_, err := s.store.ClaimClassifications(s.ctx, "a", s.base.Add(time.Minute), time.Minute, 10)
	s.Require().NoError(err)

	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: Failed, Attempt: 1, Reason: "opted_out", Outcome: "do_not_call"}))

	settled := s.contactByNumber(campaign, "+15550001")
	s.Equal(Failed, settled.Stage)
	s.Equal("opted_out", settled.Reason)
}

func (s *StoreSuite) TestALateClassificationChangesNothing() {
	campaign := s.runningCampaign(5, "+15550001")
	contact := s.claimDue("a", s.base, 10)[0].Contact
	older := s.answered("a", campaign, contact, s.base, s.base.Add(time.Minute))
	latest := s.answered("a", campaign, contact, s.base.Add(time.Minute), s.base.Add(time.Minute))
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	_, err := s.store.ClaimClassifications(s.ctx, "a", s.base.Add(time.Minute), time.Minute, 10)
	s.Require().NoError(err)
	s.Require().NoError(s.store.Classified(s.ctx, latest.ID, "success", true))
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", contact.ID, Settle{Stage: ContactCompleted, Attempt: 1}))

	s.NoError(s.store.Classified(s.ctx, latest.ID, "other", false), "a done dial")
	s.NoError(s.store.Classified(s.ctx, older.ID, "other", false), "an abandoned dial")

	_, attempts, err := s.store.CampaignContact(s.ctx, campaign.ID, contact.ID)
	s.Require().NoError(err)
	byID := map[string]CampaignAttempt{}
	for _, attempt := range attempts {
		byID[attempt.ID] = attempt
	}
	s.Equal("success", byID[latest.ID].Outcome)
	s.True(byID[latest.ID].Converted)
	s.Equal("abandoned", byID[older.ID].ClassifyState)
	s.Empty(byID[older.ID].Outcome)
}

func (s *StoreSuite) TestAClassificationPutOffIsReadAgainAtItsTimeByAnyRunner() {
	campaign := s.runningCampaign(1, "+15550001")
	claimed := s.claimDue("a", s.base, 10)[0]
	attempt := s.answered("a", campaign, claimed.Contact, s.base, s.base)
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", claimed.Contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	read, err := s.store.ClaimClassifications(s.ctx, "a", s.base, 20*time.Minute, 10)
	s.Require().NoError(err)
	s.Require().Len(read, 1)

	s.Require().NoError(s.store.PutOffClassification(s.ctx, "a", attempt.ID, s.base.Add(30*time.Second)))

	early, err := s.store.ClaimClassifications(s.ctx, "b", s.base.Add(29*time.Second), 20*time.Minute, 10)
	s.Require().NoError(err)
	s.Empty(early)
	again, err := s.store.ClaimClassifications(s.ctx, "b", s.base.Add(30*time.Second), 20*time.Minute, 10)
	s.Require().NoError(err)
	s.Require().Len(again, 1, "not held off for the rest of the first read's lease")
	s.Equal(2, again[0].Attempt.ClassifyTries)
	s.Equal("b", again[0].Contact.LeasedBy)
	s.Equal(AwaitingOutcome, again[0].Contact.Stage)
}

func (s *StoreSuite) TestOnlyTheRunnerReadingAClassificationPutsItOff() {
	campaign := s.runningCampaign(1, "+15550001")
	claimed := s.claimDue("a", s.base, 10)[0]
	attempt := s.answered("a", campaign, claimed.Contact, s.base, s.base)
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", claimed.Contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	_, err := s.store.ClaimClassifications(s.ctx, "a", s.base, 20*time.Minute, 10)
	s.Require().NoError(err)

	s.ErrorIs(s.store.PutOffClassification(s.ctx, "b", attempt.ID, s.base.Add(30*time.Second)), ErrNotLeaseHolder)

	early, err := s.store.ClaimClassifications(s.ctx, "b", s.base.Add(time.Minute), 20*time.Minute, 10)
	s.Require().NoError(err)
	s.Empty(early, "a still reads it")
}

func (s *StoreSuite) TestAClassificationWhoseContactCannotSettleStaysPending() {
	campaign := s.runningCampaign(1, "+15550001")
	claimed := s.claimDue("a", s.base, 10)[0]
	attempt := s.answered("a", campaign, claimed.Contact, s.base, s.base)
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", claimed.Contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	_, err := s.store.ClaimClassifications(s.ctx, "b", s.base, time.Minute, 10)
	s.Require().NoError(err)

	err = s.store.ClassifyAndSettle(s.ctx, "a", Classification{AttemptID: attempt.ID, Outcome: "success", Converted: true},
		claimed.Contact.ID, Settle{Stage: ContactCompleted, Attempt: 1, Outcome: "success", Converted: true})

	s.ErrorIs(err, ErrNotLeaseHolder)
	latest, _, err := s.store.LatestDial(s.ctx, claimed.Contact.ID)
	s.Require().NoError(err)
	s.Equal("pending", latest.ClassifyState, "read again rather than left done with the contact still waiting")
	s.Empty(latest.Outcome)
	s.Equal(AwaitingOutcome, s.contactByNumber(campaign, "+15550001").Stage)
}

func (s *StoreSuite) TestAClassificationAndItsContactSettleTogether() {
	campaign := s.runningCampaign(1, "+15550001")
	claimed := s.claimDue("a", s.base, 10)[0]
	attempt := s.answered("a", campaign, claimed.Contact, s.base, s.base)
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", claimed.Contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	_, err := s.store.ClaimClassifications(s.ctx, "a", s.base, time.Minute, 10)
	s.Require().NoError(err)

	s.Require().NoError(s.store.ClassifyAndSettle(s.ctx, "a", Classification{AttemptID: attempt.ID, Outcome: "success", Converted: true},
		claimed.Contact.ID, Settle{Stage: ContactCompleted, Attempt: 1, Outcome: "success", Converted: true}))

	latest, _, err := s.store.LatestDial(s.ctx, claimed.Contact.ID)
	s.Require().NoError(err)
	s.Equal("done", latest.ClassifyState)
	s.Equal("success", latest.Outcome)
	s.Equal(ContactCompleted, s.contactByNumber(campaign, "+15550001").Stage)
}

func (s *StoreSuite) TestASilentCallThatCannotSettleStaysAnswered() {
	campaign := s.runningCampaign(1, "+15550001")
	claimed := s.claimDue("a", s.base, 10)[0]
	attempt := s.answered("a", campaign, claimed.Contact, s.base, s.base)
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", claimed.Contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	_, err := s.store.ClaimClassifications(s.ctx, "b", s.base, time.Minute, 10)
	s.Require().NoError(err)
	next := s.base.Add(time.Hour)

	err = s.store.ClassifyAndSettle(s.ctx, "a", Classification{AttemptID: attempt.ID, Outcome: "no_response"},
		claimed.Contact.ID, Settle{Stage: Enrolled, Attempt: 2, NextAttemptAt: &next})

	s.ErrorIs(err, ErrNotLeaseHolder)
	latest, _, err := s.store.LatestDial(s.ctx, claimed.Contact.ID)
	s.Require().NoError(err)
	s.Empty(latest.Outcome)
	s.Equal("pending", latest.ClassifyState)
}

func (s *StoreSuite) TestASilentCallSettlesAsUnanswered() {
	campaign := s.runningCampaign(1, "+15550001")
	claimed := s.claimDue("a", s.base, 10)[0]
	attempt := s.answered("a", campaign, claimed.Contact, s.base, s.base)
	s.Require().NoError(s.store.SettleContact(s.ctx, "a", claimed.Contact.ID, Settle{Stage: AwaitingOutcome, Attempt: 1}))
	_, err := s.store.ClaimClassifications(s.ctx, "a", s.base, time.Minute, 10)
	s.Require().NoError(err)
	next := s.base.Add(time.Hour)

	s.Require().NoError(s.store.ClassifyAndSettle(s.ctx, "a", Classification{AttemptID: attempt.ID, Outcome: "no_response"},
		claimed.Contact.ID, Settle{Stage: Enrolled, Attempt: 2, NextAttemptAt: &next}))

	latest, _, err := s.store.LatestDial(s.ctx, claimed.Contact.ID)
	s.Require().NoError(err)
	s.Equal("completed", latest.CallOutcome, "the phone joined, whatever was said")
	s.Equal("no_response", latest.Outcome)
	s.Equal("done", latest.ClassifyState)
	s.Equal(Enrolled, s.contactByNumber(campaign, "+15550001").Stage)
}
