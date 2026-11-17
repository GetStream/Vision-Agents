//go:build integration

package store

import (
	"context"
	"encoding/json"
	"time"

	"github.com/uptrace/bun"
)

// draftCampaign creates a draft for a fresh customer with the given numbers added.
func (s *StoreSuite) draftCampaign(customerID string, numbers ...string) Campaign {
	campaign := Campaign{CustomerID: customerID, Name: "Renewals", ConfigID: "config-1", FromNumber: "+15550100",
		Sequence: json.RawMessage(`{"steps":[]}`), CallingHours: json.RawMessage(`{"windows":{}}`)}
	s.Require().NoError(s.store.CreateCampaign(s.ctx, &campaign))
	if len(numbers) > 0 {
		contacts := make([]Contact, len(numbers))
		for i, number := range numbers {
			contacts[i] = Contact{CampaignID: campaign.ID, ToNumber: number}
		}
		_, _, err := s.store.AddContacts(s.ctx, campaign.ID, contacts)
		s.Require().NoError(err)
	}
	return campaign
}

func (s *StoreSuite) launch(campaign Campaign, startAt time.Time) Campaign {
	launched, err := s.store.LaunchCampaign(s.ctx, campaign.CustomerID, campaign.ID, Launch{
		StartAt: startAt, Now: s.base, APIKey: "key", UserID: "user", UsageCategories: []string{"sales"},
	})
	s.Require().NoError(err)
	return launched
}

func (s *StoreSuite) contactsOf(campaign Campaign) []Contact {
	contacts, err := s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{Limit: 1000})
	s.Require().NoError(err)
	return contacts
}

func (s *StoreSuite) TestLaunchingRejectsOptedOutContactsAndEnrollsTheRest() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002", "+15550003", "+15550004")
	s.Require().NoError(s.store.OptOut(s.ctx, &OptOut{CustomerID: campaign.CustomerID, Recipient: "+15550001", Channel: "voice", Source: "test"}))
	s.Require().NoError(s.store.OptOut(s.ctx, &OptOut{CustomerID: campaign.CustomerID, Recipient: "+15550002", Channel: OptOutAll, Source: "test"}))
	s.Require().NoError(s.store.OptOut(s.ctx, &OptOut{CustomerID: campaign.CustomerID, Recipient: "+15550003", Channel: "sms", Source: "test"}))

	launched := s.launch(campaign, s.base)

	s.Equal(Running, launched.State)
	s.Equal(s.base, launched.ConsentAt.UTC())
	s.Equal("key", launched.APIKey)
	s.Equal("user", launched.UserID)
	s.Equal([]string{"sales"}, launched.UsageCategories)
	byNumber := map[string]Contact{}
	for _, contact := range s.contactsOf(campaign) {
		byNumber[contact.ToNumber] = contact
	}
	for _, number := range []string{"+15550001", "+15550002"} {
		s.Equal(Rejected, byNumber[number].Stage, number)
		s.Equal("opted_out", byNumber[number].Reason, number)
		s.Nil(byNumber[number].NextAttemptAt, number)
	}
	for _, number := range []string{"+15550003", "+15550004"} {
		contact := byNumber[number]
		s.Equal(Enrolled, contact.Stage, number)
		s.Equal("step_1", contact.StepID, number)
		s.Zero(contact.Attempt, number)
		s.Require().NotNil(contact.NextAttemptAt, number)
		s.Equal(s.base, contact.NextAttemptAt.UTC(), number)
	}
}

func (s *StoreSuite) TestARevokedOptOutDoesNotRejectAtLaunch() {
	campaign := s.draftCampaign(newID(), "+15550001")
	optOut := OptOut{CustomerID: campaign.CustomerID, Recipient: "+15550001", Channel: "voice", Source: "test"}
	s.Require().NoError(s.store.OptOut(s.ctx, &optOut))
	s.Require().NoError(s.store.RevokeOptOut(s.ctx, campaign.CustomerID, optOut.ID))

	s.launch(campaign, s.base)

	s.Equal(Enrolled, s.contactsOf(campaign)[0].Stage)
}

func (s *StoreSuite) TestLaunchingInTheFutureSchedules() {
	campaign := s.draftCampaign(newID(), "+15550001")
	start := s.base.Add(2 * time.Hour)

	launched := s.launch(campaign, start)

	s.Equal(Scheduled, launched.State)
	s.Require().NotNil(launched.StartAt)
	s.Equal(start, launched.StartAt.UTC())
	s.Require().NotNil(launched.LaunchedAt)
	s.Equal(s.base, launched.LaunchedAt.UTC())
	s.Equal(start, s.contactsOf(campaign)[0].NextAttemptAt.UTC())
}

func (s *StoreSuite) TestAScheduledCampaignCanBePausedAndResumesToScheduled() {
	campaign := s.draftCampaign(newID(), "+15550001")
	s.launch(campaign, time.Now().Add(2*time.Hour))

	s.Require().NoError(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, ""))
	s.Equal(Paused, s.campaignNow(campaign).State)
	s.Require().NoError(s.store.ResumeCampaign(s.ctx, campaign.CustomerID, campaign.ID))

	s.Equal(Scheduled, s.campaignNow(campaign).State, "its start time has not come")
}

func (s *StoreSuite) TestAMissingCampaignIsToldApartFromAReadThatFailed() {
	_, err := s.store.Campaign(s.ctx, newID(), newID())
	s.ErrorIs(err, ErrNoCampaign)

	campaign := s.draftCampaign(newID())
	cancelled, cancel := context.WithCancel(s.ctx)
	cancel()
	_, err = s.store.Campaign(cancelled, campaign.CustomerID, campaign.ID)
	s.Require().Error(err)
	s.NotErrorIs(err, ErrNoCampaign, "a read that failed says nothing about whether the campaign is there")
}

func (s *StoreSuite) TestALaunchedCampaignCannotBeLaunchedAgain() {
	campaign := s.draftCampaign(newID(), "+15550001")
	s.launch(campaign, s.base)

	_, err := s.store.LaunchCampaign(s.ctx, campaign.CustomerID, campaign.ID, Launch{StartAt: s.base, Now: s.base})

	s.ErrorIs(err, ErrCampaignLocked)
}

func (s *StoreSuite) TestALaunchedCampaignOnlyLetsNameNumberAndConcurrencyChange() {
	campaign := s.draftCampaign(newID())
	name, from, concurrency := "Renamed", "+15550199", 5
	updated, err := s.store.UpdateCampaign(s.ctx, campaign.CustomerID, campaign.ID, CampaignChanges{ConfigID: ref("config-2")})
	s.Require().NoError(err, "a draft changes freely")
	s.Equal("config-2", updated.ConfigID)
	s.launch(campaign, s.base)

	updated, err = s.store.UpdateCampaign(s.ctx, campaign.CustomerID, campaign.ID, CampaignChanges{Name: &name, FromNumber: &from, Concurrency: &concurrency})
	s.Require().NoError(err)
	s.Equal("Renamed", updated.Name)
	s.Equal("+15550199", updated.FromNumber)
	s.Equal(5, updated.Concurrency)

	for label, changes := range map[string]CampaignChanges{
		"config":        {ConfigID: ref("config-3")},
		"country":       {DefaultCountry: ref("DE")},
		"sequence":      {Sequence: json.RawMessage(`{"steps":[1]}`)},
		"calling hours": {CallingHours: json.RawMessage(`{"windows":{"monday":[]}}`)},
		"tags":          {Tags: map[string]string{"a": "b"}},
	} {
		_, err := s.store.UpdateCampaign(s.ctx, campaign.CustomerID, campaign.ID, changes)
		s.ErrorIs(err, ErrCampaignLocked, label)
	}
	kept, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	s.Equal("config-2", kept.ConfigID)
}

func (s *StoreSuite) TestOnlyFinishedOrDraftCampaignsCanBeDeleted() {
	customerID := newID()
	for state, deletable := range map[string]bool{Draft: true, Scheduled: false, Running: false, Paused: false, Stopped: true, Completed: true} {
		campaign := s.draftCampaign(customerID)
		_, err := s.store.DB().ExecContext(s.ctx, "UPDATE campaigns SET state = ? WHERE id = ?", state, campaign.ID)
		s.Require().NoError(err)

		err = s.store.DeleteCampaign(s.ctx, customerID, campaign.ID)

		if deletable {
			s.NoError(err, state)
			_, err = s.store.Campaign(s.ctx, customerID, campaign.ID)
			s.Error(err, state)
		} else {
			s.ErrorIs(err, ErrCampaignLocked, state)
		}
	}
}

func (s *StoreSuite) TestDeletingACampaignKeepsItsCalls() {
	campaign := s.draftCampaign(newID(), "+15550001")
	call := Call{ID: newID(), CustomerID: campaign.CustomerID, CampaignID: campaign.ID, StartedAt: s.base}
	_, err := s.store.DB().NewInsert().Model(&call).Exec(s.ctx)
	s.Require().NoError(err)

	s.Require().NoError(s.store.DeleteCampaign(s.ctx, campaign.CustomerID, campaign.ID))

	count, err := s.store.DB().NewSelect().Model((*Call)(nil)).Where("id = ?", call.ID).Count(s.ctx)
	s.Require().NoError(err)
	s.Equal(1, count)
	contacts, err := s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{})
	s.Require().NoError(err)
	s.Empty(contacts, "the contacts go with the campaign")
}

func (s *StoreSuite) TestADuplicateOfNotContactedTakesOnlyContactsNeverDialed() {
	original := s.draftCampaign(newID(), "+15550001", "+15550002")
	_, err := s.store.UpdateCampaign(s.ctx, original.CustomerID, original.ID, CampaignChanges{
		CallingHours: json.RawMessage(`{"windows":{"monday":[{"start":"09:00","end":"17:00"}]}}`), FromNumber: ref("+15550177"),
		DefaultCountry: ref("DE"), Concurrency: ref(7), Tags: map[string]string{"team": "sales"}})
	s.Require().NoError(err)
	original, err = s.store.Campaign(s.ctx, original.CustomerID, original.ID)
	s.Require().NoError(err)
	s.launch(original, s.base)
	dialed := s.contactsOf(original)[0]
	_, err = s.store.DB().NewInsert().Model(&CampaignAttempt{ID: newID(), CampaignID: original.ID, ContactID: dialed.ID, StepID: "step_1", Attempt: 1, Dial: 1, StartedAt: s.base}).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE campaign_contacts SET stage = 'completed', outcome = 'won', converted = true, attempt = 2 WHERE id = ?", dialed.ID)
	s.Require().NoError(err)

	_, err = s.store.DuplicateCampaign(s.ctx, original.CustomerID, original.ID, "not_contacted")
	s.ErrorIs(err, ErrCampaignLocked, "a running campaign has not finished contacting")
	s.Require().NoError(s.store.StopCampaign(s.ctx, original.CustomerID, original.ID, s.base))

	copied, err := s.store.DuplicateCampaign(s.ctx, original.CustomerID, original.ID, "not_contacted")
	s.Require().NoError(err)

	s.Equal("Copy of Renewals", copied.Name)
	s.Equal(Draft, copied.State)
	s.Equal(Standard, copied.Type)
	s.JSONEq(string(original.Sequence), string(copied.Sequence))
	s.Equal("config-1", copied.ConfigID)
	s.JSONEq(string(original.CallingHours), string(copied.CallingHours))
	s.Equal("+15550177", copied.FromNumber)
	s.Equal("DE", copied.DefaultCountry)
	s.Equal(7, copied.Concurrency)
	s.Equal(map[string]string{"team": "sales"}, copied.Tags)
	contacts := s.contactsOf(copied)
	s.Require().Len(contacts, 1)
	s.Equal("+15550002", contacts[0].ToNumber)

	everyone, err := s.store.DuplicateCampaign(s.ctx, original.CustomerID, original.ID, "all")
	s.Require().NoError(err)
	all := s.contactsOf(everyone)
	s.Require().Len(all, 2)
	for _, contact := range all {
		s.Equal(Enrolled, contact.Stage)
		s.Zero(contact.Attempt)
		s.Empty(contact.Outcome)
		s.False(contact.Converted)
		s.Nil(contact.NextAttemptAt)
	}
	nobody, err := s.store.DuplicateCampaign(s.ctx, original.CustomerID, original.ID, "none")
	s.Require().NoError(err)
	s.Empty(s.contactsOf(nobody))
}

func (s *StoreSuite) TestTheSameNumberTwiceIsADuplicate() {
	campaign := s.draftCampaign(newID(), "+15550001")

	added, duplicates, err := s.store.AddContacts(s.ctx, campaign.ID, []Contact{
		{CampaignID: campaign.ID, ToNumber: "+15550002"},
		{CampaignID: campaign.ID, ToNumber: "+15550001"},
		{CampaignID: campaign.ID, ToNumber: "+15550003"},
		{CampaignID: campaign.ID, ToNumber: "+15550003"},
	})

	s.Require().NoError(err)
	s.Equal([]int{1, 3}, duplicates)
	s.Require().Len(added, 2)
	s.Equal("+15550002", added[0].ToNumber)
	s.Equal("+15550003", added[1].ToNumber)
	s.NotEmpty(added[0].ID)
	s.Less(added[0].Seq, added[1].Seq)
	s.Len(s.contactsOf(campaign), 3)
}

func (s *StoreSuite) TestANumberAlreadyMarkedDuplicateByTheMigrationDoesNotBlockAddingIt() {
	campaign := s.draftCampaign(newID(), "+15550001")
	_, err := s.store.DB().ExecContext(s.ctx, `INSERT INTO campaign_contacts (id, campaign_id, to_number, stage, reason) VALUES (?, ?, '+15550001', 'rejected', 'duplicate')`, newID(), campaign.ID)
	s.Require().NoError(err)

	_, duplicates, err := s.store.AddContacts(s.ctx, campaign.ID, []Contact{{CampaignID: campaign.ID, ToNumber: "+15550001"}, {CampaignID: campaign.ID, ToNumber: "+15550009"}})

	s.Require().NoError(err)
	s.Equal([]int{0}, duplicates)
}

func (s *StoreSuite) TestACampaignHoldsAtMostAHundredThousandContacts() {
	campaign := s.draftCampaign(newID())
	// Replica mode skips the row triggers (the data change capture and the campaign foreign
	// key) for these setup rows only, which is most of what 99,999 rows cost: on a busy runner
	// the insert outlasted the driver's 10s read timeout. AddContacts below runs with them.
	err := s.store.DB().RunInTx(s.ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SET LOCAL session_replication_role = replica"); err != nil {
			return err
		}
		_, err := tx.ExecContext(ctx, `INSERT INTO campaign_contacts (id, campaign_id, to_number)
			SELECT 'c' || n, ?, '+1' || n FROM generate_series(1, 99999) AS n`, campaign.ID)
		return err
	})
	s.Require().NoError(err)

	again, duplicates, err := s.store.AddContacts(s.ctx, campaign.ID, []Contact{{ToNumber: "+11"}, {ToNumber: "+12"}})
	s.Require().NoError(err, "numbers already held do not count against the cap")
	s.Empty(again)
	s.Equal([]int{0, 1}, duplicates)

	added, _, err := s.store.AddContacts(s.ctx, campaign.ID, []Contact{{ToNumber: "+19999999999"}})
	s.Require().NoError(err, "the hundred thousandth fits")
	s.Len(added, 1)

	_, _, err = s.store.AddContacts(s.ctx, campaign.ID, []Contact{{ToNumber: "+18888888888"}, {ToNumber: "+11"}})
	s.ErrorIs(err, ErrTooManyContacts)
	count, err := s.store.DB().NewSelect().Model((*Contact)(nil)).Where("campaign_id = ?", campaign.ID).Count(s.ctx)
	s.Require().NoError(err)
	s.Equal(100000, count, "a refused batch stores none of it")
}

func (s *StoreSuite) TestContactsCannotBeAddedAfterLaunch() {
	campaign := s.draftCampaign(newID(), "+15550001")
	s.launch(campaign, s.base)

	_, _, err := s.store.AddContacts(s.ctx, campaign.ID, []Contact{{CampaignID: campaign.ID, ToNumber: "+15550002"}})
	s.ErrorIs(err, ErrCampaignLocked)
	s.ErrorIs(s.store.DeleteContacts(s.ctx, campaign.ID), ErrCampaignLocked)
	s.Len(s.contactsOf(campaign), 1)
}

func (s *StoreSuite) TestDeletingContactsEmptiesADraft() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002")

	s.Require().NoError(s.store.DeleteContacts(s.ctx, campaign.ID))

	s.Empty(s.contactsOf(campaign))
}

func (s *StoreSuite) TestContactsPageInTheOrderTheyWereAdded() {
	campaign := s.draftCampaign(newID(), "+15550003", "+15550001", "+15550002")

	first, err := s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{Limit: 2})
	s.Require().NoError(err)
	s.Require().Len(first, 2)
	s.Equal([]string{"+15550003", "+15550001"}, []string{first[0].ToNumber, first[1].ToNumber})

	rest, err := s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{Limit: 2, After: &first[1].Seq})
	s.Require().NoError(err)
	s.Require().Len(rest, 1)
	s.Equal("+15550002", rest[0].ToNumber)
}

func (s *StoreSuite) TestFilteringByInProgressIncludesCallsAwaitingAnOutcome() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002", "+15550003")
	for number, stage := range map[string]string{"+15550001": InProgress, "+15550002": AwaitingOutcome} {
		_, err := s.store.DB().ExecContext(s.ctx, "UPDATE campaign_contacts SET stage = ? WHERE campaign_id = ? AND to_number = ?", stage, campaign.ID, number)
		s.Require().NoError(err)
	}

	calling, err := s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{Stage: "in_progress"})
	s.Require().NoError(err)
	s.Require().Len(calling, 2)
	s.Equal("+15550001", calling[0].ToNumber)
	s.Equal("+15550002", calling[1].ToNumber)

	waiting, err := s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{Stage: Enrolled})
	s.Require().NoError(err)
	s.Require().Len(waiting, 1)
	s.Equal("+15550003", waiting[0].ToNumber)
}

func (s *StoreSuite) TestContactsAreFoundByPartOfTheirNumber() {
	campaign := s.draftCampaign(newID(), "+15551234567", "+15559876543", "+1555_234")

	found, err := s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{Number: "1234"})
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("+15551234567", found[0].ToNumber)

	found, err = s.store.QueryContacts(s.ctx, campaign.ID, ContactFilter{Number: "_"})
	s.Require().NoError(err)
	s.Require().Len(found, 1, "an underscore is a character, not a wildcard")
}

func (s *StoreSuite) TestStoppingACampaignLeavesWaitingContactsUndialed() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002")
	s.launch(campaign, s.base)

	s.Require().NoError(s.store.StopCampaign(s.ctx, campaign.CustomerID, campaign.ID, s.base.Add(time.Hour)))

	stopped, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	s.Equal(Stopped, stopped.State)
	s.Require().NotNil(stopped.StoppedAt)
	s.Equal(s.base.Add(time.Hour), stopped.StoppedAt.UTC())
	for _, contact := range s.contactsOf(campaign) {
		s.Equal(Enrolled, contact.Stage)
		s.Nil(contact.NextAttemptAt)
	}
	s.ErrorIs(s.store.StopCampaign(s.ctx, campaign.CustomerID, campaign.ID, s.base), ErrCampaignLocked, "stopped is final")
}

func (s *StoreSuite) TestPausingAndResumingFollowTheState() {
	campaign := s.draftCampaign(newID(), "+15550001")
	s.ErrorIs(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, ""), ErrCampaignLocked, "a draft is not running")
	s.launch(campaign, s.base)

	s.Require().NoError(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, "insufficient_credits"))
	paused, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	s.Equal(Paused, paused.State)
	s.Equal("insufficient_credits", paused.PausedReason)
	s.ErrorIs(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, ""), ErrCampaignLocked)

	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE campaigns SET move_paused = true WHERE id = ?", campaign.ID)
	s.Require().NoError(err)
	s.Require().NoError(s.store.ResumeCampaign(s.ctx, campaign.CustomerID, campaign.ID))
	resumed, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	s.Equal(Running, resumed.State)
	s.Empty(resumed.PausedReason)
	var movePaused bool
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT move_paused FROM campaigns WHERE id = ?", campaign.ID).Scan(&movePaused))
	s.False(movePaused)
	s.ErrorIs(s.store.ResumeCampaign(s.ctx, campaign.CustomerID, campaign.ID), ErrCampaignLocked, "only a paused campaign resumes")
}

func (s *StoreSuite) TestResumingBeforeTheStartTimeReturnsToScheduledAndAfterItRuns() {
	for name, tc := range map[string]struct {
		startAt time.Duration
		want    string
	}{"before": {time.Hour, Scheduled}, "after": {-time.Hour, Running}} {
		campaign := s.draftCampaign(newID(), "+15550001")
		_, err := s.store.DB().ExecContext(s.ctx,
			"UPDATE campaigns SET state = 'paused', start_at = ?, started_at = NULL WHERE id = ?", time.Now().Add(tc.startAt), campaign.ID)
		s.Require().NoError(err, name)

		s.Require().NoError(s.store.ResumeCampaign(s.ctx, campaign.CustomerID, campaign.ID), name)

		got, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
		s.Require().NoError(err, name)
		s.Equal(tc.want, got.State, name)
		if tc.want == Running {
			s.NotNil(got.StartedAt, name)
		} else {
			s.Nil(got.StartedAt, name)
		}
	}
}

func (s *StoreSuite) TestStoppingAContactEndsItsWaitOrMarksItsCall() {
	campaign := s.draftCampaign(newID(), "+15550001", "+15550002")
	contacts := s.contactsOf(campaign)
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE campaign_contacts SET stage = 'in_progress' WHERE id = ?", contacts[1].ID)
	s.Require().NoError(err)

	s.Require().NoError(s.store.StopContact(s.ctx, campaign.ID, contacts[0].ID))
	s.Require().NoError(s.store.StopContact(s.ctx, campaign.ID, contacts[1].ID))

	waiting, _, err := s.store.CampaignContact(s.ctx, campaign.ID, contacts[0].ID)
	s.Require().NoError(err)
	s.Equal(ContactStopped, waiting.Stage)
	calling, _, err := s.store.CampaignContact(s.ctx, campaign.ID, contacts[1].ID)
	s.Require().NoError(err)
	s.Equal(InProgress, calling.Stage, "the call goes on until it settles")
	s.Equal("stopped", calling.Reason)
}

func (s *StoreSuite) TestAContactListsItsAttemptsOldestFirst() {
	campaign := s.draftCampaign(newID(), "+15550001")
	contact := s.contactsOf(campaign)[0]
	for i, id := range []string{"b-second", "a-first"} {
		_, err := s.store.DB().NewInsert().Model(&CampaignAttempt{ID: id, CampaignID: campaign.ID, ContactID: contact.ID, StepID: "step_1", Attempt: 2 - i, Dial: 1, StartedAt: s.base.Add(time.Duration(1-i) * time.Hour)}).Exec(s.ctx)
		s.Require().NoError(err)
	}

	_, attempts, err := s.store.CampaignContact(s.ctx, campaign.ID, contact.ID)

	s.Require().NoError(err)
	s.Require().Len(attempts, 2)
	s.Equal("a-first", attempts[0].ID)
	s.Equal("b-second", attempts[1].ID)
}

func (s *StoreSuite) TestCampaignsListNewestFirstByStateAndPage() {
	customerID := newID()
	var made []Campaign
	for i := range 3 {
		campaign := s.draftCampaign(customerID)
		_, err := s.store.DB().ExecContext(s.ctx, "UPDATE campaigns SET created_at = ?, state = ? WHERE id = ?", s.base.Add(time.Duration(i)*time.Hour), []string{Draft, Stopped, Draft}[i], campaign.ID)
		s.Require().NoError(err)
		made = append(made, campaign)
	}

	page, err := s.store.QueryCampaigns(s.ctx, customerID, CampaignFilter{Limit: 2})
	s.Require().NoError(err)
	s.Require().Len(page, 2)
	s.Equal(made[2].ID, page[0].ID)
	s.Equal(made[1].ID, page[1].ID)

	rest, err := s.store.QueryCampaigns(s.ctx, customerID, CampaignFilter{Limit: 2, After: &CreatedPosition{At: page[1].CreatedAt, ID: page[1].ID}})
	s.Require().NoError(err)
	s.Require().Len(rest, 1)
	s.Equal(made[0].ID, rest[0].ID)

	drafts, err := s.store.QueryCampaigns(s.ctx, customerID, CampaignFilter{State: Draft})
	s.Require().NoError(err)
	s.Len(drafts, 2)
}

func (s *StoreSuite) TestAnotherCustomersCampaignIsInvisible() {
	campaign := s.draftCampaign(newID(), "+15550001")
	other := newID()

	_, err := s.store.Campaign(s.ctx, other, campaign.ID)
	s.Error(err)
	listed, err := s.store.QueryCampaigns(s.ctx, other, CampaignFilter{})
	s.Require().NoError(err)
	s.Empty(listed)
	name := "mine now"
	_, err = s.store.UpdateCampaign(s.ctx, other, campaign.ID, CampaignChanges{Name: &name})
	s.Error(err)
	s.NotErrorIs(err, ErrCampaignLocked)
	s.Error(s.store.DeleteCampaign(s.ctx, other, campaign.ID))
	s.launch(campaign, s.base)
	s.ErrorContains(s.store.PauseCampaign(s.ctx, other, campaign.ID, ""), "no campaign")
	s.ErrorContains(s.store.StopCampaign(s.ctx, other, campaign.ID, s.base), "no campaign")
	s.Require().NoError(s.store.PauseCampaign(s.ctx, campaign.CustomerID, campaign.ID, ""))
	s.ErrorContains(s.store.ResumeCampaign(s.ctx, other, campaign.ID), "no campaign")
	_, err = s.store.DuplicateCampaign(s.ctx, other, campaign.ID, "all")
	s.Error(err)
	_, _, err = s.store.CampaignContact(s.ctx, newID(), s.contactsOf(campaign)[0].ID)
	s.Error(err, "a contact is found only through its own campaign")
	kept, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	s.Equal(Paused, kept.State)
}

func (s *StoreSuite) TestAnotherCustomersLaunchFindsNoCampaign() {
	campaign := s.draftCampaign(newID(), "+15550001")

	_, err := s.store.LaunchCampaign(s.ctx, newID(), campaign.ID, Launch{StartAt: s.base, Now: s.base})

	s.ErrorContains(err, "no campaign")
	s.NotErrorIs(err, ErrCampaignLocked)
}

func (s *StoreSuite) TestAnotherCustomersOptOutDoesNotRejectAtLaunch() {
	campaign := s.draftCampaign(newID(), "+15550001")
	s.Require().NoError(s.store.OptOut(s.ctx, &OptOut{CustomerID: newID(), Recipient: "+15550001", Channel: "voice", Source: "test"}))

	s.launch(campaign, s.base)

	s.Equal(Enrolled, s.contactsOf(campaign)[0].Stage)
}

func (s *StoreSuite) TestARunningCampaignAcceptsAnUnchangedLockedFieldButNotAChangedOne() {
	campaign := s.draftCampaign(newID())
	_, err := s.store.UpdateCampaign(s.ctx, campaign.CustomerID, campaign.ID, CampaignChanges{
		Tags: map[string]string{"a": "b"}, DefaultCountry: ref("US")})
	s.Require().NoError(err)
	s.launch(campaign, s.base)

	updated, err := s.store.UpdateCampaign(s.ctx, campaign.CustomerID, campaign.ID, CampaignChanges{
		Name: ref("Renamed"), ConfigID: ref("config-1"), DefaultCountry: ref("US"), Tags: map[string]string{"a": "b"},
		Sequence: json.RawMessage(`{ "steps": [] }`), CallingHours: json.RawMessage(`{"windows":{}}`)})
	s.Require().NoError(err, "the same values in other spacing are no change")
	s.Equal("Renamed", updated.Name)

	for label, changes := range map[string]CampaignChanges{
		"sequence": {Name: ref("Again"), Sequence: json.RawMessage(`{"steps":[1]}`)},
		"tags":     {Tags: map[string]string{"a": "c"}},
		"config":   {ConfigID: ref("config-2")},
	} {
		_, err := s.store.UpdateCampaign(s.ctx, campaign.CustomerID, campaign.ID, changes)
		s.ErrorIs(err, ErrCampaignLocked, label)
	}
	kept, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	s.Equal("Renamed", kept.Name, "a refused update changes nothing")
}

func ref[T any](value T) *T { return &value }

func (s *StoreSuite) TestALaunchByARequestWithNoUsageCategoriesIsStored() {
	campaign := s.draftCampaign(newID(), "+15550001")

	launched, err := s.store.LaunchCampaign(s.ctx, campaign.CustomerID, campaign.ID, Launch{StartAt: s.base, Now: s.base, APIKey: "key"})

	s.Require().NoError(err)
	s.Equal(Running, launched.State)
	s.Empty(launched.UsageCategories)
}
