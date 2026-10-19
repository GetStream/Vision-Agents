//go:build integration

package store

import (
	"time"
)

// task stores a scheduled task for acme's user sam, due in an hour, defaulting
// the fields a test does not care about.
func (s *StoreSuite) task(id string, change func(*ScheduledTask)) *ScheduledTask {
	task := &ScheduledTask{
		ID:           id,
		CustomerID:   "acme",
		ConfigID:     "config-1",
		UserID:       "sam",
		CallerKind:   "authenticated",
		Title:        "Stretch",
		Instructions: "Remind Sam to stretch.",
		StartLocal:   s.base,
		Timezone:     "Europe/Amsterdam",
		State:        ScheduledTaskActive,
		CreatedBy:    "api",
	}
	next := s.base.Add(time.Hour)
	task.NextRunAt = &next
	if change != nil {
		change(task)
	}
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, task, 25))
	return task
}

func (s *StoreSuite) TestAScheduledTaskRoundTrips() {
	s.task("one", nil)

	found, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal("Stretch", found.Title)
	s.Equal("Europe/Amsterdam", found.Timezone)
	s.Require().NotNil(found.NextRunAt)
	s.Equal(s.base.Add(time.Hour), *found.NextRunAt)
}

func (s *StoreSuite) TestAScheduledTaskIsOnlyItsOwnCustomers() {
	s.task("one", nil)

	_, err := s.store.ScheduledTask(s.ctx, "globex", "one")
	s.ErrorIs(err, ErrNoScheduledTask)

	tasks, err := s.store.ScheduledTasks(s.ctx, "globex", ScheduledTaskFilter{}, 25, nil)
	s.Require().NoError(err)
	s.Empty(tasks)
}

func (s *StoreSuite) TestACreateUnderATakenIDIsRefused() {
	s.task("one", nil)

	dupe := *s.task("two", nil)
	dupe.ID = "one"
	err := s.store.CreateScheduledTask(s.ctx, &dupe, 25)
	s.ErrorIs(err, ErrScheduledTaskIDTaken)
}

func (s *StoreSuite) TestTheActiveLimitIsEnforced() {
	s.task("one", nil)

	over := &ScheduledTask{
		CustomerID: "acme", UserID: "sam", CallerKind: "authenticated",
		Title: "Second", Instructions: "Again.", StartLocal: s.base, Timezone: "UTC",
	}
	err := s.store.CreateScheduledTask(s.ctx, over, 1)
	s.ErrorIs(err, ErrTooManyScheduledTasks)

	// A paused task does not count against the limit.
	s.task("paused", func(task *ScheduledTask) { task.State = ScheduledTaskPaused })
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
		CustomerID: "acme", UserID: "sam", CallerKind: "authenticated",
		Title: "Third", Instructions: "Again.", StartLocal: s.base, Timezone: "UTC",
	}, 2))
}

func (s *StoreSuite) TestClaimingGivesEachDueTaskToOneWorker() {
	s.task("one", func(task *ScheduledTask) {
		due := s.base.Add(-time.Minute)
		task.NextRunAt = &due
	})
	other, err := Open(s.dsn)
	s.Require().NoError(err)
	defer other.Close()

	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Equal("one", claimed[0].Task.ID)
	s.Equal(ScheduledTaskRunRunning, claimed[0].Run.State)
	s.Equal("Remind Sam to stretch.", claimed[0].Run.Prompt)

	// The lease keeps the second worker out.
	again, err := other.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Empty(again)
}

func (s *StoreSuite) TestAClaimInsertsOneRunPerOccurrence() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })

	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	runs, err := s.store.ScheduledTaskRuns(s.ctx, "acme", "one", 25, nil)
	s.Require().NoError(err)
	s.Require().Len(runs, 1)
	s.Equal(due, runs[0].ScheduledFor)
}

func (s *StoreSuite) TestAStaleRunIsReclaimedUnderTheSameRun() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })

	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	first := claimed[0]

	// The worker dies: its run ages past the stale window and its lease lapses.
	_, err = s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("started_at = ?", s.base.Add(-time.Hour)).
		Where("id = ?", first.Run.ID).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", s.base.Add(-time.Minute)).
		Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)

	reclaimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(reclaimed, 1)
	s.Equal(first.Run.ID, reclaimed[0].Run.ID, "the same run, so the same command id")
	s.Equal(2, reclaimed[0].Run.Attempts)
}

func (s *StoreSuite) TestAFinishUnderAStolenClaimDoesNotAdvanceTheTask() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })

	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	// Somebody edits the task while it is claimed: the claim is fenced.
	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	stored.Title = "Edited"
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &stored))

	next := s.base.Add(24 * time.Hour)
	applied, err := s.store.FinishScheduledTaskRun(s.ctx, "acme", claimed[0].Run.ID, claimed[0].ClaimID, ScheduledTaskOutcome{
		RunState: ScheduledTaskRunDelivered, DeliveredTo: ScheduledTaskDeliveredConversation,
		NextRunAt: &next, State: ScheduledTaskActive,
	})
	s.Require().NoError(err)
	s.False(applied, "a fenced finish must not advance the schedule")

	after, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Require().NotNil(after.NextRunAt)
	s.Equal(due, *after.NextRunAt, "the edited task keeps its own next run")

	// The run is still recorded.
	runs, err := s.store.ScheduledTaskRuns(s.ctx, "acme", "one", 25, nil)
	s.Require().NoError(err)
	s.Require().Len(runs, 1)
	s.Equal(ScheduledTaskRunDelivered, runs[0].State)
}

func (s *StoreSuite) TestAFinishAdvancesTheTask() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })

	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	next := s.base.Add(24 * time.Hour)
	applied, err := s.store.FinishScheduledTaskRun(s.ctx, "acme", claimed[0].Run.ID, claimed[0].ClaimID, ScheduledTaskOutcome{
		RunState: ScheduledTaskRunDelivered, DeliveredTo: ScheduledTaskDeliveredConversation,
		NextRunAt: &next, State: ScheduledTaskActive,
	})
	s.Require().NoError(err)
	s.True(applied)

	after, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Require().NotNil(after.NextRunAt)
	s.Equal(next, *after.NextRunAt)
	s.Empty(after.ClaimID)
}

func (s *StoreSuite) TestDeletingASessionDeletesItsTasks() {
	s.task("one", func(task *ScheduledTask) { task.SessionID = "session-1" })
	s.task("two", func(task *ScheduledTask) { task.SessionID = "session-2" })

	s.Require().NoError(s.store.DeleteScheduledTasksForSession(s.ctx, "acme", "session-1"))

	_, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.ErrorIs(err, ErrNoScheduledTask)
	_, err = s.store.ScheduledTask(s.ctx, "acme", "two")
	s.Require().NoError(err)
}

func (s *StoreSuite) TestDeletingATaskDeletesItsRuns() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	_, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)

	s.Require().NoError(s.store.DeleteScheduledTask(s.ctx, "acme", "one"))

	runs, err := s.store.ScheduledTaskRuns(s.ctx, "acme", "one", 25, nil)
	s.Require().NoError(err)
	s.Empty(runs)
}

func (s *StoreSuite) TestQuietHoursRoundTrip() {
	s.task("one", func(task *ScheduledTask) {
		task.QuietHours = &ScheduledTaskQuietHours{Start: "22:00", End: "07:00"}
	})

	found, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Require().NotNil(found.QuietHours)
	s.Equal("22:00", found.QuietHours.Start)
	s.Equal("07:00", found.QuietHours.End)
}

func (s *StoreSuite) TestAPostponedTaskIsHeldUnderItsClaimUntilTheHoldEnds() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })

	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	until := time.Now().Add(2 * time.Hour)
	held, err := s.store.PostponeScheduledTask(s.ctx, "acme", "one", claimed[0].Run.ID, claimed[0].ClaimID, until)
	s.Require().NoError(err)
	s.True(held)

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Require().NotNil(task.LeaseUntil)
	s.True(task.LeaseUntil.After(time.Now().Add(time.Hour)), "the lease is the hold, not the claim's minute")
	s.Equal(claimed[0].ClaimID, task.ClaimID, "the claim is kept, so no other worker takes the occurrence")

	runs, err := s.store.ScheduledTaskRuns(s.ctx, "acme", "one", 25, nil)
	s.Require().NoError(err)
	s.Require().Len(runs, 1)
	s.Equal(ScheduledTaskRunRunning, runs[0].State, "a held run is still running")
	s.Require().NotNil(runs[0].PostponedUntil)
	s.WithinDuration(until, *runs[0].PostponedUntil, time.Second)

	// Nobody claims it while the hold stands.
	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Empty(again)

	// The hold ends: the run is taken up at once under the same run, however
	// recently it started, because nobody is delivering it.
	ended := time.Now().Add(-time.Minute)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", ended).
		Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("postponed_until = ?", ended).
		Where("id = ?", claimed[0].Run.ID).Exec(s.ctx)
	s.Require().NoError(err)
	taken, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(taken, 1)
	s.Equal(claimed[0].Run.ID, taken[0].Run.ID)
	s.Require().NotNil(taken[0].Run.PostponedUntil, "the claim remembers the hold, so the age counts from its end")
}

func (s *StoreSuite) TestAPostponeUnderAStolenClaimHoldsNothing() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })

	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	stored.Title = "Edited"
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &stored))

	held, err := s.store.PostponeScheduledTask(s.ctx, "acme", "one", claimed[0].Run.ID, claimed[0].ClaimID, time.Now().Add(time.Hour))
	s.Require().NoError(err)
	s.False(held, "a fenced claim holds nothing: the edit decides what happens")
}

func (s *StoreSuite) TestAUserTimezoneIsStoredAndMoved() {
	s.Require().NoError(s.store.SetUserTimezone(s.ctx, "acme", "sam", "authenticated", "Europe/Amsterdam"))
	zone, err := s.store.UserTimezone(s.ctx, "acme", "sam")
	s.Require().NoError(err)
	s.Equal("Europe/Amsterdam", zone)

	// The latest explicit zone wins, and the row's kind is left alone.
	s.Require().NoError(s.store.SetUserTimezone(s.ctx, "acme", "sam", "guest", "Asia/Tokyo"))
	zone, err = s.store.UserTimezone(s.ctx, "acme", "sam")
	s.Require().NoError(err)
	s.Equal("Asia/Tokyo", zone)

	var user User
	s.Require().NoError(s.store.db.NewSelect().Model(&user).
		Where("customer_id = ?", "acme").Where("id = ?", "sam").Scan(s.ctx))
	s.Equal("authenticated", user.Kind, "moving a zone never rewrites what the credential proved")
}
