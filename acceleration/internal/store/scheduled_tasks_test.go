//go:build integration

package store

import (
	"context"
	"strings"
	"time"

	"github.com/uptrace/bun"
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
		CustomerID: "acme", ConfigID: "config-1", UserID: "sam", CallerKind: "authenticated",
		Title: "Second", Instructions: "Again.", StartLocal: s.base, Timezone: "UTC",
	}
	err := s.store.CreateScheduledTask(s.ctx, over, 1)
	s.ErrorIs(err, ErrTooManyScheduledTasks)

	// A paused task does not count against the limit.
	s.task("paused", func(task *ScheduledTask) { task.State = ScheduledTaskPaused })
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
		CustomerID: "acme", ConfigID: "config-1", UserID: "sam", CallerKind: "authenticated",
		Title: "Third", Instructions: "Again.", StartLocal: s.base, Timezone: "UTC",
	}, 2))
}

func (s *StoreSuite) TestTheActiveLimitCountsTheTasksOnTheSameConfigOnly() {
	s.task("one", nil)

	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
		CustomerID: "acme", ConfigID: "config-2", UserID: "sam", CallerKind: "authenticated",
		Title: "Elsewhere", Instructions: "Again.", StartLocal: s.base, Timezone: "UTC",
	}, 1), "a task on another config does not count against this one's limit")

	for config, want := range map[string]int{"config-1": 1, "config-2": 1, "config-3": 0} {
		count, err := s.store.ActiveScheduledTaskCount(s.ctx, "acme", "sam", config)
		s.Require().NoError(err)
		s.Equal(want, count, config)
	}
	count, err := s.store.ActiveScheduledTaskCount(s.ctx, "acme", "kim", "config-1")
	s.Require().NoError(err)
	s.Zero(count, "another user's tasks are theirs")
	count, err = s.store.ActiveScheduledTaskCount(s.ctx, "globex", "sam", "config-1")
	s.Require().NoError(err)
	s.Zero(count, "another customer's tasks are theirs")
	s.task("paused", func(task *ScheduledTask) { task.State = ScheduledTaskPaused })
	count, err = s.store.ActiveScheduledTaskCount(s.ctx, "acme", "sam", "config-1")
	s.Require().NoError(err)
	s.Equal(1, count, "a paused task does not count")
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

func (s *StoreSuite) TestARenewalExtendsOnlyTheClaimItHolds() {
	due := s.base.Add(-time.Minute)
	s.task("st-renew", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 10, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	held, err := s.store.RenewScheduledTaskLease(s.ctx, "acme", "st-renew", claimed[0].ClaimID, time.Hour)
	s.Require().NoError(err)
	s.True(held)
	task, err := s.store.ScheduledTask(s.ctx, "acme", "st-renew")
	s.Require().NoError(err)
	s.Require().NotNil(task.LeaseUntil)
	s.WithinDuration(time.Now().Add(time.Hour), *task.LeaseUntil, time.Minute)

	// A pause clears the claim, so the worker that had it holds nothing.
	task.State = ScheduledTaskPaused
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))
	held, err = s.store.RenewScheduledTaskLease(s.ctx, "acme", "st-renew", claimed[0].ClaimID, time.Hour)
	s.Require().NoError(err)
	s.False(held)
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
	// recently it started, because nobody is delivering it. It started before
	// the hold was recorded, as every held run did.
	ended := time.Now().Add(-time.Minute)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", ended).
		Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("postponed_until = ?", ended).
		Set("started_at = ?", ended.Add(-time.Second)).
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

// held claims the due task id and holds its occurrence until until, as quiet
// hours do, returning the claim.
func (s *StoreSuite) held(id string, until time.Time) ClaimedScheduledTask {
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Require().Equal(id, claimed[0].Task.ID)
	held, err := s.store.PostponeScheduledTask(s.ctx, "acme", id, claimed[0].Run.ID, claimed[0].ClaimID, until)
	s.Require().NoError(err)
	s.Require().True(held)
	return claimed[0]
}

// run reads one run back.
func (s *StoreSuite) run(id string) ScheduledTaskRun {
	var run ScheduledTaskRun
	s.Require().NoError(s.store.db.NewSelect().Model(&run).Where("id = ?", id).Scan(s.ctx))
	return run
}

func (s *StoreSuite) TestPausingATaskWritesItsHeldOccurrenceMissed() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	s.task("two", func(task *ScheduledTask) { task.NextRunAt = nil })
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
		ID: "one", CustomerID: "globex", UserID: "sam", Title: "Theirs", Instructions: "Again.",
		StartLocal: s.base, Timezone: "UTC",
	}, 0))
	claim := s.held("one", time.Now().Add(8*time.Hour))
	// What the pause must leave alone: an earlier occurrence already delivered,
	// a run made by hand, and the running runs of another task and of another
	// customer's task under the same id.
	delivered := &ScheduledTaskRun{TaskID: "one", CustomerID: "acme", ScheduledFor: due.Add(-24 * time.Hour), State: ScheduledTaskRunDelivered}
	manual := &ScheduledTaskRun{TaskID: "one", CustomerID: "acme", ScheduledFor: due, Trigger: ScheduledTaskTriggerManual}
	other := &ScheduledTaskRun{TaskID: "two", CustomerID: "acme", ScheduledFor: due}
	theirs := &ScheduledTaskRun{TaskID: "one", CustomerID: "globex", ScheduledFor: due}
	for _, run := range []*ScheduledTaskRun{delivered, manual, other, theirs} {
		s.Require().NoError(s.store.InsertScheduledTaskRun(s.ctx, run))
	}

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.State = ScheduledTaskPaused
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))

	missed := s.run(claim.Run.ID)
	s.Equal(ScheduledTaskRunMissed, missed.State, "the held occurrence is settled, not left running for good")
	s.NotNil(missed.FinishedAt)
	s.Equal(ScheduledTaskRunDelivered, s.run(delivered.ID).State)
	s.Equal(ScheduledTaskRunRunning, s.run(manual.ID).State, "a run by hand records its own outcome")
	s.Equal(ScheduledTaskRunRunning, s.run(other.ID).State)
	s.Equal(ScheduledTaskRunRunning, s.run(theirs.ID).State)
}

func (s *StoreSuite) TestAnEditKeepingTheNextRunSettlesARepeatsOccurrenceRatherThanDeliveringItAgain() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt, task.RRule = &due, "FREQ=DAILY" })
	claim := s.held("one", time.Now().Add(8*time.Hour))

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.Instructions = "Remind Sam to drink water."
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))
	s.True(task.NextRunAt.Equal(due), "a repeat keeps its schedule")

	// The edit cleared the claim and kept the next run, so the task is due
	// again at once: its occurrence is only advanced past.
	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(again, 1)
	s.True(again[0].AlreadySettled, "the old instructions are not delivered")
	s.Equal(claim.Run.ID, again[0].Run.ID)
	s.Equal(ScheduledTaskRunMissed, again[0].Run.State)
}

func (s *StoreSuite) TestAnEditedOneShotWhoseOccurrenceWasHeldComesDueAgainAsItNowReads() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claim := s.held("one", time.Now().Add(8*time.Hour))

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.Instructions = "Remind Sam to drink water."
	before := time.Now()
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))

	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(ScheduledTaskActive, stored.State)
	s.Require().NotNil(stored.NextRunAt)
	s.WithinDuration(before, *stored.NextRunAt, 5*time.Second, "due again at the save")
	s.True(task.NextRunAt.Equal(*stored.NextRunAt), "the caller sees the next run that was saved")
	s.Equal(ScheduledTaskRunMissed, s.run(claim.Run.ID).State, "the held occurrence is settled")

	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(again, 1)
	s.False(again[0].AlreadySettled, "the reminder is delivered, not lost")
	s.NotEqual(claim.Run.ID, again[0].Run.ID)
	s.Equal("Remind Sam to drink water.", again[0].Run.Prompt)
	s.Nil(again[0].Run.PostponedUntil, "the new occurrence is held only if quiet hours hold it when it is delivered")
}

func (s *StoreSuite) TestAPausedOrMovedOneShotIsNotBroughtForwardBySettlingItsHeldOccurrence() {
	due := s.base.Add(-time.Minute)
	s.task("paused", func(task *ScheduledTask) { task.NextRunAt = &due })
	s.held("paused", time.Now().Add(8*time.Hour))
	task, err := s.store.ScheduledTask(s.ctx, "acme", "paused")
	s.Require().NoError(err)
	task.State = ScheduledTaskPaused
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))
	stored, err := s.store.ScheduledTask(s.ctx, "acme", "paused")
	s.Require().NoError(err)
	s.True(stored.NextRunAt.Equal(due), "a pause decides what comes next, not the save")

	s.task("moved", func(task *ScheduledTask) { task.NextRunAt = &due })
	s.held("moved", time.Now().Add(8*time.Hour))
	task, err = s.store.ScheduledTask(s.ctx, "acme", "moved")
	s.Require().NoError(err)
	later := time.Now().Add(3 * time.Hour).UTC().Truncate(time.Second)
	task.NextRunAt = &later
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))
	stored, err = s.store.ScheduledTask(s.ctx, "acme", "moved")
	s.Require().NoError(err)
	s.True(stored.NextRunAt.Equal(later), "a one-shot moved by the edit runs when it was moved to")
}

// lapse ages a run past the stale window and lets its task's lease run out,
// as a worker that died leaves them.
func (s *StoreSuite) lapse(taskID, runID string) {
	_, err := s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("started_at = ?", time.Now().Add(-time.Hour)).
		Where("id = ?", runID).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", time.Now().Add(-time.Minute)).
		Where("id = ?", taskID).Exec(s.ctx)
	s.Require().NoError(err)
}

func (s *StoreSuite) TestAnOccurrenceWhoseWorkersKeepDyingIsFailedAtTheAttemptCap() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	runID := claimed[0].Run.ID

	for attempts := 2; attempts <= maxScheduledTaskRunAttempts; attempts++ {
		s.lapse("one", runID)
		taken, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
		s.Require().NoError(err)
		s.Require().Len(taken, 1)
		s.False(taken[0].AlreadySettled, "attempt %d is still taken over", attempts)
		s.Equal(attempts, taken[0].Run.Attempts)
	}

	s.lapse("one", runID)
	failed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(failed, 1)
	s.True(failed[0].AlreadySettled, "the worker settles the task without delivering again")
	s.Equal(ScheduledTaskRunFailed, failed[0].Run.State)
	stored := s.run(runID)
	s.Equal(ScheduledTaskRunFailed, stored.State)
	s.Equal(3, stored.Attempts, "the first claim and two takeovers")
	s.Require().NotNil(stored.Error)
	s.Contains(*stored.Error, "worker was lost")
	s.NotNil(stored.FinishedAt)
	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(failed[0].ClaimID, task.ClaimID, "the settle runs under a claim of its own")
}

func (s *StoreSuite) TestALeaseLapsingAfterAHoldWasTakenUpIsParkedLikeAnyOther() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claim := s.held("one", time.Now().Add(time.Hour))
	ended := time.Now().Add(-time.Minute)
	_, err := s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", ended).Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("postponed_until = ?", ended).Set("started_at = ?", ended.Add(-time.Second)).
		Where("id = ?", claim.Run.ID).Exec(s.ctx)
	s.Require().NoError(err)
	taken, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(taken, 1, "the hold ended, so the run is taken up at once")

	// The worker that took it up stops renewing a moment later: it may only be
	// slow, so the run waits out the stale window like any other.
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", time.Now().Add(-time.Second)).Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)
	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Empty(again)
	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Require().NotNil(task.LeaseUntil)
	s.WithinDuration(taken[0].Run.StartedAt.Add(scheduledTaskStaleRun), *task.LeaseUntil, time.Second, "parked until the run is stale")
	s.Equal(2, s.run(claim.Run.ID).Attempts)
}

func (s *StoreSuite) TestALastAttemptWhoseLeaseLapsesAMomentIsParkedNotFailed() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	runID := claimed[0].Run.ID
	// Two takeovers ago, and the worker on the third started a moment ago: it
	// may only be slow, so its delivery is not cut by the attempt cap.
	started := time.Now().Add(-time.Minute)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("attempts = ?", maxScheduledTaskRunAttempts).Set("started_at = ?", started).
		Where("id = ?", runID).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", time.Now().Add(-time.Second)).Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)

	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Empty(again)
	stored := s.run(runID)
	s.Equal(ScheduledTaskRunRunning, stored.State)
	s.Equal(maxScheduledTaskRunAttempts, stored.Attempts)
	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Require().NotNil(task.LeaseUntil)
	s.WithinDuration(started.Add(scheduledTaskStaleRun), *task.LeaseUntil, time.Second, "parked until the run is stale")
}

func (s *StoreSuite) TestTheNextWakeIsWhenATaskMayBeClaimedNotWhenItWasDue() {
	at := func(offset time.Duration) *time.Time {
		when := s.base.Add(offset)
		return &when
	}
	lease := func(id string, until *time.Time) {
		_, err := s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
			Set("lease_until = ?", until).Where("id = ?", id).Exec(s.ctx)
		s.Require().NoError(err)
	}
	wake := func() time.Time {
		next, due, err := s.store.NextDueScheduledTask(s.ctx)
		s.Require().NoError(err)
		s.Require().True(due)
		return next
	}
	// Neither a paused task nor one with no next run wakes anybody.
	s.task("paused", func(task *ScheduledTask) { task.NextRunAt, task.State = at(-3*time.Hour), ScheduledTaskPaused })
	s.task("over", func(task *ScheduledTask) { task.NextRunAt = nil })
	lease("over", at(-3*time.Hour))
	s.task("held", func(task *ScheduledTask) { task.NextRunAt = at(-time.Minute) })
	lease("held", at(2*time.Hour))

	s.Equal(s.base.Add(2*time.Hour), wake(), "a held task wakes the worker when its hold ends")

	// A lease that ran out before the next run leaves the next run.
	s.task("lapsed", func(task *ScheduledTask) { task.NextRunAt = at(30 * time.Minute) })
	lease("lapsed", at(-time.Hour))
	s.Equal(s.base.Add(30*time.Minute), wake())
}

// queries keeps the text of every query a pool sends.
type queries struct{ sent []string }

func (*queries) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context { return ctx }
func (q *queries) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	q.sent = append(q.sent, event.Query)
}

// The worker asks when to wake after every look, so the soonest wake is read
// off scheduled_tasks_wake_idx in order, not by sorting every active task.
func (s *StoreSuite) TestTheNextWakeIsReadOffItsIndex() {
	sent := &queries{}
	watched := &Store{db: s.store.db.WithQueryHook(sent)}
	_, _, err := watched.NextDueScheduledTask(s.ctx)
	s.Require().NoError(err)
	s.Require().Len(sent.sent, 1)

	var plan []string
	err = s.store.db.RunInTx(s.ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SET LOCAL enable_seqscan = off"); err != nil {
			return err
		}
		return tx.NewRaw("EXPLAIN "+sent.sent[0]).Scan(ctx, &plan)
	})
	s.Require().NoError(err)
	read := strings.Join(plan, "\n")
	s.Contains(read, "Index Scan using scheduled_tasks_wake_idx", read)
	s.NotContains(read, "Sort", read)
}

func (s *StoreSuite) TestTheLastRunOfEveryTaskIsReadAtOnce() {
	for _, id := range []string{"one", "two", "three", "four"} {
		s.task(id, nil)
	}
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
		ID: "one", CustomerID: "globex", UserID: "sam", Title: "Theirs", Instructions: "Again.",
		StartLocal: s.base, Timezone: "UTC",
	}, 0))
	finished := func(customer, task string, at time.Time) {
		run := &ScheduledTaskRun{TaskID: task, CustomerID: customer, ScheduledFor: at,
			State: ScheduledTaskRunDelivered, FinishedAt: &at}
		s.Require().NoError(s.store.InsertScheduledTaskRun(s.ctx, run))
	}
	finished("acme", "one", s.base.Add(-2*time.Hour))
	finished("acme", "one", s.base.Add(-time.Hour))
	finished("globex", "one", s.base)
	finished("acme", "four", s.base)
	// Two's only run is still going.
	s.Require().NoError(s.store.InsertScheduledTaskRun(s.ctx, &ScheduledTaskRun{TaskID: "two", CustomerID: "acme", ScheduledFor: s.base}))

	last, err := s.store.LastScheduledTaskRunsAt(s.ctx, "acme", []string{"one", "two", "three"})
	s.Require().NoError(err)
	s.Equal(map[string]time.Time{"one": s.base.Add(-time.Hour)}, last,
		"the newest finished run of each task asked for, of this customer only")

	none, err := s.store.LastScheduledTaskRunsAt(s.ctx, "acme", nil)
	s.Require().NoError(err)
	s.Empty(none)
}
