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
	// A repeat, whose edit keeps its next run (an edited one-shot comes due
	// again at the edit instead).
	s.task("one", func(task *ScheduledTask) { task.NextRunAt, task.RRule = &due, "FREQ=DAILY" })

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
	// A repeat: an edit to a one-shot being delivered leaves it that delivery.
	s.task("one", func(task *ScheduledTask) { task.NextRunAt, task.RRule = &due, "FREQ=DAILY" })

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

func (s *StoreSuite) TestARunTakesOneCallUntilItIsGivenBack() {
	s.task("one", nil)
	run := ScheduledTaskRun{
		TaskID: "one", CustomerID: "acme", ScheduledFor: s.base, Trigger: ScheduledTaskTriggerManual,
		Prompt: "Remind Sam to stretch.", State: ScheduledTaskRunRunning, Attempts: 1, StartedAt: &s.base,
	}
	s.Require().NoError(s.store.InsertScheduledTaskRun(s.ctx, &run))
	take := func(customer, call string) bool {
		took, _, err := s.store.TakeScheduledTaskRunCall(s.ctx, customer, run.ID, call)
		s.Require().NoError(err)
		return took
	}

	s.False(take("globex", "call-0"), "another customer's run is not taken")
	s.True(take("acme", "call-1"))
	s.False(take("acme", "call-2"), "a run that placed a call places no second one")

	s.Require().NoError(s.store.ReleaseScheduledTaskRunCall(s.ctx, "acme", run.ID, "call-2"))
	s.Require().NoError(s.store.ReleaseScheduledTaskRunCall(s.ctx, "globex", run.ID, "call-1"))
	s.False(take("acme", "call-3"), "only the call taken, on the customer's own run, is given back")

	s.Require().NoError(s.store.ReleaseScheduledTaskRunCall(s.ctx, "acme", run.ID, "call-1"))
	s.True(take("acme", "call-3"), "a call given back may be placed again")

	s.Require().NoError(s.store.FinishManualScheduledTaskRun(s.ctx, run.ID, ScheduledTaskRunFailed, "", "call-3", "not_answered"))
	s.Require().NoError(s.store.ReleaseScheduledTaskRunCall(s.ctx, "acme", run.ID, "call-3"))
	runs, err := s.store.ScheduledTaskRuns(s.ctx, "acme", "one", 25, nil)
	s.Require().NoError(err)
	s.Require().Len(runs, 1)
	s.Equal("call-3", runs[0].SessionID, "a finished run keeps the call it made")

	s.Require().NoError(s.store.FinishManualScheduledTaskRun(s.ctx, run.ID, ScheduledTaskRunFailed, "", "", "delivery_failed"))
	s.False(take("acme", "call-4"), "a finished run places no call")
}

func (s *StoreSuite) TestARunsCallKeepsTheReasonItEndedUnderForTheWorkerTakingItOver() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	run := claimed[0].Run
	took, _, err := s.store.TakeScheduledTaskRunCall(s.ctx, "acme", run.ID, "call-1")
	s.Require().NoError(err)
	s.Require().True(took)
	s.Require().NoError(s.store.SetScheduledTaskRunCallReason(s.ctx, "acme", run.ID, "call-2", "delivery_failed"))
	s.Require().NoError(s.store.SetScheduledTaskRunCallReason(s.ctx, "globex", run.ID, "call-1", "delivery_failed"))
	took, reason, err := s.store.TakeScheduledTaskRunCall(s.ctx, "acme", run.ID, "call-3")
	s.Require().NoError(err)
	s.False(took)
	s.Empty(reason, "only the call the run took, on the customer's own run, records how it ended")

	s.Require().NoError(s.store.SetScheduledTaskRunCallReason(s.ctx, "acme", run.ID, "call-1", "not_answered"))
	// The worker dies: its run ages past the stale window and its lease lapses.
	_, err = s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("started_at = ?", s.base.Add(-time.Hour)).
		Where("id = ?", run.ID).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = ?", s.base.Add(-time.Minute)).
		Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)
	reclaimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(reclaimed, 1)
	s.Require().Equal(run.ID, reclaimed[0].Run.ID)

	took, reason, err = s.store.TakeScheduledTaskRunCall(s.ctx, "acme", run.ID, "call-4")
	s.Require().NoError(err)
	s.False(took, "the run rang already")
	s.Equal("not_answered", reason, "the taking over keeps how the call went")
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
	s.False(again[0].WorkersLost, "the run was settled by the edit, not failed by this claim")
}

func (s *StoreSuite) TestAnEditedOneShotWhoseOccurrenceWasHeldComesDueAgainAsItNowReads() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claim := s.held("one", time.Now().Add(8*time.Hour))
	// Not this occurrence's delivery under way: a run made by hand for the
	// same time, and the deliveries of another task and of another
	// customer's task under the same id.
	s.Require().NoError(s.store.InsertScheduledTaskRun(s.ctx, &ScheduledTaskRun{
		TaskID: "one", CustomerID: "acme", ScheduledFor: due, Trigger: ScheduledTaskTriggerManual,
	}))
	s.task("two", func(task *ScheduledTask) { task.NextRunAt = &due })
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
		ID: "one", CustomerID: "globex", UserID: "sam", Title: "Theirs", Instructions: "Again.",
		StartLocal: s.base, Timezone: "UTC", State: ScheduledTaskActive, NextRunAt: &due,
	}, 0))
	others, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(others, 2)

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

func (s *StoreSuite) TestAnEditedOneShotBeingDeliveredKeepsItsClaimAndItsRun() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.Instructions = "Remind Sam to drink water."
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))

	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal("Remind Sam to drink water.", stored.Instructions)
	s.Equal(claimed[0].ClaimID, stored.ClaimID, "the worker delivering it keeps its claim")
	s.True(stored.NextRunAt.Equal(due), "the one-shot is not brought forward to be delivered again")
	s.Equal(ScheduledTaskRunRunning, s.run(claimed[0].Run.ID).State, "the delivery under way is not written missed")
	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Empty(again, "nothing starts a second delivery")

	applied, err := s.store.FinishScheduledTaskRun(s.ctx, "acme", claimed[0].Run.ID, claimed[0].ClaimID, ScheduledTaskOutcome{
		RunState: ScheduledTaskRunDelivered, DeliveredTo: ScheduledTaskDeliveredConversation,
		State: ScheduledTaskCompleted,
	})
	s.Require().NoError(err)
	s.True(applied, "the delivery completes the one-shot")
}

func (s *StoreSuite) TestAnEditedOneShotTakenUpAgainAfterItsHoldKeepsItsClaim() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claim := s.held("one", time.Now().Add(8*time.Hour))
	// The hold ended a moment ago, and a worker takes the occurrence up again.
	_, err := s.store.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("started_at = now() - interval '2 seconds'").
		Set("postponed_until = now() - interval '1 second'").
		Where("id = ?", claim.Run.ID).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = now() - interval '1 second'").
		Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)
	taken, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(taken, 1)
	s.Require().Equal(claim.Run.ID, taken[0].Run.ID)

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.Instructions = "Remind Sam to drink water."
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))

	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(taken[0].ClaimID, stored.ClaimID, "a hold that has ended no longer holds the delivery")
	s.Equal(ScheduledTaskRunRunning, s.run(claim.Run.ID).State)
}

func (s *StoreSuite) TestAnEditedOneShotWhoseWorkerLeaseLapsedComesDueAgain() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = now() - interval '1 second'").
		Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.Instructions = "Remind Sam to drink water."
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))

	s.Equal(ScheduledTaskRunMissed, s.run(claimed[0].Run.ID).State, "nobody holds it: the edit settles it")
	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(again, 1, "the one-shot is due again at the edit")
	s.False(again[0].AlreadySettled)
	s.Equal("Remind Sam to drink water.", again[0].Run.Prompt)
}

func (s *StoreSuite) TestAnEditedOneShotWhoseLostWorkerPlacedACallIsCompletedNotRungAgain() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	took, _, err := s.store.TakeScheduledTaskRunCall(s.ctx, "acme", claimed[0].Run.ID, "call-1")
	s.Require().NoError(err)
	s.Require().True(took)
	_, err = s.store.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = now() - interval '1 second'").
		Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.Instructions = "Remind Sam to drink water."
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))

	s.Equal(ScheduledTaskCompleted, task.State, "the caller reads back what was saved")
	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(ScheduledTaskCompleted, stored.State)
	s.Nil(stored.NextRunAt)
	s.Equal(ScheduledTaskRunMissed, s.run(claimed[0].Run.ID).State)
	again, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Empty(again, "the user is not rung a second time")
}

func (s *StoreSuite) TestAOneShotMovedWhileBeingDeliveredLetsThatDeliveryGo() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	later := time.Now().Add(3 * time.Hour).UTC().Truncate(time.Second)
	task.NextRunAt = &later
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &task))

	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Empty(stored.ClaimID, "the move decides what comes next, not the worker")
	s.True(stored.NextRunAt.Equal(later))
	s.Equal(ScheduledTaskRunMissed, s.run(claimed[0].Run.ID).State)
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
	s.True(failed[0].WorkersLost, "the worker says why")
	s.Equal(ScheduledTaskRunFailed, failed[0].Run.State)
	stored := s.run(runID)
	s.Equal(ScheduledTaskRunFailed, stored.State)
	s.Equal(3, stored.Attempts, "the first claim and two takeovers")
	s.Require().NotNil(stored.Error)
	s.Equal("delivery_failed", *stored.Error, "a code a device is shown, not the story")
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

func (s *StoreSuite) TestDeletingAConfigPausesOnlyItsActiveTasks() {
	created := s.task("one", nil)
	s.task("done", func(task *ScheduledTask) { task.State = ScheduledTaskCompleted })
	s.task("other-config", func(task *ScheduledTask) { task.ConfigID = "config-2" })
	s.task("no-config", func(task *ScheduledTask) { task.ConfigID = "" })
	s.task("other-customer", func(task *ScheduledTask) { task.CustomerID = "globex" })

	s.Require().NoError(s.store.PauseScheduledTasksForConfig(s.ctx, "acme", "config-1"))
	s.Require().NoError(s.store.PauseScheduledTasksForConfig(s.ctx, "acme", ""))

	for id, want := range map[string]string{
		"one":          ScheduledTaskPaused,
		"done":         ScheduledTaskCompleted,
		"other-config": ScheduledTaskActive,
		"no-config":    ScheduledTaskActive,
	} {
		task, err := s.store.ScheduledTask(s.ctx, "acme", id)
		s.Require().NoError(err)
		s.Equal(want, task.State, id)
	}
	paused, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.True(paused.UpdatedAt.After(created.UpdatedAt), "the pause is a change to the task")
	other, err := s.store.ScheduledTask(s.ctx, "globex", "other-customer")
	s.Require().NoError(err)
	s.Equal(ScheduledTaskActive, other.State, "another customer's config of the same id is not this one")
}

func (s *StoreSuite) TestDeletingAConfigWritesItsTasksHeldOccurrenceMissed() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	s.task("elsewhere", func(task *ScheduledTask) { task.ConfigID, task.NextRunAt = "config-2", nil })
	s.task("one", func(task *ScheduledTask) { task.CustomerID, task.NextRunAt = "globex", nil })
	claim := s.held("one", time.Now().Add(8*time.Hour))
	// What the pause must leave alone: an earlier occurrence already delivered,
	// a run made by hand, the running run of a task on another config, and that
	// of another customer's task under the same id and config.
	delivered := &ScheduledTaskRun{TaskID: "one", CustomerID: "acme", ScheduledFor: due.Add(-24 * time.Hour), State: ScheduledTaskRunDelivered}
	manual := &ScheduledTaskRun{TaskID: "one", CustomerID: "acme", ScheduledFor: due, Trigger: ScheduledTaskTriggerManual}
	elsewhere := &ScheduledTaskRun{TaskID: "elsewhere", CustomerID: "acme", ScheduledFor: due}
	theirs := &ScheduledTaskRun{TaskID: "one", CustomerID: "globex", ScheduledFor: due}
	for _, run := range []*ScheduledTaskRun{delivered, manual, elsewhere, theirs} {
		s.Require().NoError(s.store.InsertScheduledTaskRun(s.ctx, run))
	}

	s.Require().NoError(s.store.PauseScheduledTasksForConfig(s.ctx, "acme", "config-1"))

	missed := s.run(claim.Run.ID)
	s.Equal(ScheduledTaskRunMissed, missed.State, "no worker claims a paused task, so the held occurrence is settled now")
	s.NotNil(missed.FinishedAt)
	s.Equal(ScheduledTaskRunDelivered, s.run(delivered.ID).State)
	s.Equal(ScheduledTaskRunRunning, s.run(manual.ID).State, "a run by hand records its own outcome")
	s.Equal(ScheduledTaskRunRunning, s.run(elsewhere.ID).State)
	s.Equal(ScheduledTaskRunRunning, s.run(theirs.ID).State)
	s.Require().NoError(s.store.PauseScheduledTasksForConfig(s.ctx, "acme", "config-1"),
		"a config with no active task left pauses nothing")
}

func (s *StoreSuite) TestATaskOnADeletedConfigIsRefused() {
	deleted := s.boundConfig("acme")
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme", deleted.ID))

	err := s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
		CustomerID: "acme", ConfigID: deleted.ID, UserID: "sam", CallerKind: "authenticated",
		Title: "Stretch", Instructions: "Remind Sam to stretch.", StartLocal: s.base, Timezone: "UTC",
	}, 25)
	s.ErrorIs(err, ErrNoAgentConfig)
	tasks, err := s.store.ScheduledTasks(s.ctx, "acme", ScheduledTaskFilter{}, 25, nil)
	s.Require().NoError(err)
	s.Empty(tasks)

	live := s.boundConfig("acme")
	s.task("live", func(task *ScheduledTask) { task.ConfigID = live.ID })
	s.task("theirs", func(task *ScheduledTask) { task.CustomerID, task.ConfigID = "globex", deleted.ID })
}

func (s *StoreSuite) TestACreateWaitsForAConfigDeleteUnderWayAndIsRefused() {
	config := s.boundConfig("acme")
	deleting, err := s.store.db.BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = deleting.Rollback() }()
	_, err = deleting.ExecContext(s.ctx, "UPDATE agent_configs SET deleted_at = now() WHERE id = ?", config.ID)
	s.Require().NoError(err)

	created := make(chan error, 1)
	go func() {
		created <- s.store.CreateScheduledTask(s.ctx, &ScheduledTask{
			CustomerID: "acme", ConfigID: config.ID, UserID: "sam", CallerKind: "authenticated",
			Title: "Stretch", Instructions: "Remind Sam to stretch.", StartLocal: s.base, Timezone: "UTC",
		}, 25)
	}()
	// The suite's database is its own, so a backend waiting on a lock in it is the create.
	s.Require().Eventually(func() bool {
		var waiting int
		s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT count(*) FROM pg_stat_activity"+
			" WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
		return waiting > 0
	}, 5*time.Second, 10*time.Millisecond, "the create read the config without waiting for the delete under way")
	s.Require().NoError(deleting.Commit())

	s.ErrorIs(<-created, ErrNoAgentConfig)
}

func (s *StoreSuite) TestATaskOfADeletedConfigIsNotMadeActiveAgain() {
	config := s.boundConfig("acme")
	s.task("one", func(task *ScheduledTask) { task.ConfigID, task.State = config.ID, ScheduledTaskPaused })
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme", config.ID))
	stored, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)

	stored.Title = "Renamed"
	s.Require().NoError(s.store.SaveScheduledTask(s.ctx, &stored), "a paused task's edit stands")
	stored.State = ScheduledTaskActive
	s.ErrorIs(s.store.SaveScheduledTask(s.ctx, &stored), ErrNoAgentConfig)

	after, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(ScheduledTaskPaused, after.State)
	s.Equal("Renamed", after.Title)
}

func (s *StoreSuite) TestAClaimInFlightCannotResumeADeletedConfigsTask() {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	s.Require().NoError(s.store.PauseScheduledTasksForConfig(s.ctx, "acme", "config-1"))

	next := s.base.Add(24 * time.Hour)
	applied, err := s.store.FinishScheduledTaskRun(s.ctx, "acme", claimed[0].Run.ID, claimed[0].ClaimID, ScheduledTaskOutcome{
		RunState: ScheduledTaskRunDelivered, DeliveredTo: ScheduledTaskDeliveredConversation,
		NextRunAt: &next, State: ScheduledTaskActive,
	})
	s.Require().NoError(err)
	s.False(applied, "the pause fenced the claim")
	after, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(ScheduledTaskPaused, after.State)
}

// holdRun claims the due task one and locks its run's row in a transaction
// left open, so a write of that run waits until release is called.
func (s *StoreSuite) holdRun() (ClaimedScheduledTask, func()) {
	due := s.base.Add(-time.Minute)
	s.task("one", func(task *ScheduledTask) { task.NextRunAt = &due })
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, 16, time.Minute)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	holder, err := s.store.db.BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	_, err = holder.ExecContext(s.ctx, "SELECT id FROM scheduled_task_runs WHERE id = ? FOR UPDATE", claimed[0].Run.ID)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = holder.Rollback() })
	return claimed[0], func() { s.Require().NoError(holder.Rollback()) }
}

func (s *StoreSuite) TestAConfigDeleteWhileAWorkerFinishesTheRunStillPausesTheTask() {
	claim, release := s.holdRun()
	finished := make(chan error, 1)
	go func() {
		next := s.base.Add(24 * time.Hour)
		_, err := s.store.FinishScheduledTaskRun(s.ctx, "acme", claim.Run.ID, claim.ClaimID, ScheduledTaskOutcome{
			RunState: ScheduledTaskRunDelivered, DeliveredTo: ScheduledTaskDeliveredConversation,
			NextRunAt: &next, State: ScheduledTaskActive,
		})
		finished <- err
	}()
	s.waitForLockWaiters(1, "the write of the held run waits")
	paused := make(chan error, 1)
	go func() { paused <- s.store.PauseScheduledTasksForConfig(s.ctx, "acme", "config-1") }()
	s.waitForLockWaiters(2, "the pause waits too")
	release()

	s.Require().NoError(<-finished)
	s.Require().NoError(<-paused, "the pause is not aborted as a deadlock with the finish")
	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(ScheduledTaskPaused, task.State, "the deleted config's task is not left active")
	s.Equal(ScheduledTaskRunDelivered, s.run(claim.Run.ID).State, "the finish recorded the delivery")
}

func (s *StoreSuite) TestAConfigDeleteWhileATaskIsPausedByHandStillPausesTheTask() {
	claim, release := s.holdRun()
	task, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	task.State = ScheduledTaskPaused
	saved := make(chan error, 1)
	go func() { saved <- s.store.SaveScheduledTask(s.ctx, &task) }()
	s.waitForLockWaiters(1, "the write of the held run waits")
	paused := make(chan error, 1)
	go func() { paused <- s.store.PauseScheduledTasksForConfig(s.ctx, "acme", "config-1") }()
	s.waitForLockWaiters(2, "the pause waits too")
	release()

	s.Require().NoError(<-saved)
	s.Require().NoError(<-paused, "the pause is not aborted as a deadlock with the save")
	after, err := s.store.ScheduledTask(s.ctx, "acme", "one")
	s.Require().NoError(err)
	s.Equal(ScheduledTaskPaused, after.State)
	s.Equal(ScheduledTaskRunMissed, s.run(claim.Run.ID).State, "the pause by hand settled the occurrence")
}

func (s *StoreSuite) TestAnUnchangedUserTimezoneIsNotWrittenAgain() {
	s.Require().NoError(s.store.SetUserTimezone(s.ctx, "acme", "sam", "authenticated", "Europe/Amsterdam"))
	written := s.userRowVersion("sam")

	s.Require().NoError(s.store.SetUserTimezone(s.ctx, "acme", "sam", "authenticated", "Europe/Amsterdam"))
	s.Equal(written, s.userRowVersion("sam"), "the same zone again leaves the row alone")

	s.Require().NoError(s.store.SetUserTimezone(s.ctx, "acme", "sam", "authenticated", "Asia/Tokyo"))
	s.NotEqual(written, s.userRowVersion("sam"))
	zone, err := s.store.UserTimezone(s.ctx, "acme", "sam")
	s.Require().NoError(err)
	s.Equal("Asia/Tokyo", zone)
}

// userRowVersion is the transaction that last wrote acme's user's row, which moves on
// every write to it, even one that sets a column to what it held.
func (s *StoreSuite) userRowVersion(userID string) string {
	var version string
	s.Require().NoError(s.store.db.QueryRowContext(s.ctx,
		"SELECT xmin::text FROM users WHERE customer_id = ? AND id = ?", "acme", userID).Scan(&version))
	return version
}

func (s *StoreSuite) TestAUserWithNoZoneYetIsGivenOne() {
	// A row written before zones were kept holds none, not an empty one.
	_, err := s.store.db.ExecContext(s.ctx,
		"INSERT INTO users (customer_id, id, kind, name, timezone) VALUES ('acme', 'sam', 'authenticated', '', NULL)")
	s.Require().NoError(err)

	s.Require().NoError(s.store.SetUserTimezone(s.ctx, "acme", "sam", "authenticated", "Europe/Amsterdam"))

	zone, err := s.store.UserTimezone(s.ctx, "acme", "sam")
	s.Require().NoError(err)
	s.Equal("Europe/Amsterdam", zone)
}
