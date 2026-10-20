//go:build integration

package scheduling

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type SchedulerSuite struct {
	suite.Suite
	ctx context.Context
	db  *store.Store
}

func TestSchedulerSuite(t *testing.T) {
	suite.Run(t, new(SchedulerSuite))
}

func (s *SchedulerSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}
	s.ctx = context.Background()
	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

// deliverer records what it was asked to deliver and answers as it is told.
type deliverer struct {
	mu    sync.Mutex
	calls []store.ClaimedScheduledTask
	err   error
	delay time.Duration
}

func (d *deliverer) deliver(ctx context.Context, claimed store.ClaimedScheduledTask) (string, string, error) {
	if d.delay > 0 {
		select {
		case <-time.After(d.delay):
		case <-ctx.Done():
			return "", "", ctx.Err()
		}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, claimed)
	if d.err != nil {
		return "", "", d.err
	}
	return "conversation", claimed.Task.SessionID, nil
}

// callsFor is the deliveries made for one task, since a worker over a shared
// test database may deliver the tasks other suites left due.
func (d *deliverer) callsFor(taskID string) []store.ClaimedScheduledTask {
	d.mu.Lock()
	defer d.mu.Unlock()
	var calls []store.ClaimedScheduledTask
	for _, call := range d.calls {
		if call.Task.ID == taskID {
			calls = append(calls, call)
		}
	}
	return calls
}

func (d *deliverer) count() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.calls)
}

// started starts a scheduler over a deliverer, tuned as a test says, and stops
// it with the test.
func (s *SchedulerSuite) started(deliver DeliverFunc, tune func(*Scheduler)) *Scheduler {
	scheduler := NewScheduler(s.db, deliver, slog.New(slog.DiscardHandler))
	if tune != nil {
		tune(scheduler)
	}
	scheduler.Start(context.Background())
	s.T().Cleanup(scheduler.Close)
	return scheduler
}

// task stores a task that is due at due, once, for a user of its own, and
// deletes it after the test, so a task one test leaves due is not delivered
// by the next test's worker.
func (s *SchedulerSuite) task(due time.Time, change func(*store.ScheduledTask)) store.ScheduledTask {
	task := store.ScheduledTask{
		CustomerID:   "customer-" + uuid.NewString(),
		ConfigID:     "config-1",
		UserID:       "user-1",
		CallerKind:   store.UserKindAuthenticated,
		Title:        "Stretch",
		Instructions: "Remind them to stretch.",
		StartLocal:   due,
		Timezone:     "UTC",
		NextRunAt:    &due,
		State:        store.ScheduledTaskActive,
		CreatedBy:    "agent",
	}
	if change != nil {
		change(&task)
	}
	s.Require().NoError(s.db.CreateScheduledTask(s.ctx, &task, 25))
	t := s.T()
	t.Cleanup(func() {
		if err := s.db.DeleteScheduledTask(context.Background(), task.CustomerID, task.ID); err != nil {
			t.Errorf("delete the test's task: %v", err)
		}
	})
	return task
}

// read is the task as the store now holds it.
func (s *SchedulerSuite) read(task store.ScheduledTask) store.ScheduledTask {
	stored, err := s.db.ScheduledTask(s.ctx, task.CustomerID, task.ID)
	s.Require().NoError(err)
	return stored
}

// runs is every run the task has, newest first.
func (s *SchedulerSuite) runs(task store.ScheduledTask) []store.ScheduledTaskRun {
	runs, err := s.db.ScheduledTaskRuns(s.ctx, task.CustomerID, task.ID, 25, nil)
	s.Require().NoError(err)
	return runs
}

func (s *SchedulerSuite) TestADueTaskIsDeliveredAndCompleted() {
	deliver := &deliverer{}
	due := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	task := s.task(due, nil)
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(1, deliver.count())
	s.Equal(due, deliver.calls[0].Run.ScheduledFor)
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunDelivered, runs[0].State)
	s.Equal("conversation", runs[0].DeliveredTo)
}

func (s *SchedulerSuite) TestARecurringTaskAdvancesToItsNextOccurrence() {
	deliver := &deliverer{}
	due := time.Now().UTC().Add(-30 * time.Minute).Truncate(time.Second)
	task := s.task(due, func(task *store.ScheduledTask) {
		task.StartLocal = due.AddDate(0, 0, -2)
		task.RRule = "FREQ=DAILY"
	})
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		read := s.read(task)
		return read.State == store.ScheduledTaskActive && read.NextRunAt != nil &&
			read.NextRunAt.After(time.Now())
	}, 10*time.Second, 10*time.Millisecond)
	read := s.read(task)
	s.WithinDuration(due.AddDate(0, 0, 1), *read.NextRunAt, time.Second)
	s.Equal(1, deliver.count())
}

func (s *SchedulerSuite) TestAFailedOccurrenceIsRetriedThenCountedFailed() {
	deliver := &deliverer{err: errors.New("the conversation would not take it")}
	due := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	task := s.task(due, nil)
	s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.retries = []time.Duration{time.Millisecond, time.Millisecond}
	})

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskFailed
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(3, deliver.count(), "the first attempt and both retries")
	s.Equal(1, s.read(task).ConsecutiveFailures)
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunFailed, runs[0].State)
	s.Equal(ReasonFailed, *runs[0].Error, "the error's own words stay in the logs")
}

func (s *SchedulerSuite) TestAFailureIsRecordedUnderItsReasonAndNotItsWords() {
	deliver := &deliverer{err: &Failure{Reason: ReasonNoNumber,
		Err: errors.New("+15550100 has never texted on whatsapp")}}
	task := s.task(time.Now().UTC().Add(-time.Hour).Truncate(time.Second), nil)
	s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.retries = []time.Duration{time.Millisecond, time.Millisecond}
	})

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskFailed
	}, 10*time.Second, 10*time.Millisecond)
	s.Len(deliver.callsFor(task.ID), 3, "a failure that is not final is retried")
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(ReasonNoNumber, *runs[0].Error)
}

func (s *SchedulerSuite) TestAFinalFailureIsNotRetried() {
	deliver := &deliverer{err: &Failure{Reason: ReasonNotAnswered, Final: true,
		Err: errors.New("the call was not answered")}}
	task := s.task(time.Now().UTC().Add(-time.Hour).Truncate(time.Second), nil)
	s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.retries = []time.Duration{time.Millisecond, time.Millisecond}
	})

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskFailed
	}, 10*time.Second, 10*time.Millisecond)
	s.Len(deliver.callsFor(task.ID), 1, "a call nobody answered is not rung again")
	s.Equal(1, s.read(task).ConsecutiveFailures)
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunFailed, runs[0].State)
	s.Equal(ReasonNotAnswered, *runs[0].Error)
}

func (s *SchedulerSuite) TestAnOccurrenceTheDeliveryFoundTooLateIsMissedOnceAndNotCountedFailed() {
	deliver := &deliverer{err: &Missed{Err: errors.New("the call is 20 minutes late")}}
	task := s.task(time.Now().UTC().Add(-20*time.Minute).Truncate(time.Second), func(task *store.ScheduledTask) {
		task.ConsecutiveFailures = 2
	})
	s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.retries = []time.Duration{time.Millisecond, time.Millisecond}
	})

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	s.Len(deliver.callsFor(task.ID), 1, "a missed occurrence is not tried again")
	s.Equal(2, s.read(task).ConsecutiveFailures, "a missed occurrence leaves the streak as it is")
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunMissed, runs[0].State)
	s.Nil(runs[0].Error)
}

func (s *SchedulerSuite) TestARepeatingOccurrenceTheDeliveryFoundTooLateMovesOnToTheNext() {
	deliver := &deliverer{err: &Missed{Err: errors.New("the call is 20 minutes late")}}
	due := time.Now().UTC().Add(-20 * time.Minute).Truncate(time.Second)
	task := s.task(due, func(task *store.ScheduledTask) {
		task.StartLocal = due.AddDate(0, 0, -2)
		task.RRule = "FREQ=DAILY"
		task.ConsecutiveFailures = 2
	})
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		read := s.read(task)
		return read.NextRunAt != nil && read.NextRunAt.After(time.Now())
	}, 10*time.Second, 10*time.Millisecond)
	read := s.read(task)
	s.Equal(store.ScheduledTaskActive, read.State)
	s.Equal(2, read.ConsecutiveFailures, "a missed occurrence leaves the streak as it is")
	s.WithinDuration(due.AddDate(0, 0, 1), *read.NextRunAt, time.Second)
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunMissed, runs[0].State)
}

func (s *SchedulerSuite) TestAScheduleThatDoesNotReadFailsUnderItsReason() {
	deliver := &deliverer{}
	task := s.task(time.Now().UTC().Add(-time.Minute).Truncate(time.Second), func(task *store.ScheduledTask) {
		task.RRule = "FREQ=SOMETIMES"
	})
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskPaused
	}, 10*time.Second, 10*time.Millisecond)
	s.Empty(deliver.callsFor(task.ID))
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(ReasonUnreadable, *runs[0].Error)
}

func (s *SchedulerSuite) TestAFifthFailureInARowFailsTheTask() {
	deliver := &deliverer{err: errors.New("the conversation would not take it")}
	due := time.Now().UTC().Add(-30 * time.Minute).Truncate(time.Second)
	task := s.task(due, func(task *store.ScheduledTask) {
		task.StartLocal = due.AddDate(0, 0, -2)
		task.RRule = "FREQ=DAILY"
		task.ConsecutiveFailures = 4
	})
	s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.retries = []time.Duration{time.Millisecond}
	})

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskFailed
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(5, s.read(task).ConsecutiveFailures)
}

func (s *SchedulerSuite) TestAOneShotMoreThanADayLateIsMissedNotDelivered() {
	deliver := &deliverer{}
	due := time.Now().UTC().Add(-25 * time.Hour).Truncate(time.Second)
	task := s.task(due, nil)
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(0, deliver.count())
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunMissed, runs[0].State)
}

func (s *SchedulerSuite) TestARecurringTaskSkipsOccurrencesOlderThanAnHour() {
	deliver := &deliverer{}
	now := time.Now().UTC().Truncate(time.Second)
	latest := now.Add(-59 * time.Minute)
	task := s.task(latest.Add(-24*time.Hour), func(task *store.ScheduledTask) {
		task.StartLocal = latest.Add(-48 * time.Hour)
		task.RRule = "FREQ=DAILY"
	})
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return len(deliver.callsFor(task.ID)) == 1
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(latest, deliver.callsFor(task.ID)[0].Run.ScheduledFor)
	var runs []store.ScheduledTaskRun
	s.Require().Eventually(func() bool {
		runs = s.runs(task)
		return len(runs) == 2 && runs[0].State != store.ScheduledTaskRunRunning && runs[1].State != store.ScheduledTaskRunRunning
	}, 10*time.Second, 10*time.Millisecond)
	states := []string{runs[0].State, runs[1].State}
	s.Contains(states, store.ScheduledTaskRunMissed)
	s.Contains(states, store.ScheduledTaskRunDelivered)
}

func (s *SchedulerSuite) TestARecurringOccurrenceTwoHoursLateIsMissed() {
	deliver := &deliverer{}
	due := time.Now().UTC().Add(-2 * time.Hour).Truncate(time.Second)
	task := s.task(due, func(task *store.ScheduledTask) {
		task.StartLocal = due.AddDate(0, 0, -2)
		task.RRule = "FREQ=DAILY"
	})
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		runs := s.runs(task)
		return len(runs) == 1 && runs[0].State == store.ScheduledTaskRunMissed
	}, 10*time.Second, 10*time.Millisecond)
	s.Empty(deliver.callsFor(task.ID), "a repeat is not delivered past its hour of grace")
}

// holdDeliverer holds its first delivery for hold, and delivers the rest.
type holdDeliverer struct {
	mu    sync.Mutex
	calls []store.ClaimedScheduledTask
	hold  time.Duration
}

func (d *holdDeliverer) deliver(_ context.Context, claimed store.ClaimedScheduledTask) (string, string, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, claimed)
	if len(d.calls) == 1 {
		return "", "", &Postponed{Until: time.Now().Add(d.hold)}
	}
	return "conversation", claimed.Task.SessionID, nil
}

func (s *SchedulerSuite) TestAHeldTaskIsDeliveredWhenTheHoldEnds() {
	deliver := &holdDeliverer{hold: 300 * time.Millisecond}
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	task := s.task(due, nil)
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	s.Require().Len(deliver.calls, 2, "held once, then delivered")
	s.Equal(deliver.calls[0].Run.ID, deliver.calls[1].Run.ID, "the same run, so the same command id")
	s.Equal(0, s.read(task).ConsecutiveFailures, "a hold is not a failure")
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunDelivered, runs[0].State)
	s.Equal(2, runs[0].Attempts)
	s.Require().NotNil(runs[0].PostponedUntil, "the hold is on the run")
}

func (s *SchedulerSuite) TestAHeldOccurrenceIsCountedFromTheHoldAndNotFromWhenItCameDue() {
	now := time.Now().UTC().Truncate(time.Second)
	scheduler := NewScheduler(s.db, nil, slog.New(slog.DiscardHandler))
	scheduler.now = func() time.Time { return now }
	schedule, err := FromStored(now.Add(-48*time.Hour), "UTC", "FREQ=DAILY", time.Time{})
	s.Require().NoError(err)

	heldUntil := now.Add(-5 * time.Minute)
	claim := store.ClaimedScheduledTask{
		Task: store.ScheduledTask{RRule: "FREQ=DAILY"},
		Run:  store.ScheduledTaskRun{ScheduledFor: now.Add(-25 * time.Hour), PostponedUntil: &heldUntil},
	}
	s.Nil(scheduler.missed(claim, schedule),
		"held overnight and out for five minutes is delivered, not missed")

	claim.Run.PostponedUntil = nil
	s.NotNil(scheduler.missed(claim, schedule),
		"a day old with no hold is missed")
}

func (s *SchedulerSuite) TestTwoWorkersDeliverAnOccurrenceOnce() {
	deliver := &deliverer{delay: 150 * time.Millisecond}
	due := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	task := s.task(due, nil)
	s.started(deliver.deliver, nil)
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(1, deliver.count())
}

func (s *SchedulerSuite) TestAnOccurrenceSettledBeforeItsAdvanceIsNotDeliveredAgain() {
	deliver := &deliverer{}
	due := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	task := s.task(due, nil)
	started := due.Add(-time.Minute)
	run := store.ScheduledTaskRun{
		CustomerID: task.CustomerID, TaskID: task.ID, ScheduledFor: due,
		Trigger: store.ScheduledTaskTriggerSchedule, Prompt: task.Instructions,
		State: store.ScheduledTaskRunDelivered, DeliveredTo: "conversation",
		Attempts: 1, StartedAt: &started, FinishedAt: &due,
	}
	s.Require().NoError(s.db.InsertScheduledTaskRun(s.ctx, &run))
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(0, deliver.count())
	s.Len(s.runs(task), 1)
}

// stalling holds every delivery until its context ends, and records the
// attempts and why the last one ended.
type stalling struct {
	mu       sync.Mutex
	attempts int
	cause    error
}

func (d *stalling) deliver(ctx context.Context, _ store.ClaimedScheduledTask) (string, string, error) {
	d.mu.Lock()
	d.attempts++
	d.mu.Unlock()
	<-ctx.Done()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.cause = context.Cause(ctx)
	return "", "", ctx.Err()
}

func (d *stalling) state() (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.attempts, d.cause
}

// save changes a stored task as an update through the API does.
func (s *SchedulerSuite) save(task store.ScheduledTask, change func(*store.ScheduledTask)) {
	stored := s.read(task)
	change(&stored)
	s.Require().NoError(s.db.SaveScheduledTask(s.ctx, &stored))
}

func (s *SchedulerSuite) TestPausingATaskStopsItsDeliveryInFlight() {
	deliver := &stalling{}
	task := s.task(time.Now().UTC().Add(-time.Minute).Truncate(time.Second), nil)
	scheduler := s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.renewEvery = 20 * time.Millisecond
	})
	s.Require().Eventually(func() bool {
		attempts, _ := deliver.state()
		return attempts == 1
	}, 10*time.Second, 10*time.Millisecond)

	s.save(task, func(task *store.ScheduledTask) { task.State = store.ScheduledTaskPaused })

	s.Require().Eventually(func() bool {
		_, cause := deliver.state()
		return errors.Is(cause, errClaimLost) && scheduler.Running() == 0
	}, 10*time.Second, 10*time.Millisecond)
	attempts, _ := deliver.state()
	s.Equal(1, attempts, "a stopped delivery is not tried again")
	s.Equal(store.ScheduledTaskPaused, s.read(task).State, "the pause decides, not the worker")
	for _, run := range s.runs(task) {
		s.NotEqual(store.ScheduledTaskRunFailed, run.State, "a lost claim leaves the run alone")
	}
}

func (s *SchedulerSuite) TestAnEditedTaskIsNotRetriedByTheWorkerThatLostIt() {
	deliver := &deliverer{err: errors.New("the conversation would not take it")}
	task := s.task(time.Now().UTC().Add(-time.Minute).Truncate(time.Second), nil)
	scheduler := s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.renewEvery = 20 * time.Millisecond
		scheduler.retries = []time.Duration{time.Hour}
	})
	s.Require().Eventually(func() bool {
		return deliver.count() == 1
	}, 10*time.Second, 10*time.Millisecond)

	s.save(task, func(task *store.ScheduledTask) { task.Title = "Stretch twice" })

	s.Require().Eventually(func() bool {
		return scheduler.Running() == 0
	}, 10*time.Second, 10*time.Millisecond, "the hour's wait ended with the claim")
	s.Equal(1, deliver.count())
	read := s.read(task)
	s.Equal(store.ScheduledTaskActive, read.State)
	s.Equal(0, read.ConsecutiveFailures, "a lost claim is not a failure")
	for _, run := range s.runs(task) {
		s.NotEqual(store.ScheduledTaskRunFailed, run.State, "a lost claim leaves the run alone")
	}
}

func (s *SchedulerSuite) TestAnOccurrenceThatOutlastsItsBudgetFails() {
	deliver := &stalling{}
	task := s.task(time.Now().UTC().Add(-time.Minute).Truncate(time.Second), nil)
	s.started(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.budget = 200 * time.Millisecond
	})

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskFailed
	}, 10*time.Second, 10*time.Millisecond)
	_, cause := deliver.state()
	s.ErrorIs(cause, errOutOfTime)
	s.Equal(1, s.read(task).ConsecutiveFailures)
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunFailed, runs[0].State)
	s.Equal(ReasonOutOfTime, *runs[0].Error)
}

func (s *SchedulerSuite) TestAnOccurrenceThatOutlastsItsBudgetIsTimedOutEvenWhenDeliveryAnswersWithACodedFailure() {
	deliver := func(ctx context.Context, _ store.ClaimedScheduledTask) (string, string, error) {
		<-ctx.Done()
		return "", "", &Failure{Reason: ReasonFailed, Err: ctx.Err()}
	}
	task := s.task(time.Now().UTC().Add(-time.Minute).Truncate(time.Second), nil)
	s.started(deliver, func(scheduler *Scheduler) {
		scheduler.budget = 200 * time.Millisecond
	})

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskFailed
	}, 10*time.Second, 10*time.Millisecond)
	runs := s.runs(task)
	s.Require().Len(runs, 1)
	s.Equal(ReasonOutOfTime, *runs[0].Error)
}

func (s *SchedulerSuite) TestAStoppedWorkerClaimsNothingMoreButFinishesItsDelivery() {
	deliver := &deliverer{delay: 500 * time.Millisecond}
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	first := s.task(due, nil)
	ctx, stop := context.WithCancel(s.ctx)
	defer stop()
	scheduler := NewScheduler(s.db, deliver.deliver, slog.New(slog.DiscardHandler))
	scheduler.Start(ctx)
	s.T().Cleanup(scheduler.Close)
	s.Require().Eventually(func() bool {
		return scheduler.Running() == 1
	}, 10*time.Second, 10*time.Millisecond)

	stop()
	second := s.task(due, nil)
	scheduler.Kick()

	s.Require().Eventually(func() bool {
		return s.read(first).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond, "the delivery in flight finishes")
	s.Zero(scheduler.Running())
	s.Never(func() bool {
		return len(s.runs(second)) > 0
	}, 2*poll, 50*time.Millisecond, "nothing new is claimed")
	s.Equal(1, deliver.count())
}

func (s *SchedulerSuite) TestReplayingAMissedRunKeepsTheFailureStreak() {
	deliver := &deliverer{}
	due := time.Now().UTC().Add(-30 * time.Minute).Truncate(time.Second)
	task := s.task(due, func(task *store.ScheduledTask) {
		task.StartLocal = due.AddDate(0, 0, -2)
		task.RRule = "FREQ=DAILY"
		task.ConsecutiveFailures = 3
	})
	run := store.ScheduledTaskRun{
		CustomerID: task.CustomerID, TaskID: task.ID, ScheduledFor: due,
		Trigger: store.ScheduledTaskTriggerSchedule, Prompt: task.Instructions,
		State: store.ScheduledTaskRunMissed, Attempts: 1, FinishedAt: &due,
	}
	s.Require().NoError(s.db.InsertScheduledTaskRun(s.ctx, &run))
	s.started(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		read := s.read(task)
		return read.NextRunAt != nil && read.NextRunAt.After(time.Now())
	}, 10*time.Second, 10*time.Millisecond)
	s.Equal(3, s.read(task).ConsecutiveFailures)
	s.Equal(0, deliver.count())
}
