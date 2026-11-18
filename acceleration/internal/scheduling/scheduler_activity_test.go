//go:build integration

package scheduling

import (
	"database/sql"
	"errors"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// occurrencesCounted is a task's app's scheduling row in the activity record.
type occurrencesCounted struct {
	requests, refused int64
	code              string
}

// counting starts a scheduler over deliver, as started does, that counts in an activity record
// of its own, which writes only when the test calls the stop it returns.
func (s *SchedulerSuite) counting(deliver DeliverFunc, tune func(*Scheduler)) (stop func()) {
	record, err := activity.New(activity.Options{Store: s.db, Interval: time.Hour, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	scheduler := s.started(deliver, func(scheduler *Scheduler) {
		scheduler.SetActivity(record)
		if tune != nil {
			tune(scheduler)
		}
	})
	// The scheduler first, so an occurrence it is settling is counted before the record writes.
	return func() {
		scheduler.Close()
		record.Close()
	}
}

// counted is what the activity record holds for task's app under area scheduling and client api.
func (s *SchedulerSuite) counted(task store.ScheduledTask) occurrencesCounted {
	var (
		row  occurrencesCounted
		code sql.NullString
	)
	err := s.db.DB().QueryRowContext(s.ctx, `
SELECT COALESCE(sum(requests), 0), COALESCE(sum(refused), 0), max(last_refusal_code)
FROM customer_activity WHERE app_id = ? AND area = ? AND client = ?`,
		task.CustomerID, activity.AreaScheduling, activity.ClientRouter).Scan(&row.requests, &row.refused, &code)
	s.Require().NoError(err)
	row.code = code.String
	return row
}

// Defect: a delivered occurrence left out of its app's scheduling work, or counted refused.
func (s *SchedulerSuite) TestADeliveredOccurrenceIsCountedOnceUnderItsApp() {
	task := s.task(time.Now().UTC().Add(-time.Hour).Truncate(time.Second), nil)
	stop := s.counting((&deliverer{}).deliver, nil)

	s.Require().Equal(store.ScheduledTaskRunDelivered, s.settled(task)[0].State)
	stop()

	s.Equal(occurrencesCounted{requests: 1}, s.counted(task))
}

// Defect: each attempt counted, so an occurrence tried three times reads as three, or a failed
// one counted as delivered, or refused without the reason its run failed under.
func (s *SchedulerSuite) TestAFailedOccurrenceIsCountedOnceRefusedUnderItsReason() {
	task := s.task(time.Now().UTC().Add(-time.Hour).Truncate(time.Second), nil)
	deliver := &deliverer{err: &Failure{Reason: ReasonNoNumber, Err: errors.New("no number to text")}, only: task.ID}
	stop := s.counting(deliver.deliver, func(scheduler *Scheduler) {
		scheduler.retries = []time.Duration{time.Millisecond, time.Millisecond}
	})

	s.Require().Equal(store.ScheduledTaskRunFailed, s.settled(task)[0].State)
	stop()

	s.Require().Len(deliver.callsFor(task.ID), 3, "the first attempt and both retries")
	s.Equal(occurrencesCounted{requests: 1, refused: 1, code: ReasonNoNumber}, s.counted(task))
}

// Defect: an occurrence too late to deliver counted as delivered, so a router that was down at
// 09:00 shows the reminder as sent.
func (s *SchedulerSuite) TestAMissedOccurrenceIsCountedRefusedAsMissed() {
	deliver := &deliverer{}
	task := s.task(time.Now().UTC().Add(-25*time.Hour).Truncate(time.Second), nil)
	stop := s.counting(deliver.deliver, nil)

	s.Require().Equal(store.ScheduledTaskRunMissed, s.settled(task)[0].State)
	stop()

	s.Require().Empty(deliver.callsFor(task.ID))
	s.Equal(occurrencesCounted{requests: 1, refused: 1, code: store.ScheduledTaskRunMissed}, s.counted(task))
}

// Defect: a quiet-hours hold counted as an occurrence, so a reminder held and then delivered
// reads as two, one of them refused.
func (s *SchedulerSuite) TestAHeldOccurrenceIsCountedOnceWhenItIsDelivered() {
	task := s.task(time.Now().UTC().Add(-time.Minute).Truncate(time.Second), nil)
	deliver := &holdDeliverer{hold: 300 * time.Millisecond, task: task.ID}
	stop := s.counting(deliver.deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	stop()

	s.Require().Len(deliver.calls, 2, "held once, then delivered")
	s.Equal(occurrencesCounted{requests: 1}, s.counted(task))
}

// Defect: a run settled before its advance was lost counted again by the worker that only
// advances the task, so the occurrence reads as two.
func (s *SchedulerSuite) TestAnOccurrenceSettledBeforeItsAdvanceIsNotCountedAgain() {
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
	stop := s.counting((&deliverer{}).deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskCompleted
	}, 10*time.Second, 10*time.Millisecond)
	stop()

	s.Equal(occurrencesCounted{}, s.counted(task))
}

// Defect: a run failed because each of its workers was lost skipped as settled already, though
// no worker that settled it was left to count it.
func (s *SchedulerSuite) TestAnOccurrenceWhoseWorkersKeepDyingIsCountedRefused() {
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	task := s.task(due, nil)
	// Three workers took it up and died: the store's cap.
	started := time.Now().Add(-time.Hour)
	stale := store.ScheduledTaskRun{TaskID: task.ID, CustomerID: task.CustomerID, ScheduledFor: due,
		Attempts: 3, StartedAt: &started}
	s.Require().NoError(s.db.InsertScheduledTaskRun(s.ctx, &stale))
	stop := s.counting((&deliverer{}).deliver, nil)

	s.Require().Eventually(func() bool {
		return s.read(task).State == store.ScheduledTaskFailed
	}, 10*time.Second, 10*time.Millisecond)
	stop()

	s.Equal(occurrencesCounted{requests: 1, refused: 1, code: ReasonFailed}, s.counted(task))
}
