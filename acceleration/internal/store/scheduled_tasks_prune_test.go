//go:build integration

package store

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

const day = 24 * time.Hour

// occurrence gives each run a time of its own, so two runs of a task never share an occurrence.
var occurrence atomic.Int64

// aged stores a task in a state and makes its last change age ago.
func (s *StoreSuite) aged(id, customer, state string, age time.Duration) {
	s.task(id, func(task *ScheduledTask) { task.CustomerID, task.State = customer, state })
	_, err := s.store.db.ExecContext(s.ctx,
		"UPDATE scheduled_tasks SET updated_at = now() - make_interval(secs => ?) WHERE customer_id = ? AND id = ?",
		age.Seconds(), customer, id)
	s.Require().NoError(err)
}

// ranFor stores a run of a task in a state, that finished age ago, or has not finished for
// a negative age.
func (s *StoreSuite) ranFor(id, customer, task, trigger, state string, finished time.Duration) {
	var at any
	if finished >= 0 {
		at = time.Now().Add(-finished)
	}
	_, err := s.store.db.ExecContext(s.ctx, `
INSERT INTO scheduled_task_runs (id, task_id, customer_id, scheduled_for, trigger, state, finished_at, created_at)
VALUES (?, ?, ?, now() - interval '400 days' + make_interval(secs => ?), ?, ?, ?, now() - interval '400 days')`,
		id, task, customer, float64(occurrence.Add(1)), trigger, state, at)
	s.Require().NoError(err)
}

func (s *StoreSuite) taskIDs() []string {
	var ids []string
	s.Require().NoError(s.store.db.NewRaw("SELECT customer_id || '/' || id FROM scheduled_tasks ORDER BY 1").Scan(s.ctx, &ids))
	return ids
}

func (s *StoreSuite) runIDs() []string {
	var ids []string
	s.Require().NoError(s.store.db.NewRaw("SELECT id FROM scheduled_task_runs ORDER BY 1").Scan(s.ctx, &ids))
	return ids
}

func (s *StoreSuite) TestPruneDeletesRunsThatFinishedMoreThanNinetyDaysAgo() {
	s.task("t", nil)
	s.ranFor("old-delivered", "acme", "t", ScheduledTaskTriggerSchedule, ScheduledTaskRunDelivered, 91*day)
	s.ranFor("old-failed", "acme", "t", ScheduledTaskTriggerSchedule, ScheduledTaskRunFailed, 91*day)
	s.ranFor("old-missed", "acme", "t", ScheduledTaskTriggerSchedule, ScheduledTaskRunMissed, 91*day)
	s.ranFor("old-manual", "acme", "t", ScheduledTaskTriggerManual, ScheduledTaskRunDelivered, 91*day)
	s.ranFor("recent", "acme", "t", ScheduledTaskTriggerSchedule, ScheduledTaskRunDelivered, 89*day)
	s.ranFor("running", "acme", "t", ScheduledTaskTriggerSchedule, ScheduledTaskRunRunning, -1)

	runs, tasks, err := s.store.PruneScheduledTasks(s.ctx)

	s.Require().NoError(err)
	s.EqualValues(4, runs)
	s.Zero(tasks)
	s.Equal([]string{"recent", "running"}, s.runIDs(),
		"a run counts from when it finished, and one that has not finished is never pruned")
}

func (s *StoreSuite) TestPruneKeepsTheTasksThatAreActiveOrPausedAndTakesTheirOldRuns() {
	s.aged("active", "acme", ScheduledTaskActive, 400*day)
	s.aged("paused", "acme", ScheduledTaskPaused, 400*day)
	s.ranFor("active-old", "acme", "active", ScheduledTaskTriggerSchedule, ScheduledTaskRunDelivered, 200*day)
	s.ranFor("active-new", "acme", "active", ScheduledTaskTriggerSchedule, ScheduledTaskRunDelivered, day)
	s.ranFor("paused-old", "acme", "paused", ScheduledTaskTriggerSchedule, ScheduledTaskRunMissed, 200*day)

	runs, tasks, err := s.store.PruneScheduledTasks(s.ctx)

	s.Require().NoError(err)
	s.EqualValues(2, runs)
	s.Zero(tasks)
	s.Equal([]string{"acme/active", "acme/paused"}, s.taskIDs())
	s.Equal([]string{"active-new"}, s.runIDs())
}

func (s *StoreSuite) TestPruneDeletesCompletedAndFailedTasksPastRetentionWithTheirRuns() {
	s.aged("completed-old", "acme", ScheduledTaskCompleted, 91*day)
	s.aged("failed-old", "globex", ScheduledTaskFailed, 91*day)
	s.aged("completed-new", "acme", ScheduledTaskCompleted, 89*day)
	s.aged("failed-new", "acme", ScheduledTaskFailed, 89*day)
	// Runs a task takes with it are not counted as pruned runs, and a run finished within
	// the retention goes with its task all the same.
	s.ranFor("of-completed-old", "acme", "completed-old", ScheduledTaskTriggerSchedule, ScheduledTaskRunDelivered, 10*day)
	s.ranFor("of-completed-new", "acme", "completed-new", ScheduledTaskTriggerSchedule, ScheduledTaskRunDelivered, 10*day)

	_, tasks, err := s.store.PruneScheduledTasks(s.ctx)

	s.Require().NoError(err)
	s.EqualValues(2, tasks)
	s.Equal([]string{"acme/completed-new", "acme/failed-new"}, s.taskIDs())
	s.Equal([]string{"of-completed-new"}, s.runIDs())
}

func (s *StoreSuite) TestPruneGoesOnUntilNoBatchIsLeft() {
	_, err := s.store.db.ExecContext(s.ctx, fmt.Sprintf(`
INSERT INTO scheduled_tasks (id, customer_id, user_id, title, instructions, start_local, timezone, state, updated_at)
SELECT 'done-' || n, 'acme', 'sam', 't', 'i', now()::timestamp, 'UTC', 'completed', now() - interval '91 days'
FROM generate_series(1, %d) AS n`, scheduledTasksPruned+1))
	s.Require().NoError(err)
	s.task("live", nil)
	_, err = s.store.db.ExecContext(s.ctx, fmt.Sprintf(`
INSERT INTO scheduled_task_runs (id, task_id, customer_id, scheduled_for, trigger, state, finished_at)
SELECT 'run-' || n, 'live', 'acme', now() - make_interval(mins => n), 'manual', 'delivered', now() - interval '91 days'
FROM generate_series(1, %d) AS n`, scheduledTaskRunsPruned+1))
	s.Require().NoError(err)

	runs, tasks, err := s.store.PruneScheduledTasks(s.ctx)

	s.Require().NoError(err)
	s.EqualValues(scheduledTaskRunsPruned+1, runs)
	s.EqualValues(scheduledTasksPruned+1, tasks)
	s.Equal([]string{"acme/live"}, s.taskIDs())
	s.Empty(s.runIDs())
}

// Old finished runs of active tasks are deleted while their workers finish new ones, and old
// finished tasks are deleted meanwhile: a deadlock would fail a finish or the prune.
func (s *StoreSuite) TestPruneRunsBesideFinishesWithoutDeadlock() {
	const live = 12
	due := s.base.Add(-time.Minute)
	for i := range live {
		s.task(fmt.Sprintf("live-%d", i), func(task *ScheduledTask) { task.NextRunAt = &due })
	}
	claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, live, time.Hour)
	s.Require().NoError(err)
	s.Require().Len(claimed, live)
	for _, c := range claimed {
		for j := range 30 {
			s.ranFor(fmt.Sprintf("%s-old-%d", c.Task.ID, j), "acme", c.Task.ID,
				ScheduledTaskTriggerManual, ScheduledTaskRunDelivered, 91*day)
		}
	}
	for i := range 40 {
		id := fmt.Sprintf("done-%d", i)
		s.aged(id, "acme", ScheduledTaskCompleted, 91*day)
		s.ranFor(id+"-run", "acme", id, ScheduledTaskTriggerManual, ScheduledTaskRunDelivered, day)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 2*live+8)
	for _, c := range claimed {
		wg.Go(func() {
			applied, err := s.store.FinishScheduledTaskRun(s.ctx, "acme", c.Run.ID, c.ClaimID, ScheduledTaskOutcome{
				RunState: ScheduledTaskRunDelivered, DeliveredTo: ScheduledTaskDeliveredConversation,
				State: ScheduledTaskActive, NextRunAt: &due,
			})
			if err == nil && !applied {
				err = fmt.Errorf("finish of %s was not applied", c.Task.ID)
			}
			errs <- err
		})
	}
	var runs, tasks int64
	var mu sync.Mutex
	for range 4 {
		wg.Go(func() {
			r, t, err := s.store.PruneScheduledTasks(s.ctx)
			mu.Lock()
			runs, tasks = runs+r, tasks+t
			mu.Unlock()
			errs <- err
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		s.Require().NoError(err)
	}

	s.EqualValues(live*30, runs, "every old run goes once, whichever router's prune takes it")
	s.EqualValues(40, tasks)
	s.Len(s.runIDs(), live, "the finishes of this moment stay")
}
