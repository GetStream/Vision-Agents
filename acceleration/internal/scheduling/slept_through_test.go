package scheduling

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// pausedOneShot is a one-shot paused through its time, monday.
func pausedOneShot() store.ScheduledTask {
	due := monday
	return store.ScheduledTask{ID: "task", CustomerID: "customer", Instructions: "Remind them to stretch.",
		State: store.ScheduledTaskPaused, NextRunAt: &due}
}

func TestAPausedOneShotSleepsThroughItsTimeAsAMissedRun(t *testing.T) {
	run := SleptThrough(pausedOneShot())
	require.NotNil(t, run)
	require.Equal(t, "task", run.TaskID)
	require.Equal(t, "customer", run.CustomerID)
	require.Equal(t, monday, run.ScheduledFor)
	require.Equal(t, store.ScheduledTaskTriggerSchedule, run.Trigger)
	require.Equal(t, "Remind them to stretch.", run.Prompt)
	require.Equal(t, store.ScheduledTaskRunMissed, run.State)
	require.Equal(t, ReasonPaused, *run.Error)
	require.NotNil(t, run.FinishedAt, "a missed run is finished, so it is the task's last run")
}

func TestARepeatingTaskSleepsThroughNothing(t *testing.T) {
	task := pausedOneShot()
	task.RRule = "FREQ=DAILY"
	require.Nil(t, SleptThrough(task))
}

func TestAnActiveOneShotSleepsThroughNothing(t *testing.T) {
	task := pausedOneShot()
	task.State = store.ScheduledTaskActive
	require.Nil(t, SleptThrough(task))
}

func TestAOneShotWithNoNextRunSleepsThroughNothing(t *testing.T) {
	task := pausedOneShot()
	task.NextRunAt = nil
	require.Nil(t, SleptThrough(task))
}
