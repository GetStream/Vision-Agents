-- +goose Up

-- A worker sleeps until the soonest an active task may be claimed: its next run,
-- or the end of the lease or quiet-hours hold on it when that is later
-- (NextDueScheduledTask). GREATEST skips a null lease_until.
CREATE INDEX scheduled_tasks_wake_idx ON scheduled_tasks ((GREATEST(next_run_at, lease_until)))
    WHERE state = 'active' AND next_run_at IS NOT NULL;

-- +goose Down
DROP INDEX scheduled_tasks_wake_idx;
