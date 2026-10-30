-- +goose Up

-- The daily prune (store.PruneScheduledTasks) reads the runs that finished before the cut and
-- the completed or failed tasks last changed before it, in batches. Without these two it scans
-- both tables once per batch. Both tables are small and only staging holds them, so the
-- indexes are built inside the migration's transaction, not CONCURRENTLY.
CREATE INDEX scheduled_task_runs_finished_idx ON scheduled_task_runs (finished_at);

-- The predicate is the prune's own, so the planner can use the index only for that query.
CREATE INDEX scheduled_tasks_pruned_idx ON scheduled_tasks (updated_at)
    WHERE state IN ('completed', 'failed');

-- +goose Down
DROP INDEX scheduled_tasks_pruned_idx;
DROP INDEX scheduled_task_runs_finished_idx;
