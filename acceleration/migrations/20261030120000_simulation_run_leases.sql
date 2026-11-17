-- +goose Up

-- A run is held by the router having its conversations, for as long as that router keeps
-- renewing held_until. One that lapses belongs to a router that is gone, and any other may
-- write it off; until then it is nobody else's, however long ago it started.
--
-- stop_requested is how a run is cancelled from a router that is not holding it: the one
-- that is sees it the next time it renews, and ends the conversations.
ALTER TABLE simulation_runs ADD COLUMN held_until TIMESTAMPTZ;
ALTER TABLE simulation_runs ADD COLUMN stop_requested BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX simulation_runs_held_idx ON simulation_runs (held_until) WHERE state = 'running';

-- +goose Down
DROP INDEX simulation_runs_held_idx;
ALTER TABLE simulation_runs DROP COLUMN stop_requested;
ALTER TABLE simulation_runs DROP COLUMN held_until;
