-- +goose Up

-- move_paused marks a scheduled task or a campaign that a data move (`router replicate`)
-- imported paused because it was active (a campaign: running) on the deployment it came
-- from. The source goes on delivering it until the move is cut over, so the copy waits.
-- `router replicate --resume` resumes exactly the rows marked here, and never one the user
-- paused. Every change the move carries sets it again from the source's state.
--
-- Adding a column with a constant default rewrites nothing. The ALTERs take an exclusive
-- lock, however briefly; behind a long transaction they would queue every read behind them,
-- so they give up rather than wait. LOCAL keeps the timeout to goose's transaction.
SET LOCAL lock_timeout = '5s';
ALTER TABLE scheduled_tasks ADD COLUMN IF NOT EXISTS move_paused BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE campaigns ADD COLUMN IF NOT EXISTS move_paused BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
SET LOCAL lock_timeout = '5s';
ALTER TABLE campaigns DROP COLUMN IF EXISTS move_paused;
ALTER TABLE scheduled_tasks DROP COLUMN IF EXISTS move_paused;
