-- +goose Up

-- A campaign the router paused itself says why: the code of what refused its next call, as
-- telephony_suspended or insufficient_credits, so the customer knows what to fix before
-- starting it again. NULL for a campaign paused by the customer, or never paused. Starting it
-- again clears it. Adding a nullable column rewrites nothing, and a router built before this
-- one never reads it.

-- The ALTER takes an exclusive lock on campaigns, however briefly; behind a long transaction
-- it would queue every campaign read behind it, so it gives up rather than wait. LOCAL keeps
-- the timeout to goose's transaction, off the pooled connection it runs on.
SET LOCAL lock_timeout = '5s';
ALTER TABLE campaigns ADD COLUMN IF NOT EXISTS paused_reason TEXT;

-- +goose Down
SET LOCAL lock_timeout = '5s';
ALTER TABLE campaigns DROP COLUMN IF EXISTS paused_reason;
