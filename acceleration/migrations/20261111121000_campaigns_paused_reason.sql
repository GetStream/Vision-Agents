-- +goose Up

-- A campaign the router paused itself says why: the code of what refused its next call, as
-- telephony_suspended or insufficient_credits, so the customer knows what to fix before
-- starting it again. NULL for a campaign paused by the customer, or never paused. Starting it
-- again clears it. Adding a nullable column rewrites nothing, and a router built before this
-- one never reads it.

ALTER TABLE campaigns ADD COLUMN IF NOT EXISTS paused_reason TEXT;

-- +goose Down
ALTER TABLE campaigns DROP COLUMN IF EXISTS paused_reason;
