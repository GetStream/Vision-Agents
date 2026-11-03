-- +goose Up

-- end_reason is why a session ended: agent_hangup, transferred, hangup, start or
-- unanswered. Empty while it runs, for one its caller stopped, and for every session
-- that ended before it was kept. A constant default rewrites no rows.
ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS end_reason TEXT NOT NULL DEFAULT '';

-- +goose Down

ALTER TABLE agent_sessions DROP COLUMN IF EXISTS end_reason;
