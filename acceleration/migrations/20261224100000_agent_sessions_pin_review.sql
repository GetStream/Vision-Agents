-- +goose NO TRANSACTION
-- +goose Up

-- pinned_at is when a person pinned the conversation, so a list can put the pinned ones
-- first. review_score is the score the review of the session's call gave it, kept on the
-- session as well so a list filters and sorts on it without joining calls. Both are nullable
-- with no default, which rewrites no row; the lock waits at most 5 s behind a running query,
-- past which the migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS pinned_at TIMESTAMPTZ;
    ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS review_score INTEGER;
END $$;
-- +goose StatementEnd

-- The sessions reviewed before take the score their call was given. A call is recorded under
-- its session's id, in the same database.
UPDATE agent_sessions
SET review_score = calls.review_score
FROM calls
WHERE calls.id = agent_sessions.id
  AND calls.customer_id = agent_sessions.customer_id
  AND calls.review_score IS NOT NULL
  AND agent_sessions.review_score IS NULL;

-- querySessions now sorts by the last message: when the agent last answered, else when the
-- session began. These serve the default list, an end user's own, one project's, the pinned
-- first, and the reviewed ones by score. Built without stopping the writes of sessions.
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_sessions_last_message_idx
    ON agent_sessions (customer_id, (COALESCE(last_response_at, created_at)) DESC, id DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_sessions_user_last_message_idx
    ON agent_sessions (customer_id, user_id, (COALESCE(last_response_at, created_at)) DESC, id DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_sessions_project_last_message_idx
    ON agent_sessions (customer_id, project, (COALESCE(last_response_at, created_at)) DESC, id DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_sessions_pinned_idx
    ON agent_sessions (customer_id, (pinned_at IS NOT NULL) DESC, (COALESCE(last_response_at, created_at)) DESC, id DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_sessions_review_score_idx
    ON agent_sessions (customer_id, review_score, (COALESCE(last_response_at, created_at)) DESC, id DESC)
    WHERE review_score IS NOT NULL;

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS agent_sessions_review_score_idx;
DROP INDEX CONCURRENTLY IF EXISTS agent_sessions_pinned_idx;
DROP INDEX CONCURRENTLY IF EXISTS agent_sessions_project_last_message_idx;
DROP INDEX CONCURRENTLY IF EXISTS agent_sessions_user_last_message_idx;
DROP INDEX CONCURRENTLY IF EXISTS agent_sessions_last_message_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions DROP COLUMN IF EXISTS review_score;
    ALTER TABLE agent_sessions DROP COLUMN IF EXISTS pinned_at;
END $$;
-- +goose StatementEnd
