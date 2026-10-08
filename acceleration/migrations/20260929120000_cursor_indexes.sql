-- +goose Up
-- Lists page by cursor, which seeks on every sort key, so each index ends in the tie-break.
DROP INDEX agent_sessions_customer_idx;
DROP INDEX agent_sessions_user_idx;
DROP INDEX agent_sessions_config_idx;
DROP INDEX agent_sessions_agent_idx;
DROP INDEX agent_sessions_running_idx;
DROP INDEX agent_responses_session_idx;
DROP INDEX agent_response_items_session_idx;

CREATE INDEX agent_sessions_customer_idx ON agent_sessions (customer_id, created_at DESC, id DESC);
CREATE INDEX agent_sessions_user_idx ON agent_sessions (customer_id, user_id, created_at DESC, id DESC);
CREATE INDEX agent_sessions_config_idx ON agent_sessions (customer_id, config_id, created_at DESC, id DESC);
CREATE INDEX agent_sessions_agent_idx ON agent_sessions (customer_id, agent_name, created_at DESC, id DESC);
CREATE INDEX agent_sessions_running_idx ON agent_sessions (customer_id, created_at DESC, id DESC)
    WHERE closed_at IS NULL;
CREATE INDEX agent_responses_session_idx ON agent_responses (session_id, created_at ASC, id ASC);
CREATE INDEX agent_response_items_session_idx
    ON agent_response_items (session_id, at ASC, response_id ASC, ordinal ASC);

-- +goose Down
DROP INDEX agent_sessions_customer_idx;
DROP INDEX agent_sessions_user_idx;
DROP INDEX agent_sessions_config_idx;
DROP INDEX agent_sessions_agent_idx;
DROP INDEX agent_sessions_running_idx;
DROP INDEX agent_responses_session_idx;
DROP INDEX agent_response_items_session_idx;

CREATE INDEX agent_sessions_customer_idx ON agent_sessions (customer_id, created_at DESC);
CREATE INDEX agent_sessions_user_idx ON agent_sessions (customer_id, user_id, created_at DESC);
CREATE INDEX agent_sessions_config_idx ON agent_sessions (customer_id, config_id, created_at DESC);
CREATE INDEX agent_sessions_agent_idx ON agent_sessions (customer_id, agent_name, created_at DESC);
CREATE INDEX agent_sessions_running_idx ON agent_sessions (customer_id, created_at DESC)
    WHERE closed_at IS NULL;
CREATE INDEX agent_responses_session_idx ON agent_responses (session_id, created_at ASC);
CREATE INDEX agent_response_items_session_idx ON agent_response_items (session_id, at ASC);
