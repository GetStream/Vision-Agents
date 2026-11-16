-- +goose Up
-- querySessions narrows by state and agent_id, each sorted by updated_at like the others.
CREATE INDEX agent_sessions_state_idx ON agent_sessions (customer_id, state, updated_at DESC, id DESC);
CREATE INDEX agent_sessions_agent_id_idx ON agent_sessions (customer_id, agent_id, updated_at DESC, id DESC);

-- +goose Down
DROP INDEX agent_sessions_state_idx;
DROP INDEX agent_sessions_agent_id_idx;
