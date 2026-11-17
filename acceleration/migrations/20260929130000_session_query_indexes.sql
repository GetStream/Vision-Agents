-- +goose Up
-- querySessions sorts by updated_at, narrowed by at most the user, the agent or the project.
-- agent_sessions_customer_idx stays on created_at for the usage rollups.
DROP INDEX agent_sessions_user_idx;
DROP INDEX agent_sessions_config_idx;
DROP INDEX agent_sessions_agent_idx;
DROP INDEX agent_sessions_running_idx;

CREATE INDEX agent_sessions_updated_idx ON agent_sessions (customer_id, updated_at DESC, id DESC);
CREATE INDEX agent_sessions_user_idx ON agent_sessions (customer_id, user_id, updated_at DESC, id DESC);
CREATE INDEX agent_sessions_agent_idx ON agent_sessions (customer_id, agent_name, updated_at DESC, id DESC);
CREATE INDEX agent_sessions_project_idx ON agent_sessions (customer_id, project, updated_at DESC, id DESC);

-- +goose Down
DROP INDEX agent_sessions_updated_idx;
DROP INDEX agent_sessions_user_idx;
DROP INDEX agent_sessions_agent_idx;
DROP INDEX agent_sessions_project_idx;

CREATE INDEX agent_sessions_user_idx ON agent_sessions (customer_id, user_id, created_at DESC, id DESC);
CREATE INDEX agent_sessions_config_idx ON agent_sessions (customer_id, config_id, created_at DESC, id DESC);
CREATE INDEX agent_sessions_agent_idx ON agent_sessions (customer_id, agent_name, created_at DESC, id DESC);
CREATE INDEX agent_sessions_running_idx ON agent_sessions (customer_id, created_at DESC, id DESC)
    WHERE closed_at IS NULL;
