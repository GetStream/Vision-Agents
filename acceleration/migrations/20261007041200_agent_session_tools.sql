-- +goose Up

-- The tools a session's conversation model was offered, as they were sent, so what a session
-- was offered can be read once the router no longer holds it. A table of its own because a
-- tool list is tens of kilobytes, and listing sessions has no use for it.
CREATE TABLE agent_session_tools (
    session_id TEXT PRIMARY KEY REFERENCES agent_sessions (id) ON DELETE CASCADE,
    tools JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- +goose Down
DROP TABLE agent_session_tools;
