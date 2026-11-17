-- +goose Up

-- resume_history is how a session is handed back a conversation it carries on
-- (store.ResumeHistory): whether each reply comes with the tool calls it made, and how many
-- turns and characters. An empty document is the router's default, which every existing
-- config and session keeps. Adding a column with a constant default rewrites no rows.
ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS resume_history JSONB NOT NULL DEFAULT '{}';
ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS resume_history JSONB NOT NULL DEFAULT '{}';

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN IF EXISTS resume_history;
ALTER TABLE agent_configs DROP COLUMN IF EXISTS resume_history;
