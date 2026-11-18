-- +goose NO TRANSACTION
-- +goose Up

-- agent_configs.ui_instructions is what teaches the agent's model to answer with UI, kept apart
-- from its instructions so a session can leave it out (store.AgentConfig.UIInstructions). Empty,
-- as every config starts, teaches none. agent_sessions.no_ui is a session opened with ui false,
-- which reopening it keeps; null, as on every session made before it, is a session that wrote UI
-- whenever its config taught it to. Neither default rewrites a row; each lock waits at most 5 s
-- behind a running query, past which the migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS ui_instructions TEXT NOT NULL DEFAULT '';
    ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS no_ui BOOLEAN;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions DROP COLUMN IF EXISTS no_ui;
    ALTER TABLE agent_configs DROP COLUMN IF EXISTS ui_instructions;
END $$;
-- +goose StatementEnd
