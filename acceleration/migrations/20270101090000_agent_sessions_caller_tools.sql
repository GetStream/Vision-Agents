-- +goose NO TRANSACTION
-- +goose Up

-- tool_timeout_ms and caller_tools are what the session's caller opened it with: how long its
-- model waits for a tool the caller or a worker runs, and the caller's own tool declarations
-- (definitions, never results). A conversation carried on from its row after it ended in
-- another process gets them back, as one carried on in the process that held it does. Null,
-- as on every session made before them, is the default timeout and no declared tools. Neither
-- column has a default, so adding them rewrites no row; the lock waits at most 5 s behind a
-- running query, past which the migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions
        ADD COLUMN IF NOT EXISTS tool_timeout_ms INTEGER,
        ADD COLUMN IF NOT EXISTS caller_tools JSONB;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions
        DROP COLUMN IF EXISTS caller_tools,
        DROP COLUMN IF EXISTS tool_timeout_ms;
END $$;
-- +goose StatementEnd
