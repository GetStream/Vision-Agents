-- +goose NO TRANSACTION
-- +goose Up

-- agent_configs.ui_source is agent.yaml's ui key and the two files it named, as the directory
-- last synced from its files had them (store.AgentConfig.UISource): what ui_instructions was
-- compiled from, kept so the directory can be written back. Null, as on every config synced
-- before it, is a config with no UI or one whose UI a client compiled itself. The default
-- rewrites no row; the lock waits at most 5 s behind a running query, past which the migration
-- fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS ui_source JSONB;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs DROP COLUMN IF EXISTS ui_source;
END $$;
-- +goose StatementEnd
