-- +goose Up

-- chat_channels are the Stream Chat channel types an agent config answers in besides agent
-- channels (store.AgentConfig.ChatChannels), each an object such as {"type": "callouts"}. Empty
-- for every config there is, so each answers where it did. A router of the previous release
-- names its columns when it reads a config and never reads this one. A constant default
-- rewrites no row; the lock waits at most 5 s behind a running query, past which the migration
-- fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS chat_channels JSONB NOT NULL DEFAULT '[]'::jsonb;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs DROP COLUMN IF EXISTS chat_channels;
END $$;
-- +goose StatementEnd
