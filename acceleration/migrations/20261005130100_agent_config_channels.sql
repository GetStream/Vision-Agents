-- +goose Up

-- channels are the lines an agent answers on outside a Stream Chat channel: a WhatsApp
-- number, a number to text, an iMessage line. Each names an account the app connected, and
-- identity says how a sender becomes an end user. Empty is reachable in Chat alone.
ALTER TABLE agent_configs ADD COLUMN channels JSONB NOT NULL DEFAULT '{}';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN channels;
