-- +goose Up

-- A plugin login's tokens and a plugin event subscription's signing secret are kept sealed
-- with the deployment's key encryption key, under kek_version, with the row's owner as
-- additional data, as agent_plugin_clients.secret_sealed is. The columns in the clear are
-- written empty from now on: a row stored before keeps its tokens there, and they are read
-- from there until the row is next written, which seals them.
ALTER TABLE agent_plugin_connections
    ADD COLUMN access_token_sealed BYTEA,
    ADD COLUMN refresh_token_sealed BYTEA,
    ADD COLUMN kek_version INT NOT NULL DEFAULT 0;
ALTER TABLE agent_plugin_event_subscriptions
    ADD COLUMN secret_sealed BYTEA,
    ADD COLUMN kek_version INT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE agent_plugin_event_subscriptions
    DROP COLUMN secret_sealed,
    DROP COLUMN kek_version;
ALTER TABLE agent_plugin_connections
    DROP COLUMN access_token_sealed,
    DROP COLUMN refresh_token_sealed,
    DROP COLUMN kek_version;
