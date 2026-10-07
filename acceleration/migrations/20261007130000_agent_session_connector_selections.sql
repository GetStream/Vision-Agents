-- +goose Up

-- connector_selections are the connections the caller picked for the agent config's session
-- bindings, as [{"name": alias, "connection_id": id}] (store.SessionConnectorSelection), so a
-- fork of a session that has ended re-resolves them against the config as it is now and the
-- principal asking for the fork (T22, AI-851). References only, never a credential: the
-- credentials stay sealed on connector_connections. The prototype's
-- 20260929200000_session_connector_selections.sql on codex/connector-support at cf62af0d.
ALTER TABLE agent_sessions ADD COLUMN connector_selections JSONB NOT NULL DEFAULT '[]';

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN connector_selections;
