-- +goose Up

-- Store selected connection IDs so forks from closed sessions can revalidate them against
-- the current agent config and requesting principal. These IDs are references, not secrets.
ALTER TABLE agent_sessions
    ADD COLUMN connector_selections JSONB NOT NULL DEFAULT '[]';

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN connector_selections;
