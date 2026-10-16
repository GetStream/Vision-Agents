-- +goose Up

-- left_out_connectors are the connectors an agent's last sync left out because the app has
-- no connection by the id or label agent.yaml named (AI-1371; store.AgentConfig.
-- LeftOutConnectors): a list of {name, connector_id, connection}. Only a sync writes it.
-- Empty for every config there is. Expand-only: a router built before this one does not
-- name the column, so its inserts take the default and its updates leave it alone.
ALTER TABLE agent_configs ADD COLUMN left_out_connectors JSONB NOT NULL DEFAULT '[]';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN left_out_connectors;
