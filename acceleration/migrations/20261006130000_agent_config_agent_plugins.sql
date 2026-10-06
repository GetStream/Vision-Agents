-- +goose Up

-- plugins becomes agent_plugins, so that it reads apart from user_plugins: the plugins the
-- app connects once for the agent, against those each end user connects.
ALTER TABLE agent_configs RENAME COLUMN plugins TO agent_plugins;

-- +goose Down
ALTER TABLE agent_configs RENAME COLUMN agent_plugins TO plugins;
