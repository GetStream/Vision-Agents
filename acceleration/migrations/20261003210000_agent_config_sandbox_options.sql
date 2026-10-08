-- +goose Up

-- sandbox_options says how an agent's sandbox is built (image, setup commands, size) and
-- how long one run of code in it may take. Empty is the provider's own sandbox.
ALTER TABLE agent_configs ADD COLUMN sandbox_options JSONB NOT NULL DEFAULT '{}';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN sandbox_options;
