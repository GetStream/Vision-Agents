-- +goose Up

-- calls is an agent config's call tools: {end_call, press, transfer}, each with
-- its settings, and transfer with its destinations. Empty for every config
-- there is, which is the defaults: transfer and press on, end_call off.
ALTER TABLE agent_configs ADD COLUMN calls JSONB NOT NULL DEFAULT '{}';

-- +goose Down

ALTER TABLE agent_configs DROP COLUMN calls;
