-- +goose Up

-- agents are the other text agents of the app a text agent may hand work to, each offered to
-- its model as a tool: [{"config", "tool", "description", "timeout_ms"}]. NULL, which every
-- config there is starts as, offers none.
ALTER TABLE agent_configs ADD COLUMN agents JSONB;

-- +goose Down

ALTER TABLE agent_configs DROP COLUMN agents;
