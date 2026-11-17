-- +goose Up

-- dispatch_incoming_call and dispatch_text say which work the agent leaves to the
-- customer's own dispatch worker rather than answering itself.
ALTER TABLE agent_configs ADD COLUMN dispatch_incoming_call BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE agent_configs ADD COLUMN dispatch_text BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN dispatch_text;
ALTER TABLE agent_configs DROP COLUMN dispatch_incoming_call;
