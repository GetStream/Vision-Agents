-- +goose Up
ALTER TABLE skills DROP COLUMN subagent;
ALTER TABLE agent_configs DROP COLUMN subagents;

-- +goose Down
ALTER TABLE agent_configs ADD COLUMN subagents JSONB NOT NULL DEFAULT '{}';
ALTER TABLE skills ADD COLUMN subagent TEXT NOT NULL DEFAULT '';
