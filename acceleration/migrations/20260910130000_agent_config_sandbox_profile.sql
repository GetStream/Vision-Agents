-- +goose Up
ALTER TABLE agent_configs ADD COLUMN sandbox_profile TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN sandbox_profile;
