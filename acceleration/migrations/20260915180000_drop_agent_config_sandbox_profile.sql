-- The backend no longer holds research workspaces. An agent that reads source owns its own
-- Daytona VM and offers investigate_sdk as one of its own tools, so there is no profile for
-- a config to name.

-- +goose Up
ALTER TABLE agent_configs DROP COLUMN sandbox_profile;

-- +goose Down
ALTER TABLE agent_configs ADD COLUMN sandbox_profile TEXT NOT NULL DEFAULT '';
