-- +goose Up

-- image is what the agent draws pictures with, as a provider/model or a capability shortcut.
-- Empty, which every config there is starts as, offers no drawing tool: unlike search, a
-- picture costs enough that an agent draws only when its config says so.
ALTER TABLE agent_configs ADD COLUMN image TEXT NOT NULL DEFAULT '';

-- +goose Down

ALTER TABLE agent_configs DROP COLUMN image;
