-- +goose Up

-- progressive_tools is whether the agent config's sessions offer plugin, MCP server and
-- connector tools by a one-line summary, and answer the first call to each with its full
-- description and input schema instead of running it (store.AgentConfig.ProgressiveTools).
-- False for every config there is, so tools are offered whole, as before.
ALTER TABLE agent_configs ADD COLUMN progressive_tools BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN progressive_tools;
