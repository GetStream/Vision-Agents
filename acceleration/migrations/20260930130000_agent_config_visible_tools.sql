-- +goose Up

-- visible_tools names the tools whose steps end users see on a persistent conversation's
-- replies, as names or path.Match patterns. Empty, which every existing config is, shows
-- search and web_search.
ALTER TABLE agent_configs ADD COLUMN visible_tools JSONB NOT NULL DEFAULT '[]';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN visible_tools;
