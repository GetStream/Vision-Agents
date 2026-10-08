-- +goose Up

-- episode_cards is whether a phone call under the agent config writes an episode card into
-- the caller's omni-channel (T41, AI-883; store.AgentConfig.EpisodeCards). False for every
-- config there is, so a call runs as it did before the cards existed: no call read, no
-- contact map row, no card. A config turns it on.
ALTER TABLE agent_configs ADD COLUMN episode_cards BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN episode_cards;
