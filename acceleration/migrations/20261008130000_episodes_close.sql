-- +goose Up

-- What closing and summarizing an episode needs (T55, AI-884; omnichannel.Closer).
--
--   - last_message_at is when the latest message of a thread episode came in
--     (store.OpenEpisode sets it on each one). The idle sweeper closes a thread episode
--     whose last message, or whose start while it has none, is older than the idle period.
--     NULL on the rows there are: they read as their started_at.
--   - summary_lease_until is until when the router that closed an episode, or took it
--     again, has it to summarize. An episode still ended once it runs out was left by a
--     router that stopped; the next sweep takes it. NULL once the summary is written or
--     has failed.
--
-- Two nullable columns with no default and one new index: no row is rewritten.
ALTER TABLE episodes ADD COLUMN last_message_at TIMESTAMPTZ;
ALTER TABLE episodes ADD COLUMN summary_lease_until TIMESTAMPTZ;

-- The ended episodes whose summary lease ran out, for the sweep to take again
-- (store.ClaimEpisodeSummaries). Only ended rows: every other status has no lease.
CREATE INDEX episodes_summary_lease ON episodes (summary_lease_until) WHERE status = 'ended';

-- +goose Down
DROP INDEX episodes_summary_lease;
ALTER TABLE episodes DROP COLUMN summary_lease_until;
ALTER TABLE episodes DROP COLUMN last_message_at;
