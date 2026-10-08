-- +goose Up

-- What closing and summarizing an episode needs (T55, AI-884; omnichannel.Closer).
--
-- episode_activity holds it beside episodes, one row for each episode that has any, and adds
-- no column to episodes: the release before this one reads its cards with ep.* into a struct
-- of the columns episodes had, and bun refuses a column its struct lacks
-- (model_table_struct.go), so a router of that release still serving would read no card.
--
--   - last_message_at is when the latest message of a thread episode came in
--     (store.OpenEpisode sets it on each message after the first). The idle sweeper closes a
--     thread episode whose last message, or whose start while it has none, is older than the
--     idle period. An episode with no row reads as its started_at.
--   - summary_lease_until is until when the router that closed an episode, or took it
--     again, has it to summarize. An episode still ended once it runs out was left by a
--     router that stopped; the next sweep takes it. NULL once the summary is written or
--     has failed.
--
-- A new table and two new indexes: no row is rewritten.
CREATE TABLE episode_activity (
    episode_id TEXT PRIMARY KEY REFERENCES episodes (id) ON DELETE CASCADE,
    last_message_at TIMESTAMPTZ,
    summary_lease_until TIMESTAMPTZ
);

-- The ended episodes whose summary lease ran out, for the sweep to take again
-- (store.ClaimEpisodeSummaries). Only leased rows: an episode summarized, or failed, has none.
CREATE INDEX episode_activity_summary_lease ON episode_activity (summary_lease_until)
    WHERE summary_lease_until IS NOT NULL;

-- The call episodes in progress of one call, which the call.session_ended hook ends
-- (store.EndCallEpisodes) for every call that ends in the app, video calls included.
CREATE INDEX episodes_open_call ON episodes (call_id)
    WHERE status = 'in_progress' AND session_id IS NOT NULL;

-- +goose Down
DROP INDEX episodes_open_call;
DROP TABLE episode_activity;
