-- +goose Up

-- episode_cards_record has the agent config write episode cards and their summaries while no
-- session reads them (store.AgentConfig.EpisodeCardsRecord): the record mode. episode_cards
-- stays whether sessions read the cards, and a config with it on writes them too, so off is
-- (false, false), record (false, true) and on (true, either). False for every config there
-- is, so each keeps the mode it had: true was on and false was off. A router of the previous
-- release reads episode_cards alone and takes a record config as off. A constant default
-- rewrites no row; the lock waits at most 5 s behind a running query, past which the
-- migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS episode_cards_record BOOLEAN NOT NULL DEFAULT false;
END $$;
-- +goose StatementEnd

-- channel_thread_starters holds the person who started a Linq chat: the contact map address of
-- the author of the first message the bridge admitted in it (store.SetChannelThreadStarter), as
-- the episode that message opened named them before episode_cards off opened none. A chat's
-- replies and scheduled deliveries check opt-outs, the sandbox and the limits against it while
-- the chat has no episode (channelbridge.replyTo). A table of its own, as channel_thread_people:
-- a router built before it reads channel_threads rows back whole and would fail on a new column
-- after a rollback. A data move does not carry it (store.dataTables), as it does not carry
-- channel_threads.
CREATE TABLE IF NOT EXISTS channel_thread_starters (
    channel_id TEXT PRIMARY KEY REFERENCES channel_threads (channel_id) ON DELETE CASCADE,
    starter TEXT NOT NULL
);

-- +goose Down
DROP TABLE IF EXISTS channel_thread_starters;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs DROP COLUMN IF EXISTS episode_cards_record;
END $$;
-- +goose StatementEnd
