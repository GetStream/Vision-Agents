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

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs DROP COLUMN IF EXISTS episode_cards_record;
END $$;
-- +goose StatementEnd
