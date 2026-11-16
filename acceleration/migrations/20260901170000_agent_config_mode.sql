-- +goose Up

-- mode is whether the agent is spoken to or written to. A voice agent joins a call and
-- uses both speech targets; a text agent holds the same conversation in writing and uses
-- neither, so a session created from it needs no call to join.
--
-- Every config that already exists was built to be called, which is why the default is
-- voice rather than something a backfill has to decide.
ALTER TABLE agent_configs ADD COLUMN mode TEXT NOT NULL DEFAULT 'voice';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN mode;
