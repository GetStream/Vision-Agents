-- +goose Up

-- end_of_turn is an agent's choice of when chorus-asr has a caller's turn ended
-- (store.EndOfTurn): the most a caller may wait, a preset of the server's by name, or bars of
-- its own. An empty document is the server's default, which every existing config and
-- session keeps. Adding a column with a constant default rewrites no rows.
ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS end_of_turn JSONB NOT NULL DEFAULT '{}';
ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS end_of_turn JSONB NOT NULL DEFAULT '{}';

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN IF EXISTS end_of_turn;
ALTER TABLE agent_configs DROP COLUMN IF EXISTS end_of_turn;
