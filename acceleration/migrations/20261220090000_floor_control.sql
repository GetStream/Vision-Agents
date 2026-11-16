-- +goose Up

-- floor_control is the model that controls an agent's floor (store.FloorControl): chorus-asr,
-- the one there is. NULL is the default flow controller, which every existing config and
-- session keeps. A nullable column with no default rewrites no rows.
ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS floor_control TEXT;
ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS floor_control TEXT;

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN IF EXISTS floor_control;
ALTER TABLE agent_configs DROP COLUMN IF EXISTS floor_control;
