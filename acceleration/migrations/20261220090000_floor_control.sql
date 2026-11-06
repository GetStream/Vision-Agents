-- +goose Up

-- floor_control is whether chorus-asr rules an agent's floor (store.FloorControl): the model,
-- and its mode, off, shadow or act. NULL is an agent that never chose, which is off, as every
-- existing config and session is. A nullable column with no default rewrites no rows.
ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS floor_control JSONB;
ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS floor_control JSONB;

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN IF EXISTS floor_control;
ALTER TABLE agent_configs DROP COLUMN IF EXISTS floor_control;
