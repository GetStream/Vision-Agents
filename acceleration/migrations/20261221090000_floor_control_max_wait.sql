-- +goose NO TRANSACTION
-- +goose Up

-- floor_control becomes a document (store.FloorControl): the model that controls the floor,
-- and max_wait_ms, the most a caller who has finished may wait for the answer, which was
-- end_of_turn's. end_of_turn's presets and bars are no longer offered, so end_of_turn goes.
-- A JSONB column is added beside the TEXT one and takes its name, rather than the type being
-- changed, so no table is rewritten: only rows that chose a model are written. NULL is still
-- the default flow controller. The lock waits at most 5 s behind a running query, past which
-- the migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs ADD COLUMN floor_control_document JSONB;
    UPDATE agent_configs
    SET floor_control_document = jsonb_strip_nulls(jsonb_build_object('model', floor_control, 'max_wait_ms', end_of_turn -> 'max_wait_ms'))
    WHERE floor_control IS NOT NULL;
    ALTER TABLE agent_configs DROP COLUMN floor_control, DROP COLUMN end_of_turn;
    ALTER TABLE agent_configs RENAME COLUMN floor_control_document TO floor_control;

    ALTER TABLE agent_sessions ADD COLUMN floor_control_document JSONB;
    UPDATE agent_sessions
    SET floor_control_document = jsonb_strip_nulls(jsonb_build_object('model', floor_control, 'max_wait_ms', end_of_turn -> 'max_wait_ms'))
    WHERE floor_control IS NOT NULL;
    ALTER TABLE agent_sessions DROP COLUMN floor_control, DROP COLUMN end_of_turn;
    ALTER TABLE agent_sessions RENAME COLUMN floor_control_document TO floor_control;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions ADD COLUMN floor_control_model TEXT, ADD COLUMN end_of_turn JSONB NOT NULL DEFAULT '{}';
    UPDATE agent_sessions
    SET floor_control_model = floor_control ->> 'model',
        end_of_turn = jsonb_strip_nulls(jsonb_build_object('max_wait_ms', floor_control -> 'max_wait_ms'))
    WHERE floor_control IS NOT NULL;
    ALTER TABLE agent_sessions DROP COLUMN floor_control;
    ALTER TABLE agent_sessions RENAME COLUMN floor_control_model TO floor_control;

    ALTER TABLE agent_configs ADD COLUMN floor_control_model TEXT, ADD COLUMN end_of_turn JSONB NOT NULL DEFAULT '{}';
    UPDATE agent_configs
    SET floor_control_model = floor_control ->> 'model',
        end_of_turn = jsonb_strip_nulls(jsonb_build_object('max_wait_ms', floor_control -> 'max_wait_ms'))
    WHERE floor_control IS NOT NULL;
    ALTER TABLE agent_configs DROP COLUMN floor_control;
    ALTER TABLE agent_configs RENAME COLUMN floor_control_model TO floor_control;
END $$;
-- +goose StatementEnd
