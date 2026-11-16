-- +goose Up

-- A greeting is said word for word, or reworded by the model on every call. Speed is no
-- longer a setting: the voice speaks at its own pace.
ALTER TABLE agent_configs ADD COLUMN greeting_mode TEXT NOT NULL DEFAULT 'exact';
ALTER TABLE agent_configs DROP COLUMN speed;

-- +goose Down
ALTER TABLE agent_configs ADD COLUMN speed DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE agent_configs DROP COLUMN greeting_mode;
