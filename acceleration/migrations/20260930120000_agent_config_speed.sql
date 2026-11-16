-- +goose Up

-- speed is the voice's rate of delivery, 1 being its own. Zero leaves it there.
ALTER TABLE agent_configs ADD COLUMN speed DOUBLE PRECISION NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN speed;
