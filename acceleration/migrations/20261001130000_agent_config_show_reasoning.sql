-- +goose Up

-- show_reasoning streams the model's thinking onto a persistent conversation's reply while
-- it is written, on live updates only. Off, which every existing config is, shows none.
ALTER TABLE agent_configs ADD COLUMN show_reasoning BOOLEAN NOT NULL DEFAULT FALSE;

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN show_reasoning;
