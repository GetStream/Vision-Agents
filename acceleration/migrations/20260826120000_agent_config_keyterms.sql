-- +goose Up

ALTER TABLE agent_configs ADD COLUMN keyterms JSONB NOT NULL DEFAULT '[]';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN keyterms;
