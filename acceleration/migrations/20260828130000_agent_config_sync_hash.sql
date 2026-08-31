-- +goose Up

-- sync_hash is a fingerprint of the last directory written onto this config. A second
-- sync with the same hash does nothing.
ALTER TABLE agent_configs ADD COLUMN sync_hash TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN sync_hash;
