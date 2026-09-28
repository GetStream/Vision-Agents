-- +goose Up
ALTER TABLE connector_authorization_attempts
    ADD COLUMN kek_version INTEGER NOT NULL DEFAULT 1;

-- +goose Down
ALTER TABLE connector_authorization_attempts DROP COLUMN kek_version;
