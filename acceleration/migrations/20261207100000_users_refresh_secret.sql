-- +goose NO TRANSACTION
-- +goose Up

-- refresh_secret_hash is the SHA-256 of the secret a guest the router minted was last handed,
-- which is what lets that guest come back for a fresh token once theirs has expired. A guest id
-- is no proof of being that guest: it is the author of their messages and a member of their
-- channels, so anyone who shared a channel with them knows it. Only the hash is kept, and it is
-- replaced every time the secret is used. Null for a guest handed none, who can only be given a
-- new token by their own still-valid one or by the app's backend. The column has no default, so
-- adding it rewrites no row; its lock waits at most 5 s behind a running query, past which the
-- migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE users ADD COLUMN IF NOT EXISTS refresh_secret_hash BYTEA;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE users DROP COLUMN IF EXISTS refresh_secret_hash;
END $$;
-- +goose StatementEnd
