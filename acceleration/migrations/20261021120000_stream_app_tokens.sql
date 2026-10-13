-- +goose Up

-- The newest server token the gateway minted for a Stream app, which the router acts in the
-- app with when it holds none of the app's secrets. Sealed under this deployment's key and
-- bound to the app and its key, so a token copied to another row does not open: one is the
-- run of the app until it expires, and cannot be revoked short of rotating the secret.
CREATE TABLE stream_app_tokens (
    stream_app_pk BIGINT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    api_key TEXT NOT NULL,
    token_sealed BYTEA NOT NULL,
    kek_version INTEGER NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- +goose Down
DROP TABLE stream_app_tokens;
