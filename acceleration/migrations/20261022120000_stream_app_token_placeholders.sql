-- +goose Up

-- An app that registered its keys is acted in with the gateway's tokens from now on. Each
-- connected one is given a placeholder that has already expired, naming only the key to
-- mint with, so work for it that no request carries a token for, a queued write or a
-- campaign, asks the gateway for one rather than waiting for the app's next request.
INSERT INTO stream_app_tokens (stream_app_pk, customer_id, api_key, token_sealed, kek_version, expires_at)
SELECT stream_app_pk, customer_id, primary_key, ''::bytea, 0, 'epoch'::timestamptz
FROM stream_apps
WHERE state = 'connected' AND primary_key IS NOT NULL AND customer_id = stream_app_pk::text
ON CONFLICT (stream_app_pk) DO NOTHING;

-- +goose Down
DELETE FROM stream_app_tokens WHERE token_sealed = ''::bytea;
