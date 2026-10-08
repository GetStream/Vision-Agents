-- +goose Up

-- A connector_oauth_clients row may be a provider app with no OAuth client (AI-863, T36): the
-- customer's own Linq account, whose webhook subscription signs its events with a secret of
-- its own («your subscription's signing secret», https://docs.linqapp.com/guides/webhooks/index.md,
-- opened October 8, 2026) and whose messages are sent with a bearer API key, not through an
-- OAuth client. Such a row has a provider_app_id and a signing secret and no client_id.
--
--   - client_id may be '' only when the row has a signing secret, so every row is still an
--     OAuth client, a provider app the events route can verify (api.ProviderApp), or both.
--     The store checks the same (store.checkOAuthClient), and the API takes a put without
--     client_id only for a connector whose consents need none (api.checkOAuthClient).
--   - Expand only: no column changes and no row is rewritten. Every existing row has a
--     client_id, so it meets the new CHECK, and a build from before this file reads a row
--     without one as a client whose id is '': only oauth2_code reads a client_id
--     (api.ConnectorClients), and the API refuses such a row for a connector that lists it.
--
-- The new CHECK goes on before the old one comes off, so no moment allows a row with neither.

ALTER TABLE connector_oauth_clients ADD CONSTRAINT connector_oauth_clients_client_or_signing_secret
    CHECK (client_id <> '' OR signing_secret_sealed <> ''::bytea);
ALTER TABLE connector_oauth_clients DROP CONSTRAINT connector_oauth_clients_client_id_check;

-- +goose Down
-- NOT VALID: the CHECK holds for every row written from now on and leaves the provider apps
-- already stored without a client_id in place, rather than delete a customer's record.
ALTER TABLE connector_oauth_clients ADD CONSTRAINT connector_oauth_clients_client_id_check
    CHECK (client_id <> '') NOT VALID;
ALTER TABLE connector_oauth_clients DROP CONSTRAINT connector_oauth_clients_client_or_signing_secret;
