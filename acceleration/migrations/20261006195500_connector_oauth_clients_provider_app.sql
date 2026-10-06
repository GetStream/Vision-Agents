-- +goose Up

-- A connector_oauth_clients row is also the provider app the client belongs to (T40, AI-871;
-- subtasks.md on connectors/planning): the Slack app the customer's agent is, with its id and
-- the secret Slack signs that app's events with. One row per app and connector stays the rule
-- (the primary key), so a provider app is the customer's one record for the connector.
--
--   - provider_app_id is the provider's id for the app, '' when the client has none the router
--     needs: Slack's app id, which apps.manifest.create returns as app_id beside
--     credentials.client_id, client_secret and signing_secret
--     (https://docs.slack.dev/reference/methods/apps.manifest.create, example A012ABCD0A0).
--     The signing secret is one per app: «Slack creates a unique string for your app»
--     (https://docs.slack.dev/authentication/verifying-requests-from-slack). A provider app
--     belongs to one customer, so it is unique among one connector's rows; '' is not an app
--     and repeats.
--   - registration gains managed, an app the router created for the customer (T54). Still
--     checked by the store (store.PutConnectorOAuthClient), not by a CHECK.
--   - stream_app_pk is the Stream app the record was created in, and the one the provider app's
--     work is finished in (architecture doc on connectors/planning, decision 7: «provider apps
--     keep a stream_app_pk pin»). It is 20261006180000_stream_app_pins.sql's pin: NULL is the
--     deployment's own app. A customer holds one Stream app at a time (stream_apps.customer_id
--     is its primary key), so a re-registration moves the customer and leaves the pin: a Slack
--     app created while the customer was in app 4242 still writes into 4242. Set when the row
--     is created; a put that replaces the row keeps it.
--   - signing_secret_sealed is the secret the provider signs the app's inbound requests with,
--     sealed under signing_kek_version with the customer, the connector and the provider app
--     id as AAD (api.providerAppAAD), so it opens only for the app it was issued for. It has a
--     key version of its own beside kek_version, as each sealed blob here records the key it
--     was sealed under. Empty, with version 0, for a client whose app posts no events.
--
-- The secrets stay on this row rather than on a connector_connections row: oauth2_code already
-- reads the client secret here at every exchange, refresh and revocation (api.ConnectorClients),
-- and a connection's credentials open only to the scheme that issued them (core.AccessCredential).
-- Both are sealed with the same auth.Sealer and keyring as every connector secret.
--
-- A data move still does not carry these rows (store.dataTables) and the table has no
-- record_data_change trigger, so neither store.secretColumns nor that function names
-- signing_secret_sealed. Carrying the table would need both to.

ALTER TABLE connector_oauth_clients ADD COLUMN provider_app_id TEXT NOT NULL DEFAULT '';
ALTER TABLE connector_oauth_clients ADD COLUMN stream_app_pk BIGINT;
ALTER TABLE connector_oauth_clients ADD COLUMN signing_secret_sealed BYTEA NOT NULL DEFAULT ''::bytea;
-- Key versions start at 1 (auth.NewSealerWithKeyring); 0 is no secret.
ALTER TABLE connector_oauth_clients ADD COLUMN signing_kek_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE connector_oauth_clients ADD CONSTRAINT connector_oauth_clients_signing_secret
    CHECK ((signing_secret_sealed = ''::bytea AND signing_kek_version = 0)
        OR (signing_secret_sealed <> ''::bytea AND signing_kek_version >= 1));
-- A signing secret is found by the app it signs for (store.ConnectorOAuthClientByProviderApp),
-- so one without an app could never be read.
ALTER TABLE connector_oauth_clients ADD CONSTRAINT connector_oauth_clients_signing_secret_app
    CHECK (signing_secret_sealed = ''::bytea OR provider_app_id <> '');

-- One customer per provider app of a connector, and the lookup an events URL naming the app
-- makes (store.ConnectorOAuthClientByProviderApp).
CREATE UNIQUE INDEX connector_oauth_clients_provider_app
    ON connector_oauth_clients (connector_id, provider_app_id)
    WHERE provider_app_id <> '';

-- +goose Down
DROP INDEX connector_oauth_clients_provider_app;
ALTER TABLE connector_oauth_clients DROP CONSTRAINT connector_oauth_clients_signing_secret_app;
ALTER TABLE connector_oauth_clients DROP CONSTRAINT connector_oauth_clients_signing_secret;
ALTER TABLE connector_oauth_clients DROP COLUMN signing_kek_version;
ALTER TABLE connector_oauth_clients DROP COLUMN signing_secret_sealed;
ALTER TABLE connector_oauth_clients DROP COLUMN stream_app_pk;
ALTER TABLE connector_oauth_clients DROP COLUMN provider_app_id;
