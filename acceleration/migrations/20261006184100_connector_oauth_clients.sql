-- +goose Up

-- connector_oauth_clients is the OAuth client one app uses at one connector when the client
-- was registered in advance (T19, AI-846; architecture doc on connectors/planning, «Axes where
-- providers differ» row 10, and «BYO client» in «AI-816: keep, change, add»). One row per app
-- and connector, so rotating a secret touches this row alone, and oauth2_code reads it again
-- at every exchange, refresh and revocation (clientSecret in
-- internal/connectors/schemes/oauth2code/client.go): no connection keeps a copy of the secret.
--
--   - registration is core.ClientRegistrationMethod: customer for a client the app registered
--     and put through the API, operator for this deployment's own client registered for one
--     app. Checked by the store (store.PutConnectorOAuthClient), not by a CHECK, as
--     connector_authorization_attempts.kind is: T40 adds managed, an app the router created,
--     and with the provider app id as one more column it is the same table.
--   - auth_method is core.ClientAuthMethod, '' when the app left it to the scheme
--     (preregisteredMethod in oauth2code/client.go).
--   - secret_sealed is the client secret sealed under kek_version, with customer_id and
--     connector_id as AAD (api.oauthClientAAD), so a blob copied onto another row does not
--     open. Empty, with version 0, for a public client (RFC 7591 section 2, «none»). The
--     column has the name record_data_change already strips and store.secretColumns already
--     lists, so a data move would leave the secret behind if the table were ever carried.
--   - connector_id has no foreign key: a built-in's definition is under customer '' and a
--     custom one under this row's customer (20261002190000_connector_definitions.sql), as
--     connector_connections.definition_revision has none.
--
-- A data move does not carry these rows (store.dataTables): the secret is sealed under this
-- deployment's key, and a client without it cannot authenticate, so the app puts it again.

CREATE TABLE connector_oauth_clients (
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    registration TEXT NOT NULL,
    client_id TEXT NOT NULL CHECK (client_id <> ''),
    auth_method TEXT NOT NULL DEFAULT '',
    secret_sealed BYTEA NOT NULL DEFAULT ''::bytea,
    -- Key versions start at 1 (auth.NewSealerWithKeyring); 0 is no secret.
    kek_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, connector_id),
    CONSTRAINT connector_oauth_clients_secret
        CHECK ((secret_sealed = ''::bytea AND kek_version = 0) OR (secret_sealed <> ''::bytea AND kek_version >= 1))
);

-- +goose Down
DROP TABLE connector_oauth_clients;
