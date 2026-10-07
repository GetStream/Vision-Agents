-- +goose Up

-- connector_config_tokens is the app configuration token a customer's workspace admin gives
-- the router once, so the router can create, update and delete the customer's provider app
-- (T54, AI-872): for Slack, apps.manifest.create, .update and .delete take one
-- (https://docs.slack.dev/reference/methods/apps.manifest.create). One row per customer and
-- connector, the same key as the customer's connector_oauth_clients record, which is the app
-- the token manages.
--
--   - tokens_sealed is the access token and its refresh token, sealed together as one blob
--     under kek_version with customer_id and connector_id as AAD (api.configTokenAAD), with
--     the same auth.Sealer and keyring as every connector secret. Never returned by the API.
--   - expires_at is when the access token stops working: «Each app configuration token will
--     expire 12 hours after it has been generated», and tooling.tokens.rotate returns its exp
--     (https://docs.slack.dev/app-manifests/configuring-apps-with-app-manifests#config-tokens,
--     https://docs.slack.dev/reference/methods/tooling.tokens.rotate). The router rotates it
--     before then, on use, under store.WithConnectorProviderAppLock, so two routers never
--     spend the same refresh token.
--
-- A data move does not carry these rows (store.dataTables): the blob is sealed under this
-- deployment's key, and the app it manages is pinned to this deployment's events URL.

CREATE TABLE connector_config_tokens (
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    tokens_sealed BYTEA NOT NULL CHECK (tokens_sealed <> ''::bytea),
    -- Key versions start at 1 (auth.NewSealerWithKeyring).
    kek_version INTEGER NOT NULL CHECK (kek_version >= 1),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, connector_id)
);

-- +goose Down
DROP TABLE connector_config_tokens;
