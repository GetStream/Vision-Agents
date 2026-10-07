-- +goose Up

-- connection_event_subscriptions are the MCP Events webhook subscriptions the router holds on a
-- connection's MCP server (T60, AI-899; experimental-ext-triggers-events «Webhook-Based
-- Delivery» at 6682596d): one for each event a binding declares on an agent config, with the
-- binding's fixed connection. The plugin system's agent_plugin_event_subscriptions stay as they
-- are until T23.
--
--   - The key is (connection_id, config_id, binding, key): the connection is the principal the
--     server keys its side on, and key is the event and its arguments as canonical JSON,
--     hashed (mcpevents.Key), so the same filters in another order are one subscription.
--   - token is the callback's path segment, so a delivery names its subscription; the secret
--     signs each delivery. secret_sealed is the subscription's own Standard Webhooks secret
--     (whsec_...), sealed under kek_version with the customer, the connection and the
--     token as AAD (mcpevents.secretAAD), never shown to anyone.
--   - next_attempt_at is when a worker next asks the server for it: at once for a new one,
--     ahead of refresh_before for an active one, later for a refused one. A worker that takes
--     a row pushes it by its lease, so two routers never ask at once. NULL is never: a grant
--     that does not expire.
--   - failures counts the server's refusals in a row, which double the wait before the next
--     ask, a day at most (mcpevents.retryWait); a grant sets it back to 0.
--
-- A data move does not carry these rows (store.dataTables): the secrets are sealed under this
-- deployment's key, and the callback is this deployment's URL.
CREATE TABLE connection_event_subscriptions (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connection_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    binding TEXT NOT NULL,
    event TEXT NOT NULL,
    arguments JSONB NOT NULL DEFAULT '{}',
    key TEXT NOT NULL,
    token TEXT NOT NULL,
    secret_sealed BYTEA NOT NULL CHECK (secret_sealed <> ''::bytea),
    -- Key versions start at 1 (auth.NewSealerWithKeyring).
    kek_version INTEGER NOT NULL CHECK (kek_version >= 1),
    remote_id TEXT NOT NULL DEFAULT '',
    refresh_before TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'failed')),
    error TEXT NOT NULL DEFAULT '',
    next_attempt_at TIMESTAMPTZ,
    failures INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX connection_event_subscriptions_one
    ON connection_event_subscriptions (connection_id, config_id, binding, key);
CREATE UNIQUE INDEX connection_event_subscriptions_token
    ON connection_event_subscriptions (token);
-- What a worker takes next (store.ClaimConnectionEventSubscriptions).
CREATE INDEX connection_event_subscriptions_due
    ON connection_event_subscriptions (next_attempt_at) WHERE next_attempt_at IS NOT NULL;

-- Every event delivered, so a retried delivery opens no second conversation. A subscription
-- that goes takes its rows with it.
CREATE TABLE connection_event_deliveries (
    subscription_id TEXT NOT NULL REFERENCES connection_event_subscriptions (id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (subscription_id, event_id)
);

-- +goose Down
DROP TABLE connection_event_deliveries;
DROP TABLE connection_event_subscriptions;
