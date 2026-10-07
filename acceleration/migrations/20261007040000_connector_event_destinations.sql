-- +goose Up

-- connector_event_destinations are the customer URLs a connector's raw provider events are
-- forwarded to (T46, AI-875; channels.md on connectors/planning, «Integration modes: full
-- platform, customizations, pass-through», modes B and C). One row per destination; a customer
-- has at most three for one connector (store.MaxEventDestinations).
--
--   - forward says which deliveries go there: unhandled is the ones the router acted on in no
--     way (no signal, and no message an agent answers), such as a Slack block_actions or
--     reaction_added; all is every verified delivery but a URL handshake.
--   - url passed egress.ValidatePublicHTTPSURL when it was stored, and every send dials
--     through egress again.
--   - secret_sealed is the destination's own Standard Webhooks signing secret (whsec_...),
--     sealed under kek_version with the customer, the connector and the destination id as AAD
--     (eventforward.secretAAD). Never returned after the create that made it.
--   - previous_secret_sealed is the secret a rotation replaced, which still signs beside the
--     new one until previous_until, so a receiver can switch without dropping a delivery
--     (https://www.standardwebhooks.com/, «zero downtime secret rotation»). NULL when no
--     rotation is running.
--
-- A data move does not carry these rows (store.dataTables): the secrets are sealed under this
-- deployment's key.
CREATE TABLE connector_event_destinations (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    url TEXT NOT NULL,
    forward TEXT NOT NULL CHECK (forward IN ('unhandled', 'all')),
    secret_sealed BYTEA NOT NULL CHECK (secret_sealed <> ''::bytea),
    -- Key versions start at 1 (auth.NewSealerWithKeyring).
    kek_version INTEGER NOT NULL CHECK (kek_version >= 1),
    previous_secret_sealed BYTEA,
    previous_kek_version INTEGER NOT NULL DEFAULT 0,
    previous_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- A customer's destinations of one connector, newest first: the list endpoint's order and
-- cursor (store.EventDestinations), and the forward's lookup.
CREATE INDEX connector_event_destinations_list
    ON connector_event_destinations (customer_id, connector_id, created_at DESC, id DESC);

-- connector_event_deliveries are the forwards not yet sent: one per delivery and destination,
-- deleted once the destination took it, refused it, or the retries ran out
-- (eventforward.Forwarder). The events route writes them before it answers the provider, and
-- a worker on any router sends them, so the provider's ack never waits on a customer's URL.
--
--   - id is the Standard Webhooks webhook-id, the same on every attempt: msg_ and a digest of
--     the body, so a provider's own retry of a delivery is the same id and is not queued twice
--     while the first is pending (https://www.standardwebhooks.com/, «remains the same no
--     matter how many times a webhook that has failed is retried»).
--   - headers are the provider's own headers the receiver needs to verify the body itself,
--     as they came, and body is the raw provider body, unchanged.
--   - next_attempt_at is when the next attempt is due. A worker that takes a delivery pushes it
--     forward by its lease, so another router takes it only once that lease ran out.
--
-- A data move does not carry these rows: they are in flight, minutes to hours.
CREATE TABLE connector_event_deliveries (
    destination_id TEXT NOT NULL REFERENCES connector_event_destinations (id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    headers JSONB NOT NULL DEFAULT '{}',
    body BYTEA NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (destination_id, id)
);

-- What a worker takes next (store.ClaimEventDeliveries).
CREATE INDEX connector_event_deliveries_due ON connector_event_deliveries (next_attempt_at);

-- +goose Down
DROP TABLE connector_event_deliveries;
DROP TABLE connector_event_destinations;
