-- +goose Up

-- channel_accounts is one app's line on a channel: the number people write to, and the
-- credentials of the provider carrying it. It is connected once for the app and named by
-- any number of agents, the way a plugin login is.
--
-- The credentials are sealed rather than hashed, for the reason the connector ones are:
-- verifying a delivery means recomputing its signature, and sending means presenting the
-- token, so the material has to be recoverable.
CREATE TABLE channel_accounts (
    id          TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- kind is the channel it carries: whatsapp, sms or imessage.
    kind        TEXT NOT NULL,
    e164        TEXT NOT NULL,
    -- account_id is the provider's own id for the line, such as WhatsApp's phone number
    -- id, which sending needs. Not a secret: every delivery carries it too.
    account_id  TEXT NOT NULL DEFAULT '',
    -- token is the webhook's own path segment, so the provider delivering to it is told
    -- apart from anybody who guessed the route.
    token          TEXT NOT NULL,
    secrets_sealed BYTEA,
    kek_version    INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL,
    deleted_at  TIMESTAMPTZ
);

-- One line per customer, channel and number while it is held. The same number can be
-- connected again after being disconnected, so the constraint only covers the live ones.
CREATE UNIQUE INDEX channel_accounts_line_idx
    ON channel_accounts (customer_id, kind, e164)
    WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX channel_accounts_token_idx ON channel_accounts (token);

-- Every message delivered, so a provider retrying one is answered once. Providers retry a
-- delivery that is slow, and answering twice would have the agent reply twice.
CREATE TABLE channel_deliveries (
    account_id  TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (account_id, message_id)
);

-- +goose Down
DROP TABLE channel_deliveries;
DROP TABLE channel_accounts;
