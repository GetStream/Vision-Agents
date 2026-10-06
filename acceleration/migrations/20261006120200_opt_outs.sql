-- +goose Up

-- opt_outs are the people who asked not to be reached. Nothing is sent or dialled to a
-- recipient with a live row for the channel, or for all of them. Revoking keeps the row,
-- because when somebody opted out and back in is the record a carrier asks for.
CREATE TABLE opt_outs (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    recipient TEXT NOT NULL,
    -- channel is sms, voice, whatsapp, imessage or all.
    channel TEXT NOT NULL,
    -- source is keyword (they texted STOP), api or dashboard.
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX opt_outs_recipient_idx
    ON opt_outs (customer_id, recipient, channel)
    WHERE revoked_at IS NULL;
CREATE INDEX opt_outs_customer_idx ON opt_outs (customer_id, created_at DESC, id DESC);

-- sandbox_recipients are the few numbers an app without an approved use case may text and
-- call on the hosted router.
CREATE TABLE sandbox_recipients (
    customer_id TEXT NOT NULL,
    recipient TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (customer_id, recipient)
);

-- +goose Down
DROP TABLE sandbox_recipients;
DROP TABLE opt_outs;
