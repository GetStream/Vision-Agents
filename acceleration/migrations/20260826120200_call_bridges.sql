-- +goose Up

-- Plivo, Bandwidth and Sinch do not take a call plan on the request that places a call. They
-- take a URL, and fetch it when the person answers. A bridge is what that URL has to answer
-- with, parked here between placing the call and the person picking up.
--
-- The token is the whole of the request's authentication, because the vendor fetching the URL
-- has no customer header to send. That is why a bridge is single-use and short-lived: the row
-- is deleted the first time it is rendered, so a token that leaks after the call is answered
-- is a token for a bridge that no longer exists, rather than a standing way into a customer's
-- calls.
CREATE TABLE call_bridges (
    token TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    vendor TEXT NOT NULL,
    -- trunk_uri is the SIP address the answered leg is transferred to.
    trunk_uri TEXT NOT NULL,
    -- trunk_username and trunk_password are the trunk's digest credentials, present only for
    -- a vendor that can actually send them.
    trunk_username TEXT,
    trunk_password TEXT,
    -- initial_digits are pressed at the person once they answer, before the transfer.
    initial_digits TEXT,
    -- call_id names the Stream call the leg is routed into. It is here for the audit trail
    -- rather than for the vendor, which learns nothing about Stream beyond the trunk.
    call_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Expired rows are swept rather than relied on being read, since a call nobody answers
-- leaves its bridge behind.
CREATE INDEX call_bridges_expires_idx ON call_bridges (expires_at);

-- +goose Down
DROP TABLE call_bridges;
