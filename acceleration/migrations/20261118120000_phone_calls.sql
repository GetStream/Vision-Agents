-- +goose Up

-- One row per phone call the router placed, written before the dial, and how that call ended.
--
-- The token is in the URL the vendor posts the call's status to, and is the whole of that
-- request's authentication, because the vendor has no customer header to send. It is not
-- consumed: a vendor posts several times for one call. It is cleared once the call is long
-- over, and the row kept.
--
-- This is a table of its own rather than columns of call_legs because the leg opens only after
-- the dial, and a dial that rang out still has an outcome. Rows are kept in the shard that
-- placed the call. A data move does not carry them, as it does not carry call_legs.
CREATE TABLE phone_calls (
    id BIGSERIAL PRIMARY KEY,
    -- NULL once swept.
    token TEXT UNIQUE,
    customer_id TEXT NOT NULL,
    -- The Stream app the call is in. NULL is the deployment's own app.
    stream_app_pk BIGINT,
    call_type TEXT NOT NULL,
    call_id TEXT NOT NULL,
    vendor TEXT NOT NULL,
    local_number TEXT NOT NULL,
    remote_number TEXT NOT NULL,
    -- reports is whether the vendor posts a final status, so a call that ends never joined
    -- with no report from it is no_answer rather than unknown.
    reports BOOLEAN NOT NULL,
    vendor_call_id TEXT,
    placed_at TIMESTAMPTZ NOT NULL,
    -- When the person's phone joined the Stream call, and when its leg ended.
    answered_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    -- NULL until decided.
    outcome TEXT CHECK (outcome IN ('completed', 'busy', 'no_answer', 'failed', 'canceled', 'unknown')),
    -- The vendor's own word for the call, final or not, kept for debugging: its last one
    -- before the outcome was decided.
    vendor_status TEXT,
    decided_at TIMESTAMPTZ
);

CREATE INDEX phone_calls_customer_call_idx ON phone_calls (customer_id, call_id, id DESC);
CREATE INDEX phone_calls_call_idx ON phone_calls (call_type, call_id);
-- Only the tokens still to sweep.
CREATE INDEX phone_calls_placed_idx ON phone_calls (placed_at) WHERE token IS NOT NULL;

-- +goose Down
DROP TABLE phone_calls;
