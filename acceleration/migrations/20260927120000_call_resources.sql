-- +goose Up

-- The Stream SIP trunk and routing rule created for one outbound leg, so they can be
-- deleted when the call ends. Call and Transfer each make a per-call trunk; keeping the
-- ids here is the only way the session_ended hook knows what to tear down, since a trunk
-- is a billable resource that otherwise lives forever.
--
-- Keyed by trunk_id because one call can own several: its own trunk plus one per transfer
-- into the same call. Cleanup deletes every row for the (call_type, call_id) that ended.
CREATE TABLE call_resources (
    trunk_id TEXT PRIMARY KEY,
    route_id TEXT NOT NULL,
    call_type TEXT NOT NULL,
    call_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX call_resources_call_idx ON call_resources (call_type, call_id);

-- +goose Down
DROP TABLE call_resources;
