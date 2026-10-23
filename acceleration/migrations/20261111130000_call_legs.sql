-- +goose Up

-- One leg of a phone call, from when it was connected to when the call ended: what call
-- minutes are billed from, and what a limit on concurrent calls counts. A placed call and
-- each transfer into it are an outbound leg each; a call to one of the app's numbers is an
-- inbound one. Kept in the shard that opened it, beside the call's other rows.
CREATE TABLE call_legs (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    organization_id TEXT NOT NULL DEFAULT '',
    -- Who the leg's minutes are billed to and counted against, from the leg's owner when it
    -- opened: a key's outbound call keeps its key; an inbound one is phone:inbound.
    organization_tier TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    usage_categories TEXT[] NOT NULL DEFAULT '{}',
    -- The Stream app the call is in. NULL is the deployment's own app.
    stream_app_pk BIGINT,
    call_type TEXT NOT NULL,
    call_id TEXT NOT NULL,
    vendor TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    local_number TEXT NOT NULL,
    remote_number TEXT NOT NULL,
    -- The ISO 3166-1 alpha-2 country the leg is priced by, empty when unknown.
    country TEXT NOT NULL DEFAULT '',
    agent_id TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ
);
CREATE INDEX call_legs_call_idx ON call_legs (call_type, call_id) WHERE ended_at IS NULL;
CREATE INDEX call_legs_open_idx ON call_legs (organization_id) WHERE ended_at IS NULL;
-- A call that arrives is announced by every delivery of its session_started, and only the
-- first opens its leg.
CREATE UNIQUE INDEX call_legs_inbound_idx ON call_legs (call_type, call_id, COALESCE(stream_app_pk, 0))
    WHERE ended_at IS NULL AND direction = 'inbound';

-- +goose Down
DROP TABLE call_legs;
