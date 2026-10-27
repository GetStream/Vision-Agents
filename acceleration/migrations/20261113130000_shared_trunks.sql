-- +goose Up

-- One Stream SIP trunk carries up to 50 of a customer's numbers of one vendor in one Stream
-- app, each with a routing rule of its own. Stream returns the trunk's password only when it
-- is made, so it is kept here, sealed, for every number that joins the trunk later.
CREATE TABLE IF NOT EXISTS stream_trunks (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- NULL is the deployment's own app.
    stream_app_pk BIGINT,
    vendor TEXT NOT NULL,
    numbers TEXT[] NOT NULL,
    bridge_uri TEXT NOT NULL,
    bridge_username TEXT NOT NULL,
    bridge_password_sealed BYTEA NOT NULL,
    bridge_password_kek_version INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS stream_trunks_group_idx ON stream_trunks (customer_id, stream_app_pk, vendor);

-- A number on a trunk only to place calls is not attached: its callers reach no agent.
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS attached_at TIMESTAMPTZ;
UPDATE phone_numbers SET attached_at = purchased_at
WHERE stream_trunk_id IS NOT NULL AND released_at IS NULL AND attached_at IS NULL;

-- Calls on a shared trunk each record it, so the trunk no longer identifies a row.
ALTER TABLE call_resources
    DROP CONSTRAINT IF EXISTS call_resources_pkey,
    ADD COLUMN IF NOT EXISTS id BIGSERIAL PRIMARY KEY,
    ADD COLUMN IF NOT EXISTS shared BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
DELETE FROM call_resources WHERE shared;
ALTER TABLE call_resources
    DROP COLUMN IF EXISTS shared,
    DROP COLUMN IF EXISTS id,
    ADD PRIMARY KEY (trunk_id);
ALTER TABLE phone_numbers DROP COLUMN IF EXISTS attached_at;
DROP TABLE IF EXISTS stream_trunks;
