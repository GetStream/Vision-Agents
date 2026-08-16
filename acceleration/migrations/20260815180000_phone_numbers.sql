-- +goose Up

-- A number is inventory, not usage: it is bought once, costs money every month whether or
-- not anybody rings it, and is eventually given back. That is why it is its own table
-- rather than more request rows, and why releasing one is a timestamp rather than a
-- delete: what a number cost last month is still true after it is gone.
CREATE TABLE phone_numbers (
    id BIGSERIAL PRIMARY KEY,
    e164 TEXT NOT NULL,
    vendor TEXT NOT NULL,
    country TEXT NOT NULL,
    -- capabilities is what the number can carry: voice, sms, mms, fax.
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    monthly_cost_micros BIGINT NOT NULL DEFAULT 0,
    customer_id TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    -- vendor_id is the vendor's own identifier, which is what releasing or reconfiguring
    -- the number needs.
    vendor_id TEXT,
    -- stream_trunk_id is the SIP trunk calls to this number arrive on, empty until the
    -- number has been attached to one.
    stream_trunk_id TEXT,
    purchased_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    released_at TIMESTAMPTZ
);

-- One vendor cannot sell the same number twice while it is still held, but the same
-- number can be bought again after being released, so the constraint only covers the ones
-- still in service.
CREATE UNIQUE INDEX phone_numbers_held_idx ON phone_numbers (vendor, e164)
    WHERE released_at IS NULL;
CREATE INDEX phone_numbers_customer_idx ON phone_numbers (customer_id, purchased_at DESC);
CREATE INDEX phone_numbers_tags_idx ON phone_numbers USING GIN (tags);

-- +goose Down
DROP TABLE phone_numbers;
