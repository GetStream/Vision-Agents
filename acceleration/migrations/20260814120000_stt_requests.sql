-- +goose Up
CREATE TABLE stt_requests (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    audio_ms BIGINT NOT NULL DEFAULT 0,
    latency_ms DOUBLE PRECISION,
    success BOOLEAN NOT NULL,
    error_code TEXT
);

-- Stats are always read for one customer over a time range.
CREATE INDEX stt_requests_customer_started_idx ON stt_requests (customer_id, started_at DESC);

-- The rollup job scans by time across all customers.
CREATE INDEX stt_requests_started_idx ON stt_requests (started_at);

-- +goose Down
DROP TABLE stt_requests;
