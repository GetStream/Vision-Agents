-- +goose Up
CREATE TABLE stt_stats_hourly (
    customer_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    audio_ms_total BIGINT NOT NULL,
    request_count BIGINT NOT NULL,
    error_count BIGINT NOT NULL,
    latency_p50_ms DOUBLE PRECISION,
    latency_p95_ms DOUBLE PRECISION,
    -- Uptime is successes over total, so it is derived rather than stored twice.
    uptime DOUBLE PRECISION GENERATED ALWAYS AS (
        (request_count - error_count)::double precision / NULLIF(request_count, 0)
    ) STORED,
    PRIMARY KEY (customer_id, provider, model, bucket)
);

CREATE TABLE stt_stats_daily (
    customer_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    audio_ms_total BIGINT NOT NULL,
    request_count BIGINT NOT NULL,
    error_count BIGINT NOT NULL,
    latency_p50_ms DOUBLE PRECISION,
    latency_p95_ms DOUBLE PRECISION,
    uptime DOUBLE PRECISION GENERATED ALWAYS AS (
        (request_count - error_count)::double precision / NULLIF(request_count, 0)
    ) STORED,
    PRIMARY KEY (customer_id, provider, model, bucket)
);

-- +goose Down
DROP TABLE stt_stats_daily;
DROP TABLE stt_stats_hourly;
