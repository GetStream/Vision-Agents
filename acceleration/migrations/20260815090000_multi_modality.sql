-- +goose Up

-- One request table serves every modality. Existing rows were all speech-to-text, so the
-- default backfills them before it is dropped and the column becomes required.
ALTER TABLE stt_requests RENAME TO requests;
ALTER TABLE requests ADD COLUMN modality TEXT NOT NULL DEFAULT 'stt';
ALTER TABLE requests ALTER COLUMN modality DROP DEFAULT;

-- Text-to-speech bills by characters, speech-to-text by audio duration, and both have a
-- price, so cost is stored rather than recomputed from rates that may since have changed.
ALTER TABLE requests ADD COLUMN characters BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN cost_micros BIGINT NOT NULL DEFAULT 0;

ALTER INDEX stt_requests_started_idx RENAME TO requests_started_idx;
DROP INDEX stt_requests_customer_started_idx;
CREATE INDEX requests_customer_started_idx ON requests (modality, customer_id, started_at DESC);

-- Rollups are derived, so they are recreated rather than migrated: re-running the rollup
-- over a window rebuilds its buckets.
DROP TABLE stt_stats_hourly;
DROP TABLE stt_stats_daily;

CREATE TABLE stats_hourly (
    modality TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    audio_ms_total BIGINT NOT NULL,
    characters_total BIGINT NOT NULL,
    cost_micros_total BIGINT NOT NULL,
    request_count BIGINT NOT NULL,
    error_count BIGINT NOT NULL,
    latency_p50_ms DOUBLE PRECISION,
    latency_p95_ms DOUBLE PRECISION,
    -- Uptime is successes over total, so it is derived rather than stored twice.
    uptime DOUBLE PRECISION GENERATED ALWAYS AS (
        (request_count - error_count)::double precision / NULLIF(request_count, 0)
    ) STORED,
    PRIMARY KEY (modality, customer_id, provider, model, bucket)
);

CREATE TABLE stats_daily (
    modality TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    audio_ms_total BIGINT NOT NULL,
    characters_total BIGINT NOT NULL,
    cost_micros_total BIGINT NOT NULL,
    request_count BIGINT NOT NULL,
    error_count BIGINT NOT NULL,
    latency_p50_ms DOUBLE PRECISION,
    latency_p95_ms DOUBLE PRECISION,
    uptime DOUBLE PRECISION GENERATED ALWAYS AS (
        (request_count - error_count)::double precision / NULLIF(request_count, 0)
    ) STORED,
    PRIMARY KEY (modality, customer_id, provider, model, bucket)
);

-- +goose Down
DROP TABLE stats_daily;
DROP TABLE stats_hourly;

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

DROP INDEX requests_customer_started_idx;
ALTER INDEX requests_started_idx RENAME TO stt_requests_started_idx;
ALTER TABLE requests DROP COLUMN cost_micros;
ALTER TABLE requests DROP COLUMN characters;
ALTER TABLE requests DROP COLUMN modality;
ALTER TABLE requests RENAME TO stt_requests;
CREATE INDEX stt_requests_customer_started_idx ON stt_requests (customer_id, started_at DESC);
