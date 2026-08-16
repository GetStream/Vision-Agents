-- +goose Up

-- Cost tags are the customer's own labels, so what drives spend can be broken down by
-- whatever the customer cares about rather than only by provider and model. The keys are
-- not typed or known in advance, which is why they are one jsonb column and not columns.
ALTER TABLE requests ADD COLUMN tags JSONB NOT NULL DEFAULT '{}';
ALTER TABLE requests ADD COLUMN agent_id TEXT;
ALTER TABLE requests ADD COLUMN call_id TEXT;

CREATE INDEX requests_tags_idx ON requests USING GIN (tags);
CREATE INDEX requests_agent_started_idx ON requests (agent_id, started_at DESC)
    WHERE agent_id IS NOT NULL;

-- Tags get their own rollups rather than more columns on stats_hourly, because a row
-- carries a set of labels rather than one more dimension. Unrolling them one key at a
-- time means a request tagged {project, environment} lands in both breakdowns.
CREATE TABLE stats_tags_hourly (
    modality TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    tag_key TEXT NOT NULL,
    tag_value TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    audio_ms_total BIGINT NOT NULL,
    characters_total BIGINT NOT NULL,
    input_tokens_total BIGINT NOT NULL,
    cached_input_tokens_total BIGINT NOT NULL,
    output_tokens_total BIGINT NOT NULL,
    cost_micros_total BIGINT NOT NULL,
    request_count BIGINT NOT NULL,
    error_count BIGINT NOT NULL,
    latency_p50_ms DOUBLE PRECISION,
    latency_p95_ms DOUBLE PRECISION,
    uptime DOUBLE PRECISION GENERATED ALWAYS AS (
        (request_count - error_count)::double precision / NULLIF(request_count, 0)
    ) STORED,
    PRIMARY KEY (modality, customer_id, tag_key, tag_value, bucket)
);

CREATE TABLE stats_tags_daily (
    modality TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    tag_key TEXT NOT NULL,
    tag_value TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    audio_ms_total BIGINT NOT NULL,
    characters_total BIGINT NOT NULL,
    input_tokens_total BIGINT NOT NULL,
    cached_input_tokens_total BIGINT NOT NULL,
    output_tokens_total BIGINT NOT NULL,
    cost_micros_total BIGINT NOT NULL,
    request_count BIGINT NOT NULL,
    error_count BIGINT NOT NULL,
    latency_p50_ms DOUBLE PRECISION,
    latency_p95_ms DOUBLE PRECISION,
    uptime DOUBLE PRECISION GENERATED ALWAYS AS (
        (request_count - error_count)::double precision / NULLIF(request_count, 0)
    ) STORED,
    PRIMARY KEY (modality, customer_id, tag_key, tag_value, bucket)
);

-- +goose Down
DROP TABLE stats_tags_daily;
DROP TABLE stats_tags_hourly;
DROP INDEX requests_agent_started_idx;
DROP INDEX requests_tags_idx;
ALTER TABLE requests DROP COLUMN call_id;
ALTER TABLE requests DROP COLUMN agent_id;
ALTER TABLE requests DROP COLUMN tags;
