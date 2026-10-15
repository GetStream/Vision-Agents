-- +goose Up

-- A call's turns and decisions explain a call somebody is looking at now, so they are kept
-- in Redis for an hour instead of a row each here, forever. What lasts about a turn is
-- counted into its hour as the turn finishes: the counts, and a histogram per leg that the
-- percentiles are read off. A day is its hours added up.
DROP TABLE turns;
DROP TABLE call_events;
DROP TABLE turn_stats_daily;

ALTER TABLE turn_stats_hourly
    DROP COLUMN stt_latency_p50_ms,
    DROP COLUMN stt_latency_p95_ms,
    DROP COLUMN llm_ttft_p50_ms,
    DROP COLUMN llm_ttft_p95_ms,
    DROP COLUMN tts_ttfb_p50_ms,
    DROP COLUMN tts_ttfb_p95_ms,
    DROP COLUMN roundtrip_p50_ms,
    DROP COLUMN roundtrip_p95_ms,
    DROP COLUMN roundtrip_p99_ms;

-- turn_latency_hourly counts the turns of one hour whose leg took about as long as bin
-- says: bin b holds latencies in (1.05^(b-1), 1.05^b] ms, so a percentile read off it is
-- within 5% of the true one.
CREATE TABLE turn_latency_hourly (
    customer_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    -- leg is stt_latency, llm_ttft, tts_ttfb or roundtrip.
    leg TEXT NOT NULL,
    bin SMALLINT NOT NULL,
    count BIGINT NOT NULL,
    PRIMARY KEY (customer_id, agent_id, bucket, leg, bin)
);

-- +goose Down
DROP TABLE turn_latency_hourly;

ALTER TABLE turn_stats_hourly
    ADD COLUMN stt_latency_p50_ms DOUBLE PRECISION,
    ADD COLUMN stt_latency_p95_ms DOUBLE PRECISION,
    ADD COLUMN llm_ttft_p50_ms DOUBLE PRECISION,
    ADD COLUMN llm_ttft_p95_ms DOUBLE PRECISION,
    ADD COLUMN tts_ttfb_p50_ms DOUBLE PRECISION,
    ADD COLUMN tts_ttfb_p95_ms DOUBLE PRECISION,
    ADD COLUMN roundtrip_p50_ms DOUBLE PRECISION,
    ADD COLUMN roundtrip_p95_ms DOUBLE PRECISION,
    ADD COLUMN roundtrip_p99_ms DOUBLE PRECISION;

CREATE TABLE turn_stats_daily (
    customer_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    bucket TIMESTAMPTZ NOT NULL,
    turn_count BIGINT NOT NULL,
    interrupted_count BIGINT NOT NULL,
    audio_out_ms_total DOUBLE PRECISION NOT NULL,
    stt_latency_p50_ms DOUBLE PRECISION,
    stt_latency_p95_ms DOUBLE PRECISION,
    llm_ttft_p50_ms DOUBLE PRECISION,
    llm_ttft_p95_ms DOUBLE PRECISION,
    tts_ttfb_p50_ms DOUBLE PRECISION,
    tts_ttfb_p95_ms DOUBLE PRECISION,
    roundtrip_p50_ms DOUBLE PRECISION,
    roundtrip_p95_ms DOUBLE PRECISION,
    roundtrip_p99_ms DOUBLE PRECISION,
    PRIMARY KEY (customer_id, agent_id, bucket)
);

CREATE TABLE call_events (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    at TIMESTAMPTZ NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    turn_id TEXT,
    participant TEXT,
    said TEXT,
    latency_ms DOUBLE PRECISION
);
CREATE INDEX call_events_call_at_idx ON call_events (call_id, at);
CREATE INDEX call_events_customer_at_idx ON call_events (customer_id, at DESC);

CREATE TABLE turns (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    call_id TEXT,
    turn_id TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL,
    stt_latency_ms DOUBLE PRECISION,
    llm_ttft_ms DOUBLE PRECISION,
    tts_ttfb_ms DOUBLE PRECISION,
    roundtrip_ms DOUBLE PRECISION,
    speech_end_to_audio_ms DOUBLE PRECISION,
    audio_out_ms DOUBLE PRECISION,
    interrupted BOOLEAN NOT NULL DEFAULT FALSE,
    audio_dropped_ms DOUBLE PRECISION,
    cadence_ms DOUBLE PRECISION,
    decision_ms DOUBLE PRECISION,
    model_to_first_text_ms DOUBLE PRECISION,
    text_to_tts_ms DOUBLE PRECISION,
    tts_to_audio_ms DOUBLE PRECISION,
    first_frame_queued_ms DOUBLE PRECISION,
    first_audible_frame_ms DOUBLE PRECISION,
    speech_end_to_audible_ms DOUBLE PRECISION
);
CREATE INDEX turns_customer_started_idx ON turns (customer_id, started_at DESC);
CREATE INDEX turns_agent_started_idx ON turns (agent_id, started_at DESC);
CREATE UNIQUE INDEX turns_agent_turn_idx ON turns (agent_id, turn_id);
