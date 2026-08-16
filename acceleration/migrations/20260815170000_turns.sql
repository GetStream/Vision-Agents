-- +goose Up

-- A request row measures one provider call. A turn measures what the caller felt: the
-- gap between finishing a sentence and hearing the answer start. The legs are separate
-- columns so a slow turn can be attributed to transcription, the model or the voice.
--
-- Every leg is nullable because not every pipeline has all of them: a realtime model
-- that hears and speaks for itself fills roundtrip_ms and nothing else, and an
-- interrupted turn never reaches the ones that come after the interruption.
CREATE TABLE turns (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    call_id TEXT,
    turn_id TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL,
    -- stt_latency_ms is the provider's own decode time for the settling transcript.
    stt_latency_ms DOUBLE PRECISION,
    -- llm_ttft_ms is the wait between asking the model and its first token.
    llm_ttft_ms DOUBLE PRECISION,
    -- tts_ttfb_ms is the wait between sending the first sentence and the first audio.
    tts_ttfb_ms DOUBLE PRECISION,
    -- roundtrip_ms is the whole delay: settled transcript to first audio published.
    roundtrip_ms DOUBLE PRECISION,
    -- speech_end_to_audio_ms is voice in to voice out, which is roundtrip plus the time
    -- the provider spent settling the turn after the speaker stopped.
    speech_end_to_audio_ms DOUBLE PRECISION,
    -- audio_out_ms is how much speech the agent published for this turn.
    audio_out_ms DOUBLE PRECISION,
    interrupted BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX turns_customer_started_idx ON turns (customer_id, started_at DESC);
CREATE INDEX turns_agent_started_idx ON turns (agent_id, started_at DESC);
CREATE UNIQUE INDEX turns_agent_turn_idx ON turns (agent_id, turn_id);

CREATE TABLE turn_stats_hourly (
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

-- +goose Down
DROP TABLE turn_stats_daily;
DROP TABLE turn_stats_hourly;
DROP TABLE turns;
