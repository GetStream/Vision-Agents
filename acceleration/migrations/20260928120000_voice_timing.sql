-- +goose Up
ALTER TABLE turns
    ADD COLUMN cadence_ms DOUBLE PRECISION,
    ADD COLUMN decision_ms DOUBLE PRECISION,
    ADD COLUMN model_to_first_text_ms DOUBLE PRECISION,
    ADD COLUMN text_to_tts_ms DOUBLE PRECISION,
    ADD COLUMN tts_to_audio_ms DOUBLE PRECISION;

ALTER TABLE requests
    ADD COLUMN operation_id TEXT,
    ADD COLUMN purpose TEXT,
    ADD COLUMN turn_id TEXT,
    ADD COLUMN duration_ms DOUBLE PRECISION;

-- +goose Down
ALTER TABLE requests
    DROP COLUMN operation_id,
    DROP COLUMN purpose,
    DROP COLUMN turn_id,
    DROP COLUMN duration_ms;

ALTER TABLE turns
    DROP COLUMN cadence_ms,
    DROP COLUMN decision_ms,
    DROP COLUMN model_to_first_text_ms,
    DROP COLUMN text_to_tts_ms,
    DROP COLUMN tts_to_audio_ms;
