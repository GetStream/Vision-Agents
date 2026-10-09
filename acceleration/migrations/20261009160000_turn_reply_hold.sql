-- +goose Up

-- First-audio silence hold, included in tts_to_audio_ms and roundtrip_ms.
-- Null when nothing was held.
ALTER TABLE turns
    ADD COLUMN reply_hold_ms DOUBLE PRECISION;

-- +goose Down
ALTER TABLE turns
    DROP COLUMN reply_hold_ms;
