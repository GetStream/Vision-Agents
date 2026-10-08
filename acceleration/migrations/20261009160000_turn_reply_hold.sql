-- +goose Up

-- reply_hold_ms is how long the reply's audio was held for the caller to have been quiet, before
-- its first sound and before each sentence that followed a pause in it, added together. A hold
-- before the first sound is inside tts_to_audio_ms, roundtrip_ms and the playout columns; one
-- before a later sentence comes after them and is inside none. It is null where nothing was held.
ALTER TABLE turns
    ADD COLUMN reply_hold_ms DOUBLE PRECISION;

-- +goose Down
ALTER TABLE turns
    DROP COLUMN reply_hold_ms;
