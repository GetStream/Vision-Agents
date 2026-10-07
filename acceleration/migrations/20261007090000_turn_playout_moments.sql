-- +goose Up

-- roundtrip_ms and speech_end_to_audio_ms stop at the moment publishing the first chunk of a
-- reply returned. For a chunk longer than the outgoing queue that is later than the reply
-- began to be heard, by the part of the chunk that did not fit.
--
-- first_frame_queued_ms is when the first frame of the reply was queued for the outgoing
-- track, and first_audible_frame_ms when the track took the first frame that was not
-- silence, both from the last transcript revision. speech_end_to_audible_ms is
-- speech_end_to_audio_ms measured to the second of them. They are null where the edge does
-- not report them.
--
-- reply_hold_ms is how long the first audio of the reply was held for the caller to have been
-- quiet. It is inside tts_to_audio_ms, roundtrip_ms and the columns above, and null where the
-- reply was not held.
ALTER TABLE turns
    ADD COLUMN first_frame_queued_ms DOUBLE PRECISION,
    ADD COLUMN first_audible_frame_ms DOUBLE PRECISION,
    ADD COLUMN speech_end_to_audible_ms DOUBLE PRECISION,
    ADD COLUMN reply_hold_ms DOUBLE PRECISION;

-- +goose Down
ALTER TABLE turns
    DROP COLUMN first_frame_queued_ms,
    DROP COLUMN first_audible_frame_ms,
    DROP COLUMN speech_end_to_audible_ms,
    DROP COLUMN reply_hold_ms;
