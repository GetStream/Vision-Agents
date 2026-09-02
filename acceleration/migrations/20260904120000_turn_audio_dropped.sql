-- +goose Up

-- audio_dropped_ms is speech that was synthesised for a turn, and billed for, but never
-- reached the participant, because the turn had been abandoned by the time it arrived.
--
-- On an interrupted turn this is the ordinary cost of barge-in. On a turn that was not
-- interrupted it means the agent talked over itself and cut its own sentence off, which
-- roundtrip_ms cannot show: that only times the first chunk, so a reply whose tail was
-- thrown away still looks like it was answered promptly.
ALTER TABLE turns ADD COLUMN audio_dropped_ms DOUBLE PRECISION;

-- +goose Down
ALTER TABLE turns DROP COLUMN audio_dropped_ms;
