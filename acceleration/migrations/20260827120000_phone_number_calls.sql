-- +goose Up

-- A number that has been attached reaches a Stream call, and an inbound call arriving over
-- the webhook names that call and nothing else. Without these columns there is no way back
-- from "call phone-+15125551234 started" to the customer who holds the number, because the
-- webhook is app-global and carries no customer of its own.
ALTER TABLE phone_numbers ADD COLUMN stream_call_id TEXT;
ALTER TABLE phone_numbers ADD COLUMN stream_call_type TEXT;

-- Looking a number up by the call it lands in is what receiving a webhook does, on every
-- inbound call, so it is an index rather than a scan.
CREATE INDEX phone_numbers_call_idx ON phone_numbers (stream_call_type, stream_call_id)
    WHERE released_at IS NULL;

-- +goose Down
DROP INDEX phone_numbers_call_idx;
ALTER TABLE phone_numbers DROP COLUMN stream_call_type;
ALTER TABLE phone_numbers DROP COLUMN stream_call_id;
