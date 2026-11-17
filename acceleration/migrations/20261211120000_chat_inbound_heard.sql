-- +goose Up

-- heard is what the voice recording sent with a queued message said, kept once it was heard
-- (store.KeepChatInboundHeard), so a try after one nothing took, on this router or another,
-- hands on what was heard rather than hearing it, and billing it, again. NULL until then, and
-- for a message with no recording.
ALTER TABLE chat_inbound ADD COLUMN IF NOT EXISTS heard TEXT;

-- +goose Down

ALTER TABLE chat_inbound DROP COLUMN IF EXISTS heard;
