-- +goose Up

-- The Stream Chat message that carries this response's answer, in a conversation kept in
-- Chat. A rewind to the response rewinds the channel to this message. Empty for a response
-- that wrote no Chat message, and for one recorded before the id was kept.
ALTER TABLE agent_responses ADD COLUMN chat_message_id TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE agent_responses DROP COLUMN chat_message_id;
