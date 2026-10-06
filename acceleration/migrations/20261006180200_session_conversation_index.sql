-- +goose Up
-- A conversation with no outbox record here is read in the app its newest session held it in.
CREATE INDEX agent_sessions_conversation_idx ON agent_sessions (customer_id, conversation_id, created_at DESC);

-- +goose Down
DROP INDEX agent_sessions_conversation_idx;
