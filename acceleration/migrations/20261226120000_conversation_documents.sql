-- +goose Up

-- conversation_documents holds the files a person attached to a conversation of theirs, read as
-- text once, when they arrived, so every reply after and a conversation carried on or forked
-- read the text rather than the file again. They are a session's, so they live on the shard
-- beside agent_responses, and are deleted with the session (store.DeleteSession). No foreign
-- key: the session's row is written by its recorder in the background, and may land after the
-- first file. owner_user_id is the person whose conversation it is: a session another user
-- opens never reads one. conversation_id is the Chat channel the conversation is kept in,
-- empty for one kept nowhere. A new table: the previous release never reads it.
CREATE TABLE conversation_documents (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL DEFAULT '',
    response_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    pages INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL,
    markdown TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX conversation_documents_session_idx ON conversation_documents (customer_id, session_id, created_at);
CREATE INDEX conversation_documents_conversation_idx ON conversation_documents (customer_id, conversation_id, created_at)
    WHERE conversation_id <> '';

-- +goose Down
DROP TABLE conversation_documents;
