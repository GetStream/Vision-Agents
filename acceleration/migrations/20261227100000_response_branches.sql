-- +goose NO TRANSACTION
-- +goose Up

-- parent_response_id is the response before this one on the path it was answered on, NULL for
-- a session's first. question_response_id is the response whose question this one answers: its
-- own id for a new question or an edit, the replaced response's for a regenerate. Responses
-- sharing a parent are alternatives to one another. Rows from before have neither, which keeps
-- them out of every set of alternatives. Both are nullable with no default, which rewrites no
-- row; the lock waits at most 5 s behind a running query, past which the migration fails and
-- the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_responses ADD COLUMN IF NOT EXISTS parent_response_id TEXT;
    ALTER TABLE agent_responses ADD COLUMN IF NOT EXISTS question_response_id TEXT;
END $$;
-- +goose StatementEnd

-- Built without stopping the writes of responses. A build that fails leaves an invalid index
-- that IF NOT EXISTS skips, and DROP INDEX CONCURRENTLY cannot run in the DO block that would
-- check for one, so every Migrate drops and builds it again in Go (store.usageIndexes).
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_responses_parent_idx
    ON agent_responses (session_id, parent_response_id);

-- agent_branch_switches is one move of a conversation to another branch, kept until it is
-- done so a router that stopped partway finishes it on the session's next open. hidden_from
-- is when the message the channel was rewound to was created; with created_at it bounds the
-- messages the rewind hid. No foreign key to agent_responses: a session's responses can be in
-- another shard's database.
CREATE TABLE IF NOT EXISTS agent_branch_switches (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES agent_sessions (id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('regenerate', 'edit', 'activate')),
    target_response_id TEXT NOT NULL,
    fork_message_id TEXT NOT NULL,
    inclusive BOOLEAN NOT NULL,
    hidden_from TIMESTAMPTZ NOT NULL,
    state TEXT NOT NULL DEFAULT 'snapshotted' CHECK (state IN ('snapshotted', 'rewound', 'resending', 'done')),
    resent INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_branch_switches_unfinished_idx
    ON agent_branch_switches (session_id, created_at)
    WHERE state <> 'done';

CREATE INDEX IF NOT EXISTS agent_branch_switches_session_idx
    ON agent_branch_switches (session_id, hidden_from);

-- agent_branch_messages is a copy of a Stream Chat message a rewind hid, so the branch it was
-- on can be shown and sent again. message is what the message said: its text, attachments
-- and custom data.
CREATE TABLE IF NOT EXISTS agent_branch_messages (
    session_id TEXT NOT NULL REFERENCES agent_sessions (id) ON DELETE CASCADE,
    original_message_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    response_id TEXT NOT NULL DEFAULT '',
    switch_id TEXT NOT NULL,
    chat_created_at TIMESTAMPTZ NOT NULL,
    author_id TEXT NOT NULL,
    message JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (session_id, original_message_id)
);

CREATE INDEX IF NOT EXISTS agent_branch_messages_response_idx
    ON agent_branch_messages (session_id, response_id);

-- +goose Down
DROP TABLE IF EXISTS agent_branch_messages;
DROP TABLE IF EXISTS agent_branch_switches;
DROP INDEX CONCURRENTLY IF EXISTS agent_responses_parent_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_responses DROP COLUMN IF EXISTS question_response_id;
    ALTER TABLE agent_responses DROP COLUMN IF EXISTS parent_response_id;
END $$;
-- +goose StatementEnd
