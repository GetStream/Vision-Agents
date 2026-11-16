-- +goose Up

-- One row per session a caller opened, so "what did we talk about last Tuesday" has an
-- answer. Sessions live in a map in memory, which says what is happening now and nothing
-- at all about what happened; calls answers the same question for telephony, but a call row
-- is a call, keyed by the Stream call it joined and carrying the review a call gets
-- afterwards. A text conversation somebody opens from a page is neither, and conflating the
-- two would mean every browser conversation pretending to be a phone call.
--
-- What was said is still not duplicated here. The transcript is in Stream Chat, keyed by
-- conversation_id, and the per-turn timings are in turns. This row is the handle: it says
-- which agent, whose it is, what it was called, and what the caller labelled it with.
CREATE TABLE agent_sessions (
    -- id is the session id the caller already holds the session by, so a row appears under
    -- the same name the socket and every REST call use.
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- config_id names the stored agent config, and agent_name the name that config was
    -- found by. The name is denormalised because it is what a caller filters on -- they
    -- asked for "docs", not for a config id they never saw -- and because a config that is
    -- later renamed must not silently rewrite what old sessions were opened against.
    config_id TEXT,
    agent_name TEXT NOT NULL DEFAULT '',
    -- agent_id is the transcript channel, conversation_id the same thing written as a CID.
    agent_id TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    -- user_id is whose session it is, and caller_kind says what that name is worth: a
    -- verified user, a guest, or a caller that went by no name at all. Both are kept
    -- because filtering on the name alone would let an anonymous caller claim another's
    -- sessions by guessing their user id.
    user_id TEXT NOT NULL DEFAULT '',
    caller_kind TEXT NOT NULL DEFAULT '',
    -- title and description are the caller's own, for a list a person reads.
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    -- project is what the conversation belongs to. It is a column as well as a cost tag:
    -- billing wants it in tags, and a sidebar grouped by project wants to filter on it
    -- without unpacking JSON on every row.
    project TEXT NOT NULL DEFAULT '',
    -- custom is whatever the caller wants to remember about the session, handed back
    -- untouched. Never read by the router: a field it interpreted would be a field it
    -- could break.
    custom JSONB NOT NULL DEFAULT '{}',
    -- model_overwrites is what the caller asked to change about the models for this
    -- session, kept so a fork can inherit it and a list can show it.
    model_overwrites JSONB NOT NULL DEFAULT '{}',
    call_id TEXT,
    call_type TEXT,
    -- forked_from is the session this one continued from, null for one opened fresh. No
    -- foreign key: the parent may be deleted while its children are still worth reading,
    -- and a fork is a record of where a conversation came from rather than a dependency
    -- on it still existing.
    forked_from TEXT,
    state TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- closed_at is null while the session is still running, which is how the live ones are
    -- found again after a restart.
    closed_at TIMESTAMPTZ,
    last_response_at TIMESTAMPTZ,
    -- searchable is the text search() reads. Title and description are weighted above the
    -- project and the agent name, so searching "billing" finds the conversation called
    -- billing before every conversation in the billing project.
    searchable tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(project, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(agent_name, '')), 'D')
    ) STORED
);

-- Every list is a customer's, newest first. The user index carries the common case of a
-- page asking for one person's conversations; the config one carries a dashboard asking
-- for one agent's.
CREATE INDEX agent_sessions_customer_idx ON agent_sessions (customer_id, created_at DESC);
CREATE INDEX agent_sessions_user_idx ON agent_sessions (customer_id, user_id, created_at DESC);
CREATE INDEX agent_sessions_config_idx ON agent_sessions (customer_id, config_id, created_at DESC);
CREATE INDEX agent_sessions_agent_idx ON agent_sessions (customer_id, agent_name, created_at DESC);
CREATE INDEX agent_sessions_running_idx ON agent_sessions (customer_id, created_at DESC)
    WHERE closed_at IS NULL;
CREATE INDEX agent_sessions_search_idx ON agent_sessions USING GIN (searchable);
CREATE INDEX agent_sessions_custom_idx ON agent_sessions USING GIN (custom);

-- One row per turn the agent took, so a conversation can be replayed without the socket
-- that heard it. The transcript in Stream Chat is the rendered answer; this is what the
-- session did to arrive at it, which is what items.unwind() hands back.
CREATE TABLE agent_responses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL,
    -- said is what the person asked, which is the first item of every response and worth
    -- having on the row so a list of responses reads without loading their items.
    said TEXT NOT NULL DEFAULT '',
    -- status is running, completed, failed, cancelled: a response interrupted halfway is
    -- different from one that finished and different again from one that never started.
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX agent_responses_session_idx ON agent_responses (session_id, created_at ASC);

-- The items a response was made of, in the order they happened. Deltas are not kept: a
-- hundred fragments of one sentence are the sentence, and keeping them would make the
-- table mostly punctuation. What is kept is what a reader would want back -- the question,
-- the tool calls and what they returned, the answer, and whatever went wrong.
CREATE TABLE agent_response_items (
    response_id TEXT NOT NULL REFERENCES agent_responses(id) ON DELETE CASCADE,
    -- ordinal is the position within the response, assigned by the writer rather than by a
    -- sequence, so items written in a batch keep the order they were produced in.
    ordinal INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    -- kind is said, thought, tool_call, tool_result, answer, error, blocked.
    kind TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}',
    at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (response_id, ordinal)
);

CREATE INDEX agent_response_items_session_idx ON agent_response_items (session_id, at ASC);

-- +goose Down
DROP TABLE agent_response_items;
DROP TABLE agent_responses;
DROP TABLE agent_sessions;
