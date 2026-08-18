-- +goose Up

-- One row per conversation the service ran, so a call can be found after the process that
-- held it is gone. Sessions live in a map in memory, which answers "what is happening now"
-- and nothing at all about last Tuesday.
--
-- What was said is not duplicated here: the transcript is already in Stream Chat, keyed by
-- agent_id, and the timings are already in turns. This row is what ties them together and
-- carries the judgements made afterwards.
CREATE TABLE calls (
    -- id is the session id, which is the handle the caller already holds the call by.
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- call_id is the Stream call, and agent_id is the transcript channel.
    call_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    -- config_id names the agent config the call ran under, empty when the caller spelled
    -- the whole spec out.
    config_id TEXT,
    campaign_id TEXT,
    contact_id TEXT,
    from_number TEXT,
    to_number TEXT,
    -- direction is inbound or outbound: who rang whom.
    direction TEXT NOT NULL DEFAULT 'inbound',
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- ended_at is null while the call is still running, which is how the running ones are
    -- found after a restart.
    ended_at TIMESTAMPTZ,
    -- summary, review_score and review_notes are written after the call by a short model
    -- pass over the transcript, so they are null until it has run.
    summary TEXT,
    review_score INTEGER,
    review_notes TEXT,
    tags JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX calls_customer_idx ON calls (customer_id, started_at DESC);
CREATE INDEX calls_agent_idx ON calls (agent_id, started_at DESC);
CREATE INDEX calls_campaign_idx ON calls (campaign_id, started_at DESC)
    WHERE campaign_id IS NOT NULL;
CREATE INDEX calls_tags_idx ON calls USING GIN (tags);

-- +goose Down
DROP TABLE calls;
