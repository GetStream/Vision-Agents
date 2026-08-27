-- +goose Up

-- A turns row says what a call cost the caller in waiting. It cannot say why the call
-- went the way it did: why the agent waited, why it answered something it was not asked,
-- why it stopped mid-sentence. Those are judgements the conversation made, and until now
-- they only ever reached a debug log nobody had switched on.
--
-- One row is one judgement, with the reason in words. Rows are written off the
-- conversation's path and are allowed to be lost under load: a call that ran is worth
-- more than a complete account of one.
CREATE TABLE call_events (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- call_id is the session the judgement was made in, which is the handle a caller
    -- already holds the call by.
    call_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    at TIMESTAMPTZ NOT NULL,
    -- kind is what was decided: ask, wait, ignore, answer, queue, interrupt, shorten,
    -- backchannel, supersede, compact, delegate or fail.
    kind TEXT NOT NULL,
    -- reason is why, in the words the conversation used to explain itself.
    reason TEXT NOT NULL,
    -- turn_id is the exchange the judgement was about, so a decision can be lined up
    -- against the timings of the turn it produced.
    turn_id TEXT,
    participant TEXT,
    -- said is what was heard, or what the agent decided to say.
    said TEXT,
    -- latency_ms is what the flow controller took to rule, where anything was asked.
    latency_ms DOUBLE PRECISION
);

CREATE INDEX call_events_call_at_idx ON call_events (call_id, at);
CREATE INDEX call_events_customer_at_idx ON call_events (customer_id, at DESC);

-- +goose Down
DROP TABLE call_events;
