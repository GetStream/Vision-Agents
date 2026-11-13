-- +goose Up

-- image_jobs are the pictures agents were asked to draw (generate_image), drawn after the turn
-- that asked ended and posted into the conversation when ready. A row is written before the
-- tool answers, so a picture survives a router restart: a worker on any router draws and posts
-- it, with nothing of the turn left, so a row holds what drawing, billing and posting take.
--
--   - id is the job, and the id of the one message it posts: the picture or, when it could not
--     be drawn, the failure. created_at is that message's start, so posting it again after a
--     crash finds it written (conversation.Service.Post).
--   - customer_id, stream_app, cid and agent_user are the conversation: who it is billed to, the
--     Stream app it is kept in, its channel and the agent's Stream user, who posts.
--   - tool_call_id is the call that asked: one row per call in a channel, so a call run again
--     queues nothing.
--   - agent_id, call_id, region, target, tags and attribution are what the draw is routed and
--     billed under, as the turn would have: the session's attribution, so its spend limits hold.
--   - prompt and aspect_ratio are what to draw.
--   - state is queued until the message is posted (done) or could not be (failed). On a queued
--     row next_attempt_at is when it is due; a worker that claims it moves it to the end of its
--     lease and counts the claim in attempts, so a router that stopped mid-draw leaves the row to
--     the next once the lease ran out.
--   - picture is the uploaded picture once drawn, so a later attempt posts it rather than drawing
--     and billing it again. outcome is what the failure message said.
--   - Done rows are kept a day and failed ones 30 days: they hold the prompt
--     (store.PruneImageJobs).
--
-- A data move does not carry these rows (store.dataTables): they are in flight for minutes.
CREATE TABLE image_jobs (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    stream_app BIGINT NOT NULL DEFAULT 0,
    cid TEXT NOT NULL,
    agent_user TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT '',
    call_id TEXT NOT NULL DEFAULT '',
    region TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL,
    tags JSONB,
    attribution JSONB NOT NULL,
    prompt TEXT NOT NULL,
    aspect_ratio TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'queued' CHECK (state IN ('queued', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL,
    picture JSONB,
    outcome TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    UNIQUE (cid, tool_call_id)
);

-- The queued rows by when they are due (store.ClaimImageJobs, store.NextImageJobAt).
CREATE INDEX image_jobs_due ON image_jobs (next_attempt_at) WHERE state = 'queued';

-- +goose Down

DROP TABLE image_jobs;
