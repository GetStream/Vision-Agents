-- +goose Up

-- image_jobs are the pictures agents were asked to draw (generate_image), drawn after the turn
-- that asked ended and posted into the conversation when ready. A row is written before the
-- tool answers, so a picture survives a router restart: a worker on any router draws and posts
-- it, with nothing of the turn left, so a row holds what drawing, billing and posting take.
--
--   - id is the job. The message it posts has the id id (a picture) or id-failed (a failure),
--     and created_at as its start, so posting it again after a crash finds it written
--     (conversation.Service.Post).
--   - customer_id, stream_app, cid and agent_user are the conversation: who it is billed to, the
--     Stream app it is kept in, its channel and the agent's Stream user, who posts.
--   - session_id, turn_id and tool_call_id are the call that asked. request_key is a digest of
--     them with the prompt and shape: one row per request, so a call run again in its turn
--     queues nothing, while a call id a later turn or a reopened session reuses (ask-1), or the
--     same call asking for another picture, is another job.
--   - agent_id, call_id, region, target, tags and attribution are what the draw is routed and
--     billed under, as the turn would have: the session's attribution, so its spend limits hold.
--   - prompt and aspect_ratio are what to draw.
--   - state is queued until the message is posted (done) or could not be (failed). On a queued
--     row next_attempt_at is when it is due; a worker that claims it moves it to the end of its
--     lease and counts the claim in attempts, so only that claim may write the row, and a router
--     that stopped mid-job leaves the row to the next once the lease ran out.
--   - draws counts the draws started; a draw the provider never took is not counted.
--     released counts the claims given back uncounted (a draw that never left its slot wait, a
--     router stopping), so the cap on claims that never finished reads attempts - released.
--   - drawn and drawn_type are the picture as drawn, kept the moment it is, so it is never drawn
--     and billed again; picture is it once uploaded, when drawn is let go of.
--   - message is what the job posts once decided: the picture's words, or why there is no
--     picture. It is decided before it is posted, so a post tried again posts the same message.
--     failed says which of the two it is. posts counts the posts that failed, for their backoff.
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
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL DEFAULT '',
    tool_call_id TEXT NOT NULL,
    request_key TEXT NOT NULL UNIQUE,
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
    draws INTEGER NOT NULL DEFAULT 0,
    released INTEGER NOT NULL DEFAULT 0,
    posts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL,
    drawn BYTEA,
    drawn_type TEXT NOT NULL DEFAULT '',
    picture JSONB,
    message TEXT NOT NULL DEFAULT '',
    failed BOOLEAN NOT NULL DEFAULT false,
    outcome TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ
);

-- The queued rows by when they are due, apart for those still to be drawn and those with their
-- message decided (store.ClaimImageJobs, store.NextImageJobAt).
CREATE INDEX image_jobs_due ON image_jobs (next_attempt_at) WHERE state = 'queued';

-- +goose Down

DROP TABLE image_jobs;
