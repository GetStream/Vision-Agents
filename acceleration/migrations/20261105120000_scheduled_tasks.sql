-- +goose Up

-- A scheduled task is something the agent agreed to do for one end user at a
-- wall-clock time in that user's zone, once or on an RRULE repeat. The worker
-- on every router claims due rows from here; scheduled_task_runs is the record
-- of each occurrence it claimed. No foreign keys to sessions, configs or
-- users: a task is a record of an agreement, and the rows it names may be
-- deleted while the agreement is still worth reading (the run then fails and
-- the task pauses, rather than the row vanishing).
CREATE TABLE scheduled_tasks (
    id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    -- config_id is the agent config the task runs on, as it is when it fires.
    config_id TEXT NOT NULL DEFAULT '',
    -- user_id is whose task it is, and caller_kind what that name is worth,
    -- the pair a session is owned by.
    user_id TEXT NOT NULL,
    caller_kind TEXT NOT NULL DEFAULT '',
    -- origin_session_id is the session the task was created in, session_id the
    -- session it follows up in (the same one until a backend-created task
    -- opens one at its first occurrence), conversation_id that session's Chat
    -- conversation.
    origin_session_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    -- instructions are what the agent is told when the task fires. The owner
    -- can read them; they are never a secret from the user they act for.
    instructions TEXT NOT NULL,
    -- start_local is the first occurrence as the user's clock reads it, a
    -- timestamp with no zone; timezone says which clock.
    start_local TIMESTAMP NOT NULL,
    timezone TEXT NOT NULL,
    -- rrule is the normalised repeat (FREQ, INTERVAL, BYDAY, BYMONTHDAY,
    -- COUNT), null for a task that runs once. until_date is the last local
    -- date it repeats on.
    rrule TEXT,
    until_date DATE,
    -- next_run_at is the next occurrence as an instant, in UTC. Null when the
    -- task has none left or is not counting (paused).
    next_run_at TIMESTAMPTZ,
    state TEXT NOT NULL DEFAULT 'active',
    -- channels is where an occurrence is delivered, in order, as JSON list of
    -- channel names (conversation, sms, whatsapp, imessage, phone).
    channels JSONB NOT NULL DEFAULT '["conversation"]',
    custom JSONB NOT NULL DEFAULT '{}',
    -- quiet_hours is the span of the user's local day an occurrence is held
    -- through rather than delivered into, as {start, end} in "22:00" form, read
    -- in the task's own zone. Null is no hold.
    quiet_hours JSONB,
    -- lease_until and claim_id are the worker's claim: claim_id fences a late
    -- finish from advancing a schedule somebody changed under it.
    lease_until TIMESTAMPTZ,
    claim_id TEXT NOT NULL DEFAULT '',
    consecutive_failures INT NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL DEFAULT 'api',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, id)
);

CREATE INDEX scheduled_tasks_due_idx ON scheduled_tasks (next_run_at)
    WHERE state = 'active';
CREATE INDEX scheduled_tasks_user_idx ON scheduled_tasks (customer_id, user_id, created_at DESC, id DESC);
CREATE INDEX scheduled_tasks_session_idx ON scheduled_tasks (customer_id, session_id)
    WHERE session_id <> '';
-- Deleting a session deletes the tasks made in it as well as those that follow
-- up in it (DeleteScheduledTasksForSession).
CREATE INDEX scheduled_tasks_origin_session_idx ON scheduled_tasks (customer_id, origin_session_id)
    WHERE origin_session_id <> '';

CREATE TABLE scheduled_task_runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    -- scheduled_for is the occurrence this run is, as an instant.
    scheduled_for TIMESTAMPTZ NOT NULL,
    -- trigger is schedule for an occurrence that came due, manual for /run.
    trigger TEXT NOT NULL DEFAULT 'schedule',
    -- prompt is the task's instructions as they were when the occurrence was
    -- claimed, so a retry says the same thing even if the task was edited.
    prompt TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'running',
    delivered_to TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    attempts INT NOT NULL DEFAULT 0,
    error TEXT,
    -- postponed_until is when the hold on a run ends, recorded when the worker
    -- holds the occurrence (quiet hours), so when the task is claimed again the
    -- occurrence's age is counted from the hold's end and not from when it came
    -- due: a reminder held overnight is delivered in the morning, not missed.
    postponed_until TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (customer_id, task_id)
        REFERENCES scheduled_tasks (customer_id, id) ON DELETE CASCADE
);

-- One run per occurrence: the second claim of the same occurrence finds this
-- row and knows the occurrence is spoken for.
CREATE UNIQUE INDEX scheduled_task_runs_occurrence_idx
    ON scheduled_task_runs (customer_id, task_id, scheduled_for)
    WHERE trigger = 'schedule';
CREATE INDEX scheduled_task_runs_task_idx
    ON scheduled_task_runs (customer_id, task_id, created_at DESC, id DESC);

-- timezone is the zone the user was last explicitly known to be in, which a
-- scheduled task resolves to when nothing nearer names one.
ALTER TABLE users ADD COLUMN timezone TEXT;

-- scheduling is an agent config's scheduled-task settings:
-- {enabled, max_tasks_per_user, default_timezone}. Empty for every config
-- there is, which is scheduling off.
ALTER TABLE agent_configs ADD COLUMN scheduling JSONB NOT NULL DEFAULT '{}';

-- from_address is who a delivered message came from. A line's deliveries
-- were only ever its dedup record; with it, the last time a person wrote is
-- readable, which is what WhatsApp's 24-hour window is counted from.
ALTER TABLE channel_deliveries ADD COLUMN from_address TEXT NOT NULL DEFAULT '';
-- LastChannelMessageAt reads the newest delivery from one address on one line.
CREATE INDEX channel_deliveries_from_idx
    ON channel_deliveries (account_id, from_address, received_at DESC);

-- thread is where a message is sent to reach this person again: the chat,
-- for iMessage, whose replies are posted to a conversation rather than to a
-- person. Empty for the channels that reply to the address itself.
ALTER TABLE channel_identities ADD COLUMN thread TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE channel_identities DROP COLUMN thread;
DROP INDEX channel_deliveries_from_idx;
ALTER TABLE channel_deliveries DROP COLUMN from_address;
ALTER TABLE agent_configs DROP COLUMN scheduling;
ALTER TABLE users DROP COLUMN timezone;
DROP TABLE scheduled_task_runs;
DROP TABLE scheduled_tasks;
