-- +goose Up

-- A simulation is a conversation to have with an agent and something that has to be true
-- at the end of it. It is stored rather than held in memory because the point of it is to
-- be run again after the agent changes: the same question, asked of a newer agent.
CREATE TABLE simulations (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    -- mode is text or audio. Audio generates speech and runs the whole pipeline; text
    -- hands the agent the words and tests everything between hearing and answering.
    mode TEXT NOT NULL DEFAULT 'text',
    -- config_id is the agent being tested.
    config_id TEXT NOT NULL DEFAULT '',
    -- scenario is what to ask, in the customer's own words and over as many turns as it
    -- takes. It is a brief for the caller rather than a script: a scenario that says to
    -- change your mind once the order is handled only means anything to somebody reading
    -- the replies.
    scenario TEXT NOT NULL DEFAULT '',
    -- assertion is what has to be true at the end for the run to have passed.
    assertion TEXT NOT NULL DEFAULT '',
    -- variations is how many ways of asking the same thing one run tries. 1 is the
    -- scenario as written, and it is always the first of them.
    variations INTEGER NOT NULL DEFAULT 1,
    -- judge_target and caller_target are routing targets like any other. Empty takes the
    -- default, which for the judge is a quality tier rather than a fast one: nobody is
    -- waiting for it, and the point of it is the judgement.
    judge_target TEXT NOT NULL DEFAULT '',
    caller_target TEXT NOT NULL DEFAULT '',
    -- caller_tts, caller_stt and caller_voice are how the caller speaks and listens in an
    -- audio simulation, and mean nothing in a text one.
    caller_tts TEXT NOT NULL DEFAULT '',
    caller_stt TEXT NOT NULL DEFAULT '',
    caller_voice TEXT NOT NULL DEFAULT '',
    -- max_turns bounds one conversation, since a caller that never decides it is finished
    -- would otherwise talk until the clock stopped it.
    max_turns INTEGER NOT NULL DEFAULT 12,
    tags JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX simulations_customer_idx ON simulations (customer_id, created_at DESC)
    WHERE deleted_at IS NULL;

-- One press of Run. It is the parent of however many conversations the variations asked
-- for, and it is what the log of runs lists.
--
-- What was run is copied onto the row rather than referenced, because editing a simulation
-- must not rewrite what an old run tested.
CREATE TABLE simulation_runs (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    simulation_id TEXT NOT NULL REFERENCES simulations (id) ON DELETE CASCADE,
    -- state is running, passed, failed, cancelled or errored. A run passed only if every
    -- one of its cases did.
    state TEXT NOT NULL DEFAULT 'running',
    -- The tally, so the log can list a run without reading its conversations.
    cases INTEGER NOT NULL DEFAULT 0,
    passed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT '',
    config_id TEXT NOT NULL DEFAULT '',
    scenario TEXT NOT NULL DEFAULT '',
    assertion TEXT NOT NULL DEFAULT '',
    judge_target TEXT NOT NULL DEFAULT '',
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX simulation_runs_customer_idx ON simulation_runs (customer_id, started_at DESC);
CREATE INDEX simulation_runs_simulation_idx ON simulation_runs (simulation_id, started_at DESC);

-- One conversation. With variations off a run has one of these; with them expanded, ten,
-- each asking the same thing a different way.
CREATE TABLE simulation_cases (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES simulation_runs (id) ON DELETE CASCADE,
    -- variation is which way of asking this was, and the order they are listed in.
    variation INTEGER NOT NULL DEFAULT 0,
    -- scenario is the wording this case used, which for the first is the simulation's own.
    scenario TEXT NOT NULL DEFAULT '',
    -- state is pending, running, passed, failed, errored or cancelled.
    state TEXT NOT NULL DEFAULT 'pending',
    -- call_id is the session that held the conversation, written as soon as it exists so a
    -- run in progress can be watched on the paths a call is already watched on.
    call_id TEXT,
    -- transcript is what was said, oldest first.
    transcript JSONB NOT NULL DEFAULT '[]',
    turns INTEGER NOT NULL DEFAULT 0,
    -- passed, verdict and score are the judge's ruling. Null means it never got to rule,
    -- which is not the same as having ruled against.
    passed BOOLEAN,
    verdict TEXT,
    score INTEGER,
    -- ended is why the conversation stopped: complete, turns, timeout or failed.
    ended TEXT NOT NULL DEFAULT '',
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX simulation_cases_run_idx ON simulation_cases (run_id, variation);

-- +goose Down
DROP TABLE simulation_cases;
DROP TABLE simulation_runs;
DROP TABLE simulations;
