-- +goose Up

-- A router config is what an agent config is for a caller that routes one modality at a
-- time: the target, the language and every per-modality option, decided once and named.
--
-- It is a separate table from agent_configs because it configures a different thing. An
-- agent config describes a conversation - a voice, instructions, a greeting, the skills
-- the subagent may be handed. This describes transcribing, speaking, answering and
-- searching on their own, with no conversation behind them.
--
-- Each block is one JSONB column rather than a column per option because nothing queries
-- inside them: a config is read whole by whoever is about to route with it, and the
-- options are the ones the modality has, which change as providers do.
CREATE TABLE router_configs (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    stt JSONB NOT NULL DEFAULT '{}',
    tts JSONB NOT NULL DEFAULT '{}',
    llm JSONB NOT NULL DEFAULT '{}',
    search JSONB NOT NULL DEFAULT '{}',
    -- tags are cost labels carried onto every request made under this config.
    tags JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

-- Two configs a customer can still reach cannot share a name, but the name is free again
-- once the config holding it is deleted.
CREATE UNIQUE INDEX router_configs_name_idx ON router_configs (customer_id, name)
    WHERE deleted_at IS NULL;
CREATE INDEX router_configs_customer_idx ON router_configs (customer_id, created_at DESC);

-- A recording is one non-realtime job: audio to transcribe or a text to speak. It is a
-- row rather than something held in memory because it outlives the request that created
-- it - the caller is handed an id and comes back for the result, and an hour of audio is
-- minutes of work.
--
-- There is no deleted_at. A finished job is a result somebody asked for and will read
-- once, not a configuration other rows point at.
CREATE TABLE recordings (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- modality is stt for a transcription and tts for a recorded voice.
    modality TEXT NOT NULL,
    status TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    -- source is the URL of the audio. Audio sent inline is not kept: it was already
    -- handed to a provider, and keeping a copy of somebody's recording is a decision
    -- nobody asked us to make.
    source TEXT,
    -- text is what to say, for a voice job.
    text TEXT,
    stt JSONB NOT NULL DEFAULT '{}',
    tts JSONB NOT NULL DEFAULT '{}',
    -- callback is a URL the finished job is POSTed to. Empty means the caller polls.
    callback TEXT,
    tags JSONB NOT NULL DEFAULT '{}',
    -- result is the finished transcript, or the finished audio, whole. A voice job keeps
    -- its audio here because there is nowhere else a caller could come back for it; a
    -- deployment that stores audio of its own puts the location in there instead.
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX recordings_customer_idx ON recordings (customer_id, created_at DESC);

-- +goose Down
DROP TABLE recordings;
DROP TABLE router_configs;
