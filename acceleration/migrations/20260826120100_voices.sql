-- +goose Up

-- A voice a customer brought with them: recordings of somebody speaking, and whatever each
-- text-to-speech provider made of them.
--
-- It is stored rather than named directly because the router fails over between providers
-- mid-call. A raw provider voice id would be meaningless to the provider it failed over
-- to, so what a session asks for is this row, and which id that means is worked out once
-- the provider is chosen.
CREATE TABLE voices (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

-- Two voices a customer can still reach cannot share a name, but the name is free again
-- once the voice holding it is deleted.
CREATE UNIQUE INDEX voices_name_idx ON voices (customer_id, name)
    WHERE deleted_at IS NULL;
CREATE INDEX voices_customer_idx ON voices (customer_id, created_at DESC);

-- One recording. The audio lives in object storage, because a database is the wrong place
-- for a minute of speech, so the row holds where it went rather than what it holds.
CREATE TABLE voice_samples (
    id TEXT PRIMARY KEY,
    voice_id TEXT NOT NULL REFERENCES voices (id) ON DELETE CASCADE,
    object_key TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT '',
    bytes BIGINT NOT NULL DEFAULT 0,
    -- transcript is what is said in the recording, which the providers that ask for one
    -- clone more faithfully with.
    transcript TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX voice_samples_voice_idx ON voice_samples (voice_id, created_at);

-- What one provider made of the samples. A voice prepared with two providers has two of
-- these, which is what lets the router move between them without changing voice.
CREATE TABLE voice_bindings (
    id TEXT PRIMARY KEY,
    voice_id TEXT NOT NULL REFERENCES voices (id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    -- external_id is what the provider calls this voice, and what a session asks it for.
    external_id TEXT NOT NULL DEFAULT '',
    -- state is pending, ready or failed. Only a ready binding may be spoken in.
    state TEXT NOT NULL DEFAULT 'pending',
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Preparing the same voice with the same provider twice replaces what was there, rather
-- than leaving two answers to which id to use.
CREATE UNIQUE INDEX voice_bindings_provider_idx ON voice_bindings (voice_id, provider);

-- +goose Down
DROP TABLE voice_bindings;
DROP TABLE voice_samples;
DROP TABLE voices;
