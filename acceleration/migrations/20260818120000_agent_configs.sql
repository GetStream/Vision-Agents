-- +goose Up

-- An agent config is a named set of the decisions a session is created with, so a caller
-- can point at one by id rather than restating the whole spec every time a call starts.
--
-- Deleting one is a timestamp rather than a delete because calls that already ran under a
-- config still name it, and a call nobody can say what the agent was configured as is
-- worth less than the row costs to keep.
CREATE TABLE agent_configs (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    -- The routing targets, each a provider/model or a capability shortcut. Empty means
    -- the session default, so a config only has to say what it wants changed.
    stt TEXT NOT NULL DEFAULT '',
    tts TEXT NOT NULL DEFAULT '',
    voice TEXT NOT NULL DEFAULT '',
    llm TEXT NOT NULL DEFAULT '',
    -- subagent is the model that does the thinking. Empty means skills mean nothing.
    subagent TEXT NOT NULL DEFAULT '',
    instructions TEXT NOT NULL DEFAULT '',
    greeting TEXT NOT NULL DEFAULT '',
    -- skills names entries in the skill registry rather than carrying their instructions,
    -- so editing a skill changes every config that uses it.
    skills JSONB NOT NULL DEFAULT '[]',
    -- knowledge_namespace is what the agent may look things up in. Empty means it knows
    -- only what it was told.
    knowledge_namespace TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

-- Two configs a customer can still reach cannot share a name, but the name is free again
-- once the config holding it is deleted.
CREATE UNIQUE INDEX agent_configs_name_idx ON agent_configs (customer_id, name)
    WHERE deleted_at IS NULL;
CREATE INDEX agent_configs_customer_idx ON agent_configs (customer_id, created_at DESC);

-- +goose Down
DROP TABLE agent_configs;
