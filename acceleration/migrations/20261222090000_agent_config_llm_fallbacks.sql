-- +goose Up

-- llm_fallbacks is an agent config's priority list after llm: the targets a turn
-- falls back to, in order, when llm fails it. NULL, which every config there is
-- starts as, falls back only among llm's own candidates.
ALTER TABLE agent_configs ADD COLUMN llm_fallbacks JSONB;

-- +goose Down

ALTER TABLE agent_configs DROP COLUMN llm_fallbacks;
