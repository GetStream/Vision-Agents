-- +goose Up

-- What a model answers under belongs to the agent asking, not to the config that decides
-- where the asking goes, so a router config no longer takes a system prompt: the model
-- block cannot be sent one and the conversation block is refused if it names one. The
-- ones already stored would still be written over an agent's, now unseen by anything
-- that can read a config back.
UPDATE router_configs SET llm = llm - 'instructions', sts = sts - 'instructions';

-- +goose Down
SELECT 1;
