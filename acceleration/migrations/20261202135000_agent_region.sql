-- +goose Up

-- The region an agent's work is done in, which its agent.yaml or its config names: every model
-- its sessions run, its knowledge base and its sandbox are there. A sync naming a model with no
-- instance there is refused, so nothing is looked up when its work arrives. Every agent written
-- before is in us, and so is one that names none. The CHECK is the form knowledge_namespaces and
-- provider_keys hold a region in. Its lock waits at most 5 s behind a running query rather than
-- queueing every config read.
SET LOCAL lock_timeout = '5s';
ALTER TABLE agent_configs
    ADD COLUMN region TEXT NOT NULL DEFAULT 'us' CHECK (region ~ '^[a-z][a-z0-9-]{0,31}$');

-- +goose Down
-- An agent kept in another region than us reads as one in us afterwards: export the column first.
SET LOCAL lock_timeout = '5s';
ALTER TABLE agent_configs DROP COLUMN region;
