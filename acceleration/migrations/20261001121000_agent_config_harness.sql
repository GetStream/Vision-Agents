-- +goose Up

-- harness is which harness a session of this agent runs: what hands work to the subagent,
-- loads skills, compacts the conversation and starts the sandbox. "default" is the one
-- there is. It is named on the config, never on a session, so a caller reaching an agent
-- cannot run it under a harness its owner did not choose.
ALTER TABLE agent_configs ADD COLUMN harness TEXT NOT NULL DEFAULT 'default';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN harness;
