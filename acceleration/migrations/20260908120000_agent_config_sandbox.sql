-- +goose Up

-- sandbox is where the subagent may run code it writes, "daytona" being the one provider
-- there is. Deciding it on the config rather than per session is the point: which sandbox
-- an agent is allowed is a property of the agent, not of the call that reaches it.
--
-- Empty means the subagent runs no code and works everything out in its head, which is
-- what every config that already exists was doing.
ALTER TABLE agent_configs ADD COLUMN sandbox TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN sandbox;
