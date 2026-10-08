-- +goose Up

-- audit_log is one row for each change somebody made to the app's configuration: what was
-- changed, who changed it, which client they changed it from, and the before and after of
-- every field that moved.
--
-- Configuration is what goes in here, and only configuration: the agents, their skills, the
-- knowledge they read, the routers, the plugin logins and the policies. What an agent does
-- while it runs -- sessions, simulation runs, calls, invocations -- is not configuration, it
-- is traffic, and traffic is already kept in tables of its own. The line is how often a
-- thing changes: a setting somebody sat down and edited belongs here, a row written by a
-- conversation does not. .factory/features/audit_log.md says it at length.
--
--   - There is no foreign key to anything: a row outlives what it describes, so a deletion
--     is on record after the thing it deleted is gone.
--   - agent_id is the agent the change was to or under, so one agent's history is one index
--     seek. It is empty for a change to something that belongs to no agent, such as a
--     router.
--   - changes is the diff, a list of {field, before, after}. A write that moved nothing
--     writes no row at all, so the list is never empty.
--   - source is which client made the change, from X-Stream-Client. A caller that names
--     none is api: it reached the API directly, which is all that can be said about it.
--
-- A data move does not carry these rows: they are this deployment's record of who changed
-- what in it, not the customer's data.
CREATE TABLE audit_log (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    resource_type TEXT NOT NULL CHECK (resource_type IN
        ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy')),
    resource_id TEXT NOT NULL,
    resource_name TEXT NOT NULL DEFAULT '',
    -- The agent the change was to or under, empty for a resource that belongs to none.
    agent_id TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL CHECK (action IN ('created', 'updated', 'deleted', 'synced')),
    source TEXT NOT NULL CHECK (source IN ('dashboard', 'cli', 'sdk', 'api')),
    -- Who made it, as the client named them. Both are empty for a change nobody signed.
    actor_id TEXT NOT NULL DEFAULT '',
    actor_name TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    changes JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL
);

-- The customer's history, newest first, whole or narrowed: the query endpoint's orders and
-- cursors (store.AuditEntries). The agent index is what an agent's own page reads, and what
-- a sync asks for the changes it would overwrite.
CREATE INDEX audit_log_customer_idx
    ON audit_log (customer_id, created_at DESC, id DESC);
CREATE INDEX audit_log_agent_idx
    ON audit_log (customer_id, agent_id, created_at DESC, id DESC);
CREATE INDEX audit_log_resource_idx
    ON audit_log (customer_id, resource_type, created_at DESC, id DESC);

-- +goose Down
DROP TABLE audit_log;
