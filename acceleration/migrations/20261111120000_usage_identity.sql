-- +goose Up

-- Every usage row names who asked for the work beside the app: the key it was started with,
-- the end user it was for and the customer's usage categories. Work a key scheduled to run
-- later is owned by whoever scheduled it, so a campaign and a scheduled task keep them.
--
-- Constant defaults, so no table is rewritten. A campaign or task from before names no key, so
-- the column is added with its origin as the default, which every existing row reads back, and
-- the default is then set to empty for the rows written from now on, which name their key.
ALTER TABLE campaigns
    ADD COLUMN api_key TEXT NOT NULL DEFAULT 'job:campaign',
    ADD COLUMN user_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN usage_categories TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE campaigns ALTER COLUMN api_key SET DEFAULT '';

-- A scheduled task already keeps its user.
ALTER TABLE scheduled_tasks
    ADD COLUMN api_key TEXT NOT NULL DEFAULT 'job:scheduled_task',
    ADD COLUMN usage_categories TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE scheduled_tasks ALTER COLUMN api_key SET DEFAULT '';

-- A session keeps the key and categories it was opened with for its whole life, also when it
-- is carried on in another process from its row. A session from before names none, and takes
-- the key of whoever carries it on.
ALTER TABLE agent_sessions
    ADD COLUMN api_key TEXT NOT NULL DEFAULT '',
    ADD COLUMN usage_categories TEXT[] NOT NULL DEFAULT '{}';

-- The tier Stream last sent for each organization, an opaque string, for work no request
-- carried: a webhook, a scheduled call, a session's later turns on another router.
CREATE TABLE organization_tiers (
    organization_id TEXT PRIMARY KEY,
    tier TEXT NOT NULL CHECK (tier <> ''),
    seen_at TIMESTAMPTZ NOT NULL
);

-- An end user forgotten in the usage records is audited under the id that replaced theirs.
-- The CHECK is dropped and added again with the new value beside the ones it had, NOT VALID,
-- as 20261110120000 did: that narrows nothing, and the table is not read again to prove what
-- the old CHECK already did. A router built before this one writes none of the new value.
ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_resource_type_check,
    ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
        ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy',
         'price_list', 'end_user')) NOT VALID;

-- +goose Down
-- NOT VALID: the old CHECK holds for every row written from now on and leaves the end_user rows
-- already stored in place, rather than delete the record of who was forgotten.
ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_resource_type_check,
    ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
        ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy',
         'price_list')) NOT VALID;
DROP TABLE organization_tiers;
ALTER TABLE agent_sessions DROP COLUMN usage_categories, DROP COLUMN api_key;
ALTER TABLE scheduled_tasks DROP COLUMN usage_categories, DROP COLUMN api_key;
ALTER TABLE campaigns DROP COLUMN usage_categories, DROP COLUMN user_id, DROP COLUMN api_key;
