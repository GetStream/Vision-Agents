-- +goose NO TRANSACTION
-- +goose Up

-- The customer's spend limits, at every level, in one table of one shape (T7 A).
--
-- A row limits one subject (whose spend is counted) at one level (whose ids the row names) over
-- one interval. A row at a level above its subject is a default for every subject of that kind
-- there: an app-level api_key row limits each of the app's keys on its own, and a row naming
-- the key wins over it. Every row names its organization; an app seen under none is its own,
-- with its id as the organization's.
--
-- Outside a transaction, so the indexes on requests below build without stopping writes. All
-- else is one DO block, which commits whole or not at all, and can run again after an index
-- build failed: every table and index is IF NOT EXISTS, and the budgets it moved are no longer
-- in the policies.
-- +goose StatementBegin
DO $$
BEGIN
    -- An hourly budget has no interval to move to, so the migration stops rather than drop it.
    IF EXISTS (SELECT 1 FROM policies WHERE document->'budget'->>'interval' = 'hourly') THEN
        RAISE EXCEPTION 'an hourly budget exists, and limits have no hourly interval: change it to daily, or remove it, before migrating';
    END IF;

    CREATE TABLE IF NOT EXISTS limits (
        id BIGSERIAL PRIMARY KEY,
        subject TEXT NOT NULL,
        level TEXT NOT NULL,
        organization_id TEXT NOT NULL CHECK (organization_id <> ''),
        app_id TEXT NOT NULL DEFAULT '',
        api_key TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL DEFAULT '',
        category TEXT NOT NULL DEFAULT '',
        interval TEXT NOT NULL CHECK (interval IN ('daily', 'weekly', 'biweekly', 'monthly')),
        limit_micros BIGINT NOT NULL CHECK (limit_micros >= 0),
        note TEXT NOT NULL DEFAULT '',
        updated_by TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        -- The levels each subject can be limited at.
        CONSTRAINT limits_level_check CHECK ((subject, level) IN (
            ('organization', 'organization'), ('app', 'app'),
            ('api_key', 'organization'), ('api_key', 'app'), ('api_key', 'api_key'),
            ('user', 'organization'), ('user', 'app'), ('user', 'user'),
            ('category', 'organization'), ('app_category', 'app'))),
        -- Each level names exactly its ids.
        CONSTRAINT limits_ids_check CHECK (
            (app_id <> '') = (level IN ('app', 'api_key', 'user'))
            AND (api_key <> '') = (level = 'api_key')
            AND (user_id <> '') = (level = 'user')
            AND (category <> '') = (subject IN ('category', 'app_category')))
    );
    CREATE UNIQUE INDEX IF NOT EXISTS limits_one_per_level ON limits (organization_id, subject, level, interval,
        app_id, api_key, user_id, category);

    -- A reset: from window_start's window on, the subject's limits in that interval count its
    -- spend less baseline_micros, what it had spent when the reset was made. The usage records
    -- are not touched.
    CREATE TABLE IF NOT EXISTS usage_resets (
        id BIGSERIAL PRIMARY KEY,
        organization_id TEXT NOT NULL CHECK (organization_id <> ''),
        subject TEXT NOT NULL,
        app_id TEXT NOT NULL DEFAULT '',
        api_key TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL DEFAULT '',
        category TEXT NOT NULL DEFAULT '',
        interval TEXT NOT NULL,
        window_start TIMESTAMPTZ NOT NULL,
        baseline_micros BIGINT NOT NULL CHECK (baseline_micros >= 0),
        reason TEXT NOT NULL,
        actor TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS usage_resets_organization_idx ON usage_resets (organization_id, window_start);

    -- A one-time amount for one subject, drawn while that subject's own limits are spent, until
    -- it is used up, expires or is revoked.
    CREATE TABLE IF NOT EXISTS limit_grants (
        id BIGSERIAL PRIMARY KEY,
        organization_id TEXT NOT NULL CHECK (organization_id <> ''),
        subject TEXT NOT NULL CHECK (subject IN
            ('organization', 'app', 'api_key', 'user', 'category', 'app_category')),
        app_id TEXT NOT NULL DEFAULT '',
        api_key TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL DEFAULT '',
        category TEXT NOT NULL DEFAULT '',
        amount_micros BIGINT NOT NULL CHECK (amount_micros > 0),
        expires_at TIMESTAMPTZ,
        revoked_at TIMESTAMPTZ,
        revoked_by TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL CHECK (reason <> ''),
        actor TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT limit_grants_expiry_check CHECK (expires_at IS NULL OR expires_at > created_at),
        -- A grant names one subject, never a default: exactly the ids its subject has.
        CONSTRAINT limit_grants_ids_check CHECK (
            (app_id <> '') = (subject IN ('app', 'api_key', 'user', 'app_category'))
            AND (api_key <> '') = (subject = 'api_key')
            AND (user_id <> '') = (subject = 'user')
            AND (category <> '') = (subject IN ('category', 'app_category')))
    );
    CREATE INDEX IF NOT EXISTS limit_grants_live_idx ON limit_grants (organization_id) WHERE revoked_at IS NULL;

    -- A limit, a grant and a reset are audited as limit. NOT VALID, as 20261111120000 did: that
    -- narrows nothing, and the table is not read again to prove what the old CHECK already
    -- did. Its lock waits at most 5 s behind a running query rather than queueing every audit
    -- write behind it.
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE audit_log
        DROP CONSTRAINT audit_log_resource_type_check,
        ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
            ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy',
             'price_list', 'end_user', 'limit')) NOT VALID;
    SET LOCAL lock_timeout = 0;

    -- Budgets become limits: an organization's is the organization's own limit, an app's the
    -- app's, under the organization the app was last seen in, or itself when it was seen in none.
    INSERT INTO limits (subject, level, organization_id, app_id, interval, limit_micros, note, updated_by)
    SELECT p.scope, p.scope,
           CASE WHEN p.scope = 'organization' THEN p.scope_id
                ELSE coalesce(nullif(ao.organization_id, ''), p.scope_id) END,
           CASE WHEN p.scope = 'app' THEN p.scope_id ELSE '' END,
           p.document->'budget'->>'interval',
           (p.document->'budget'->>'limit_micros')::bigint,
           'moved from the policy budget', 'migration'
    FROM policies p
    LEFT JOIN app_organizations ao ON p.scope = 'app' AND ao.app_id = p.scope_id
    WHERE p.document ? 'budget';
    UPDATE policies SET document = document - 'budget' WHERE document ? 'budget';
END $$;
-- +goose StatementEnd

-- What a subject spent, summed from the request rows by a router with no usage service to count
-- it: an app's key, an app's user, and the rows carrying a usage category. Each covers its sum,
-- so it reads the index alone. Built without stopping writes; Store.Migrate builds one again
-- when a build failed and left it invalid.
CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_key_spend_idx ON requests (customer_id, api_key, started_at)
    INCLUDE (cost_micros) WHERE billable;
CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_user_spend_idx ON requests (customer_id, user_id, started_at)
    INCLUDE (cost_micros) WHERE billable AND user_id <> '';
CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_category_spend_idx ON requests (customer_id, started_at)
    INCLUDE (cost_micros, usage_categories) WHERE billable AND usage_categories <> '{}';

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS requests_category_spend_idx;
DROP INDEX CONCURRENTLY IF EXISTS requests_user_spend_idx;
DROP INDEX CONCURRENTLY IF EXISTS requests_key_spend_idx;
-- Each policy gets back one budget from its scope's own limits: the one the migration moved
-- when it is still there, else the oldest. The other limits have no place in a policy and are
-- lost with the table, so export it first. The audit rows of limits stay, NOT VALID, rather
-- than delete the record of who changed what.
-- +goose StatementBegin
DO $$
BEGIN
    INSERT INTO policies (scope, scope_id, document, updated_at)
    SELECT DISTINCT ON (l.subject, scope_id) l.subject,
           CASE WHEN l.subject = 'app' THEN l.app_id ELSE l.organization_id END AS scope_id,
           jsonb_build_object('budget', jsonb_build_object('limit_micros', l.limit_micros, 'interval', l.interval)),
           now()
    FROM limits l
    WHERE l.subject IN ('organization', 'app') AND l.level = l.subject AND l.limit_micros > 0
    ORDER BY l.subject, scope_id, (l.note = 'moved from the policy budget') DESC, l.id
    ON CONFLICT (scope, scope_id) DO UPDATE SET document = policies.document || EXCLUDED.document;
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE audit_log
        DROP CONSTRAINT audit_log_resource_type_check,
        ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
            ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy',
             'price_list', 'end_user')) NOT VALID;
    DROP TABLE limit_grants;
    DROP TABLE usage_resets;
    DROP TABLE limits;
END $$;
-- +goose StatementEnd
