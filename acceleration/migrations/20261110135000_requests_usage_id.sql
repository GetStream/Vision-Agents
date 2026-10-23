-- +goose NO TRANSACTION
-- +goose Up

-- One ALTER TABLE, so requests is locked once. Its lock waits at most 5 s behind a running
-- query rather than queueing every write behind it: past that the migration fails, the router
-- does not start, and its next start tries again. Outside a transaction each statement of this
-- file may run on another pooled connection, so the timeout is set inside the one block it
-- bounds.
--
-- usage_id is the unit of work a row is for, chosen before the row is first written, so
-- writing it again after a failure is a no-op rather than a second charge. Rows from before
-- this have none. Its default is for a router from before this, which writes no usage id: its
-- rows get a random one rather than being refused. Setting a default rewrites no row. NOT
-- VALID checks every new row without scanning the old; it applies to an UPDATE too, so an
-- update of a row from before this must set a usage id or skip rows without one.
--
-- The rest: who is billed (organization_id, resolved when the row is written), who asked
-- (tier, key, user and categories; the writer defaults an empty key and tier), the session the
-- work happened in, empty outside one, and whether the row still has to reach the usage
-- service (to_ship, false from a router with none, so its rows never wait for one), when it
-- did, and why the service refused it (a refused row is not sent again until somebody reships
-- it). Every added default is a constant, so no column rewrites the table.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE requests
        ADD COLUMN IF NOT EXISTS usage_id UUID,
        ALTER COLUMN usage_id SET DEFAULT gen_random_uuid(),
        ADD COLUMN IF NOT EXISTS organization_id TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS organization_tier TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS api_key TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS usage_categories TEXT[] NOT NULL DEFAULT '{}',
        ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS to_ship BOOLEAN NOT NULL DEFAULT false,
        ADD COLUMN IF NOT EXISTS shipped_at TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS ship_error TEXT NOT NULL DEFAULT '';
    -- In the same transaction, so under the lock already held. A failed index build below
    -- leaves this in place, and running the migration again must not trip over it.
    IF NOT EXISTS (SELECT FROM pg_constraint WHERE conrelid = 'requests'::regclass AND conname = 'requests_usage_id_present') THEN
        ALTER TABLE requests ADD CONSTRAINT requests_usage_id_present CHECK (usage_id IS NOT NULL) NOT VALID;
    END IF;
END $$;
-- +goose StatementEnd

-- Built without stopping writes. Store.Migrate builds either again when a build failed and
-- left it invalid, or somebody dropped it: writing usage needs requests_usage_id_idx.
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS requests_usage_id_idx ON requests (usage_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_to_ship_idx ON requests (started_at) WHERE to_ship;

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS requests_to_ship_idx;
DROP INDEX CONCURRENTLY IF EXISTS requests_usage_id_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE requests
        DROP CONSTRAINT IF EXISTS requests_usage_id_present,
        DROP COLUMN IF EXISTS ship_error,
        DROP COLUMN IF EXISTS shipped_at,
        DROP COLUMN IF EXISTS to_ship,
        DROP COLUMN IF EXISTS session_id,
        DROP COLUMN IF EXISTS usage_categories,
        DROP COLUMN IF EXISTS user_id,
        DROP COLUMN IF EXISTS api_key,
        DROP COLUMN IF EXISTS organization_tier,
        DROP COLUMN IF EXISTS organization_id,
        DROP COLUMN IF EXISTS usage_id;
END $$;
-- +goose StatementEnd
