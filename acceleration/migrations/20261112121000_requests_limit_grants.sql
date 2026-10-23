-- +goose NO TRANSACTION
-- +goose Up

-- The grants a row's work was admitted on, and what each grant has paid for.
--
-- requests.limit_grants names the grants a row was drawn from. Rows from before this name none.
-- A constant default rewrites no row; the lock waits at most 5 s behind a running query, as
-- 20261110135000 does, and outside a transaction it is set inside the one block it bounds.
--
-- limit_grants.used_micros is the billable cost of the rows naming the grant, added to by the
-- writer as it writes them. limit_grant_uses (shared database) is each row counted toward each
-- grant, by usage id, so a row written twice, or counted again by a reconcile, counts once.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE requests ADD COLUMN IF NOT EXISTS limit_grants BIGINT[] NOT NULL DEFAULT '{}';
    ALTER TABLE limit_grants ADD COLUMN IF NOT EXISTS used_micros BIGINT NOT NULL DEFAULT 0
        CHECK (used_micros >= 0);
    CREATE TABLE IF NOT EXISTS limit_grant_uses (
        grant_id BIGINT NOT NULL REFERENCES limit_grants (id) ON DELETE CASCADE,
        usage_id UUID NOT NULL,
        micros BIGINT NOT NULL CHECK (micros > 0),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (grant_id, usage_id)
    );
END $$;
-- +goose StatementEnd

-- Built without stopping writes, and only over the rows that name a grant, for the reconcile
-- that counts a row the writer wrote and could not count.
CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_limit_grants_idx ON requests
    USING gin (limit_grants) WHERE limit_grants <> '{}';

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS requests_limit_grants_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    DROP TABLE IF EXISTS limit_grant_uses;
    ALTER TABLE limit_grants DROP COLUMN IF EXISTS used_micros;
    ALTER TABLE requests DROP COLUMN IF EXISTS limit_grants;
END $$;
-- +goose StatementEnd
