-- +goose NO TRANSACTION
-- +goose Up

-- The grants a row's work was admitted on (limit_grants in the shared database): what a grant
-- has paid for is the billable cost of the rows naming it. Rows from before this name none. A
-- constant default rewrites no row; the lock waits at most 5 s behind a running query, as
-- 20261110135000 does, and outside a transaction it is set inside the one block it bounds.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE requests ADD COLUMN IF NOT EXISTS limit_grants BIGINT[] NOT NULL DEFAULT '{}';
END $$;
-- +goose StatementEnd

-- Built without stopping writes, and only over the rows that name a grant.
CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_limit_grants_idx ON requests
    USING gin (limit_grants) WHERE limit_grants <> '{}';

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS requests_limit_grants_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE requests DROP COLUMN IF EXISTS limit_grants;
END $$;
-- +goose StatementEnd
