-- +goose NO TRANSACTION
-- +goose Up

-- source_id is the id an agent log copied here by `router replicate` had where it came from.
-- A copied log is given an id of this deployment's, since the source's may be another log's
-- here, and is known by this instead, so a copy run again, or a change replayed, writes it once
-- (store.dataTable.copied). A log written here has none. The column has no default, so adding
-- it rewrites no row; its lock waits at most 5 s behind a running query, past which the
-- migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS source_id BIGINT;
END $$;
-- +goose StatementEnd

-- Built without stopping the writes of logs, and of the requests whose trigger writes one.
-- Store.Migrate builds it again when a build failed and left it invalid: a copy of logs needs it.
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS agent_logs_source_idx ON agent_logs (customer_id, source_id)
    WHERE source_id IS NOT NULL;

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS agent_logs_source_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_logs DROP COLUMN IF EXISTS source_id;
END $$;
-- +goose StatementEnd
