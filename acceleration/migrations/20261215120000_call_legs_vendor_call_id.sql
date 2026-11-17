-- +goose NO TRANSACTION
-- +goose Up

-- vendor_call_id is the id the vendor gave an outbound leg when it took the dial, a placed
-- call's or a transfer's, so digits are pressed only on a leg the customer placed. An inbound
-- leg has none. The column has no default, so adding it rewrites no row; its lock waits at
-- most 5 s behind a running query, past which the migration fails and the router's next
-- start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE call_legs ADD COLUMN IF NOT EXISTS vendor_call_id TEXT;
END $$;
-- +goose StatementEnd

-- Built without stopping the writes of legs.
CREATE INDEX CONCURRENTLY IF NOT EXISTS call_legs_vendor_call_idx ON call_legs (vendor_call_id)
    WHERE vendor_call_id IS NOT NULL;

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS call_legs_vendor_call_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE call_legs DROP COLUMN IF EXISTS vendor_call_id;
END $$;
-- +goose StatementEnd
