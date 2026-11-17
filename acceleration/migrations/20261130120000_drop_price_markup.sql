-- +goose Up

-- markup_bps on price_changes and price_entries has been read by nothing and left NULL on
-- every row since 20261118120000_price_entry_prices stored each entry's price. Only a router
-- built before that migration reads it, and such a router can no longer read the price list
-- once this has run. Dropping a column marks it gone without rewriting the table,
-- and fires no row trigger, so price_entries_guard has nothing to refuse. Each column's check
-- goes with it.
ALTER TABLE price_changes DROP COLUMN markup_bps;
ALTER TABLE price_entries DROP COLUMN markup_bps;

-- The guard compared every column but the cancel's, markup_bps among them, and a PL/pgSQL
-- body is checked only when it runs: left as it was, it would fail every cancel from now on.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION price_changes_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'price_changes rows are never deleted';
    END IF;
    IF OLD.cancelled_at IS NOT NULL OR OLD.effective_at <= now() OR NEW.cancelled_at IS NULL
        OR (NEW.name, NEW.kind, NEW.reason, NEW.source, NEW.effective_at,
            NEW.created_by, NEW.idempotency_key, NEW.created_at)
           IS DISTINCT FROM
           (OLD.name, OLD.kind, OLD.reason, OLD.source, OLD.effective_at,
            OLD.created_by, OLD.idempotency_key, OLD.created_at) THEN
        RAISE EXCEPTION 'a price change can only be cancelled, once, before it takes effect';
    END IF;
    RETURN NEW;
END $$;
-- +goose StatementEnd

-- +goose Down

-- The columns come back with their checks and NULL on every row: the markups they held are
-- gone, and each entry's stored price is what it is charged.
ALTER TABLE price_changes
    ADD COLUMN markup_bps INTEGER CONSTRAINT price_changes_markup_bps_check CHECK (markup_bps BETWEEN 0 AND 100000);
ALTER TABLE price_entries
    ADD COLUMN markup_bps INTEGER CONSTRAINT price_entries_markup_bps_check CHECK (markup_bps BETWEEN 0 AND 100000);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION price_changes_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'price_changes rows are never deleted';
    END IF;
    IF OLD.cancelled_at IS NOT NULL OR OLD.effective_at <= now() OR NEW.cancelled_at IS NULL
        OR (NEW.name, NEW.kind, NEW.reason, NEW.source, NEW.effective_at, NEW.markup_bps,
            NEW.created_by, NEW.idempotency_key, NEW.created_at)
           IS DISTINCT FROM
           (OLD.name, OLD.kind, OLD.reason, OLD.source, OLD.effective_at, OLD.markup_bps,
            OLD.created_by, OLD.idempotency_key, OLD.created_at) THEN
        RAISE EXCEPTION 'a price change can only be cancelled, once, before it takes effect';
    END IF;
    RETURN NEW;
END $$;
-- +goose StatementEnd
