-- +goose Up

-- The price list, kept in the shared database. A router's first start loads it from its price
-- files; after that a release only adds the items the database has never priced, and every
-- other change is a new change written here. Rows are only ever added, so the price that
-- billed any request can always be read back.
--
-- The migration runs on every shard too, as every migration does; only the shared database's
-- tables hold prices. A router built before this one reads and writes none of them, and
-- writes requests rows without the two new columns, which their constant defaults fill.

-- One write to the price list: the first load, the items a release added, or a change made
-- later. Never deleted. A change not yet in force can be cancelled, which is the only update
-- allowed.
CREATE TABLE price_changes (
    name TEXT PRIMARY KEY,                -- <UTC yyyymmddhhmmss>_<snake_case>
    kind TEXT NOT NULL CHECK (kind IN ('seed', 'release', 'api')),
    reason TEXT NOT NULL,                 -- what changed, from what to what
    source TEXT NOT NULL,                 -- the vendor's page and the date read, the operator's decision, or the files and release
    effective_at TIMESTAMPTZ NOT NULL,
    -- The list's markup from effective_at on, in basis points. NULL leaves it as it was.
    markup_bps INTEGER CHECK (markup_bps BETWEEN 0 AND 100000),
    created_by TEXT NOT NULL,             -- who made it: an operator, or router <release>
    idempotency_key TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    cancelled_at TIMESTAMPTZ,
    cancelled_by TEXT,
    cancel_reason TEXT
);

-- Every price the router has billed with, or will. Rows are only ever added, each under the
-- change that wrote it. A price changes when a newer row for the same item takes effect, and
-- a model stops being priced by a row that says so.
CREATE TABLE price_entries (
    id BIGSERIAL PRIMARY KEY,
    change TEXT NOT NULL REFERENCES price_changes (name),
    modality TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    -- The item attributes the entry applies to, such as a service tier. Empty is the default.
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    free BOOLEAN NOT NULL DEFAULT false,
    -- The vendor's rates. What a customer pays is these plus the markup.
    rates JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Context tiers: a longer prompt priced at other rates, whole.
    tiers JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- This entry's markup in basis points, replacing the list's. NULL takes the list's.
    markup_bps INTEGER CHECK (markup_bps BETWEEN 0 AND 100000),
    withdrawn BOOLEAN NOT NULL DEFAULT false,
    effective_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX price_entries_item_idx ON price_entries (modality, provider, model, effective_at DESC);
CREATE INDEX price_entries_change_idx ON price_entries (change);

-- An entry takes effect with its change, so nothing can write one that backdates itself, and
-- nothing rewrites a price once written.
-- +goose StatementBegin
CREATE FUNCTION price_entries_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION 'price_entries rows are never changed or deleted: write a newer one';
    END IF;
    NEW.effective_at := (SELECT effective_at FROM price_changes WHERE name = NEW.change);
    RETURN NEW;
END $$;
-- +goose StatementEnd
CREATE TRIGGER price_entries_guard BEFORE INSERT OR UPDATE OR DELETE ON price_entries
    FOR EACH ROW EXECUTE FUNCTION price_entries_guard();

-- A change is cancelled once, before it takes effect; nothing else about it ever changes.
-- +goose StatementBegin
CREATE FUNCTION price_changes_guard() RETURNS trigger LANGUAGE plpgsql AS $$
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
CREATE TRIGGER price_changes_guard BEFORE UPDATE OR DELETE ON price_changes
    FOR EACH ROW EXECUTE FUNCTION price_changes_guard();

-- Which change priced a row, and its cost at the vendor's rates before the markup. Nothing
-- writes them yet; '' and 0 are what every row reads until something does. Each default is a
-- constant, so adding the columns does not rewrite the table.
ALTER TABLE requests
    ADD COLUMN price_version TEXT NOT NULL DEFAULT '',
    ADD COLUMN base_cost_micros BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE requests DROP COLUMN base_cost_micros, DROP COLUMN price_version;
DROP TABLE price_entries;
DROP FUNCTION price_entries_guard();
DROP TABLE price_changes;
DROP FUNCTION price_changes_guard();
