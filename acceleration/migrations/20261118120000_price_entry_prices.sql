-- +goose Up

-- Every entry of the price list stores what its item is charged and the SKU that names the
-- item across its price changes. rates and tiers stay the vendor's cost, for reference; a quote
-- charges prices and price_tiers only, so no markup is added when a row is priced. markup_bps
-- on both tables is left in place, read by nothing and NULL on every row written from now on,
-- for a router built before this one: a later release drops it.
--
-- The defaults are constants, so adding the columns rewrites no table. A router built before
-- this one inserts entries without them, which the checks below refuse: it cannot write a
-- price change once this has run.
ALTER TABLE price_entries
    ADD COLUMN sku TEXT NOT NULL DEFAULT '',
    -- What the work is charged, by the meters of rates, in their shape.
    ADD COLUMN prices JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- The context tiers of prices, at the thresholds of tiers.
    ADD COLUMN price_tiers JSONB NOT NULL DEFAULT '[]'::jsonb;

-- The SKU each row was priced under, beside price_version. '' on every row priced before it,
-- and on work no entry priced.
ALTER TABLE requests ADD COLUMN sku TEXT NOT NULL DEFAULT '';

-- rates with every usd multiplied by (10000 + markup_bps) / 10000 and kept to 12 significant
-- digits, as the router rounds a price it works out from a rate.
-- +goose StatementBegin
CREATE FUNCTION price_entries_marked(rates JSONB, markup_bps BIGINT) RETURNS JSONB
    LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(jsonb_object_agg(meter, CASE
        WHEN usd IS NULL THEN rate
        WHEN usd = 0 THEN rate || jsonb_build_object('usd', 0)
        ELSE rate || jsonb_build_object('usd', trim_scale(round(usd, 11 - floor(log(abs(usd)))::int)))
    END), '{}'::jsonb)
    FROM jsonb_each(CASE WHEN jsonb_typeof(rates) = 'object' THEN rates ELSE '{}'::jsonb END) AS r (meter, rate),
         LATERAL (SELECT (rate->>'usd')::numeric * (10000 + markup_bps) / 10000 AS usd) AS marked
$$;
-- +goose StatementEnd

-- An entry is charged today at its rates plus its own markup, else the list's markup in force
-- when it is (the newest change not cancelled that set one, by the later of the entry's
-- effective time and now), else nothing. Its stored price is exactly that, so no row is
-- charged differently after this runs. A free row has no price. The guard refuses every update,
-- so it is off for the backfill alone, inside this migration's transaction.
ALTER TABLE price_entries DISABLE TRIGGER price_entries_guard;

UPDATE price_entries AS pe
SET prices = price_entries_marked(pe.rates, m.bps),
    price_tiers = coalesce((
        SELECT jsonb_agg(tier || jsonb_build_object('rates', price_entries_marked(tier->'rates', m.bps)) ORDER BY position)
        FROM jsonb_array_elements(CASE WHEN jsonb_typeof(pe.tiers) = 'array' THEN pe.tiers ELSE '[]'::jsonb END)
            WITH ORDINALITY AS t (tier, position)
    ), '[]'::jsonb)
FROM (
    SELECT e.id, coalesce(e.markup_bps, (
        SELECT pc.markup_bps FROM price_changes AS pc
        WHERE pc.cancelled_at IS NULL AND pc.markup_bps IS NOT NULL
          AND pc.effective_at <= GREATEST(e.effective_at, now())
        ORDER BY pc.effective_at DESC, pc.name DESC
        LIMIT 1
    ), 0)::bigint AS bps
    FROM price_entries AS e
) AS m
WHERE m.id = pe.id AND NOT pe.free;

-- One SKU per item, shared by every row of it: its modality, provider, model and each
-- attribute and value in attribute order, lowercased, with every run of characters outside
-- a-z and 0-9 one hyphen, as pricing.BaseSKU makes it. An item whose SKU another took first,
-- in the order the items were first written, is numbered -2, -3 and on.
-- +goose StatementBegin
DO $$
DECLARE
    item RECORD;
    base TEXT;
    candidate TEXT;
    n INT;
BEGIN
    CREATE TEMPORARY TABLE price_skus_taken (sku TEXT PRIMARY KEY) ON COMMIT DROP;
    FOR item IN
        SELECT e.modality, e.provider, e.model, e.attributes,
               e.modality || '-' || e.provider || '-' || e.model || coalesce((
                   SELECT string_agg('-' || a.key || '-' || a.value, '' ORDER BY a.key COLLATE "C")
                   FROM jsonb_each_text(e.attributes) AS a (key, value)
               ), '') AS name
        FROM price_entries AS e
        GROUP BY e.modality, e.provider, e.model, e.attributes
        ORDER BY min(e.id)
    LOOP
        base := trim(BOTH '-' FROM regexp_replace(lower(item.name), '[^a-z0-9]+', '-', 'g'));
        IF base = '' THEN
            base := 'sku';
        END IF;
        candidate := base;
        n := 2;
        WHILE EXISTS (SELECT 1 FROM price_skus_taken WHERE sku = candidate) LOOP
            candidate := base || '-' || n;
            n := n + 1;
        END LOOP;
        INSERT INTO price_skus_taken (sku) VALUES (candidate);
        UPDATE price_entries SET sku = candidate
        WHERE modality = item.modality AND provider = item.provider AND model = item.model
          AND attributes = item.attributes;
    END LOOP;
END $$;
-- +goose StatementEnd

ALTER TABLE price_entries ENABLE TRIGGER price_entries_guard;
DROP FUNCTION price_entries_marked(JSONB, BIGINT);

-- Every row names its item's SKU, and a priced row that is not free has a price. The guard,
-- on again, refuses any update or delete of a row, these columns among the rest.
ALTER TABLE price_entries
    ADD CONSTRAINT price_entries_sku CHECK (sku <> ''),
    ADD CONSTRAINT price_entries_priced CHECK (free OR withdrawn OR rates = '{}'::jsonb OR prices <> '{}'::jsonb);
CREATE INDEX price_entries_sku_idx ON price_entries (sku);

-- +goose Down
DROP INDEX price_entries_sku_idx;
ALTER TABLE price_entries
    DROP CONSTRAINT price_entries_priced,
    DROP CONSTRAINT price_entries_sku,
    DROP COLUMN price_tiers,
    DROP COLUMN prices,
    DROP COLUMN sku;
ALTER TABLE requests DROP COLUMN sku;
