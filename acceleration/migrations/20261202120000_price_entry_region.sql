-- +goose Up

-- Every entry of the price list names the region its work runs in, and whether it prices work
-- done with the customer's own key (byok), which the vendor bills the customer for. Both are
-- matched exactly: an item is priced only by an entry of its own region and byok state. Every
-- entry written so far is a US entry and not byok.
--
-- The table is renamed so that a router built before this one cannot read a byok entry, which
-- it would take for its item's newest price. Under the new name that router's reads and writes
-- of the price list fail: it keeps pricing with the list it holds in memory, and reads again
-- on its interval, failing each time, until it is replaced.
ALTER TABLE price_entries RENAME TO price_list_entries;

-- The defaults are constants, so adding the columns rewrites no table. region's default is
-- dropped once every row has it, so every later insert names its region.
ALTER TABLE price_list_entries
    ADD COLUMN region TEXT NOT NULL DEFAULT 'us',
    ADD COLUMN byok BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE price_list_entries
    ALTER COLUMN region DROP DEFAULT,
    ADD CONSTRAINT price_list_entries_region CHECK (region <> '');

-- Every item's SKU ends in its region, as pricing.BaseSKU makes it: llm-openai-a becomes
-- llm-openai-a-us, and llm-openai-a-2 becomes llm-openai-a-2-us. requests.sku and the usage
-- ledger are not rewritten: a row keeps the SKU it was billed under, so a row priced before
-- this migration names its item's SKU without the region. The guard refuses every update, so
-- it is off for the backfill alone, inside this migration's transaction.
ALTER TABLE price_list_entries DISABLE TRIGGER price_entries_guard;
UPDATE price_list_entries SET sku = sku || '-us';
ALTER TABLE price_list_entries ENABLE TRIGGER price_entries_guard;

-- The list in force is read by item, and an item is its region and byok state too.
DROP INDEX price_entries_item_idx;
CREATE INDEX price_list_entries_item_idx ON price_list_entries (modality, provider, model, region, byok, effective_at DESC);

-- +goose Down

-- A router built before this one reads every entry of an item as its price, the newest
-- winning, so it would price US work at a byok fee or at another region's price. Down refuses
-- a list holding either, rather than hand those entries to it.
-- +goose StatementBegin
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM price_list_entries WHERE byok OR region <> 'us') THEN
        RAISE EXCEPTION 'the price list holds byok or non-US entries, which a router built before regions would read as US prices';
    END IF;
END $$;
-- +goose StatementEnd

DROP INDEX price_list_entries_item_idx;
CREATE INDEX price_entries_item_idx ON price_list_entries (modality, provider, model, effective_at DESC);

ALTER TABLE price_list_entries DISABLE TRIGGER price_entries_guard;
UPDATE price_list_entries SET sku = left(sku, -length('-us')) WHERE sku LIKE '%-us';
ALTER TABLE price_list_entries ENABLE TRIGGER price_entries_guard;

ALTER TABLE price_list_entries
    DROP CONSTRAINT price_list_entries_region,
    DROP COLUMN byok,
    DROP COLUMN region;
ALTER TABLE price_list_entries RENAME TO price_entries;
