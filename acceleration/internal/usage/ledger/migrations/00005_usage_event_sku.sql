-- +goose Up
-- The SKU of the item an event's work was priced as, the same across its price changes. A
-- producer built before it names none, and its events read ''. The default is a constant, so
-- adding the column rewrites no row.
ALTER TABLE usage_events ADD COLUMN sku TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE usage_events DROP COLUMN sku;
