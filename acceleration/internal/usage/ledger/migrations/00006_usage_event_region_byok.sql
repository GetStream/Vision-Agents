-- +goose Up
-- Where an event's work ran, such as us or eu, and whether it ran on the customer's own key,
-- so spend splits by them and unpriced work, which has no SKU, still says where it ran. A
-- producer built before them names neither, and its events read '' and false. The defaults
-- are constants, so adding the columns rewrites no row.
ALTER TABLE usage_events ADD COLUMN region TEXT NOT NULL DEFAULT '';
ALTER TABLE usage_events ADD COLUMN byok BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
ALTER TABLE usage_events DROP COLUMN byok;
ALTER TABLE usage_events DROP COLUMN region;
