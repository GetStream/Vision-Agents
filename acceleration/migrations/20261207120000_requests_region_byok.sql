-- +goose Up

-- region is where the work ran, as the item it was priced as names it, such as us or eu, and
-- byok whether it ran on the customer's own key. Kept beside the SKU so spend splits by them
-- and unpriced work, which has no SKU, still says where it ran. Rows from before them read ''
-- and false. A constant default rewrites no rows.
ALTER TABLE requests ADD COLUMN IF NOT EXISTS region TEXT NOT NULL DEFAULT '';
ALTER TABLE requests ADD COLUMN IF NOT EXISTS byok BOOLEAN NOT NULL DEFAULT false;

-- +goose Down

ALTER TABLE requests DROP COLUMN IF EXISTS byok;
ALTER TABLE requests DROP COLUMN IF EXISTS region;
