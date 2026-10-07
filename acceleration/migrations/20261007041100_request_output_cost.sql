-- +goose Up

-- The part of cost_micros the generated tokens were priced at, so a request's cost splits
-- into what it read and what it wrote. Rows from before, and models that write no tokens,
-- read zero.
ALTER TABLE requests ADD COLUMN output_cost_micros BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE requests DROP COLUMN output_cost_micros;
