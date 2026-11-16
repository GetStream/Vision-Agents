-- +goose Up

-- An LLM bills by tokens rather than by audio or characters, and it bills the prompt and
-- the generated text at different rates, so the two directions are counted separately.
-- Cached prompt tokens are a subset of the input, tracked because providers charge less
-- for them. Rows from the other modalities keep the default of zero.
ALTER TABLE requests ADD COLUMN input_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN cached_input_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN output_tokens BIGINT NOT NULL DEFAULT 0;

-- The rollups gain the same three, so token usage aggregates the same way every other
-- billable unit does. Existing buckets read zero until the rollup is re-run over them.
ALTER TABLE stats_hourly ADD COLUMN input_tokens_total BIGINT NOT NULL DEFAULT 0;
ALTER TABLE stats_hourly ADD COLUMN cached_input_tokens_total BIGINT NOT NULL DEFAULT 0;
ALTER TABLE stats_hourly ADD COLUMN output_tokens_total BIGINT NOT NULL DEFAULT 0;

ALTER TABLE stats_daily ADD COLUMN input_tokens_total BIGINT NOT NULL DEFAULT 0;
ALTER TABLE stats_daily ADD COLUMN cached_input_tokens_total BIGINT NOT NULL DEFAULT 0;
ALTER TABLE stats_daily ADD COLUMN output_tokens_total BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE stats_daily DROP COLUMN output_tokens_total;
ALTER TABLE stats_daily DROP COLUMN cached_input_tokens_total;
ALTER TABLE stats_daily DROP COLUMN input_tokens_total;

ALTER TABLE stats_hourly DROP COLUMN output_tokens_total;
ALTER TABLE stats_hourly DROP COLUMN cached_input_tokens_total;
ALTER TABLE stats_hourly DROP COLUMN input_tokens_total;

ALTER TABLE requests DROP COLUMN output_tokens;
ALTER TABLE requests DROP COLUMN cached_input_tokens;
ALTER TABLE requests DROP COLUMN input_tokens;
