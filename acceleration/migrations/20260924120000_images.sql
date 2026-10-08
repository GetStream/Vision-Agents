-- +goose Up

-- Image generation bills by the picture, so how many were drawn is counted the way audio,
-- characters and tokens already are. Rows from the other modalities keep the default of
-- zero.
ALTER TABLE requests ADD COLUMN images BIGINT NOT NULL DEFAULT 0;

-- The rollups gain the same count, so pictures aggregate the same way every other billable
-- unit does. Existing buckets read zero until the rollup is re-run over them.
ALTER TABLE stats_hourly ADD COLUMN images_total BIGINT NOT NULL DEFAULT 0;
ALTER TABLE stats_daily ADD COLUMN images_total BIGINT NOT NULL DEFAULT 0;
ALTER TABLE stats_tags_hourly ADD COLUMN images_total BIGINT NOT NULL DEFAULT 0;
ALTER TABLE stats_tags_daily ADD COLUMN images_total BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE stats_tags_daily DROP COLUMN images_total;
ALTER TABLE stats_tags_hourly DROP COLUMN images_total;
ALTER TABLE stats_daily DROP COLUMN images_total;
ALTER TABLE stats_hourly DROP COLUMN images_total;
ALTER TABLE requests DROP COLUMN images;
