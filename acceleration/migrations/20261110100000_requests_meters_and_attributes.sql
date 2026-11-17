-- +goose Up

-- What priced each row beyond the typed columns: quantities with no column of their own
-- (cache writes, speech to speech by modality, requests, pixels), and the attributes the
-- price was chosen by (a service tier, a country) or that explain it. billable is false for
-- free and local work, which counts toward no spend. Each default is a constant, so adding
-- the column does not rewrite the table, and rows from before read as billable: a free or
-- local row among them cost nothing.
ALTER TABLE requests
    ADD COLUMN meters JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN billable BOOLEAN NOT NULL DEFAULT true;

-- +goose Down
ALTER TABLE requests DROP COLUMN billable, DROP COLUMN attributes, DROP COLUMN meters;
