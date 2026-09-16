-- +goose Up

-- What an app has turned on or off. The first two entries are which levels of end user it
-- admits — allow_anonymous and allow_guest — and they are read on every authenticated
-- request, in the same query that resolves the API key.
--
-- One JSONB column rather than a boolean column for each, because these two are plainly
-- the start of a list and a column apiece would mean a migration for every toggle anybody
-- thinks of. What the feature actually is is a document keyed by app.
--
-- The default is an empty document rather than one spelling the two defaults out. A key
-- that is absent is a feature nobody has expressed an opinion on, which is what lets a
-- default change without rewriting every row, and it means the existing rows need no
-- backfill: they already mean what they should.
ALTER TABLE apps ADD COLUMN settings JSONB NOT NULL DEFAULT '{}'::jsonb;

-- +goose Down

ALTER TABLE apps DROP COLUMN settings;
