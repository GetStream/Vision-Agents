-- +goose Up

-- The voice a call asked for, next to the models it ran on. Empty is the provider's
-- default. A session moved onto other models mid-call rewrites both.
ALTER TABLE calls ADD COLUMN voice TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE calls DROP COLUMN voice;
