-- +goose Up

-- refresh_hours is how often a page is read again on its own. Zero is never: the page is
-- read when it is added and when somebody asks for it to be read again.
ALTER TABLE knowledge_urls ADD COLUMN refresh_hours INTEGER NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE knowledge_urls DROP COLUMN refresh_hours;
