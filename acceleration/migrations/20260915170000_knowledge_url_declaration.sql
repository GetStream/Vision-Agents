-- +goose Up
-- What a page was subscribed as, which is separate from what it calls itself: title is
-- overwritten by the crawler on every read, so a declared one would not survive.
ALTER TABLE knowledge_urls
    ADD COLUMN declared_title TEXT NOT NULL DEFAULT '',
    ADD COLUMN description TEXT NOT NULL DEFAULT '';
-- +goose Down
ALTER TABLE knowledge_urls DROP COLUMN declared_title, DROP COLUMN description;
