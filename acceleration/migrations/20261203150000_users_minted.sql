-- +goose Up
-- minted says the router itself handed this guest out (POST /v1/agents/guests). A guest row
-- the users recorder writes from a verified guest token is not minted: an app signs those
-- itself, so they can name any id. Rows already here stay false; their guests mint anew.
ALTER TABLE users ADD COLUMN minted BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
ALTER TABLE users DROP COLUMN minted;
