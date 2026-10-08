-- +goose Up

-- Passages repeat their section's heading, so a document cannot be put back together from
-- them. Keeping what was posted is what lets it be read back and edited. Documents written
-- before this have none.
ALTER TABLE knowledge_documents ADD COLUMN text TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE knowledge_documents DROP COLUMN text;
