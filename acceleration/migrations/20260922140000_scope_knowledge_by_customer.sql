-- +goose Up

-- Knowledge bases moved from turbopuffer namespaces named only by the caller to ones that
-- carry the customer too, so what was written before is not where anything reads now.
-- These make it get written again rather than listed as if it were still there.

-- A synced directory refills its knowledge base on its next sync instead of skipping it as
-- unchanged.
UPDATE agent_configs SET sync_hash = '' WHERE knowledge_namespace <> '';

-- A page is pending again, with nothing written, until it is read into its new base.
UPDATE knowledge_urls SET state = 'pending', passages = 0, error = '', last_indexed_at = NULL
    WHERE deleted_at IS NULL;

-- A posted document has to be posted again; nothing here can refill it.
DELETE FROM knowledge_documents;

-- +goose Down
SELECT 1;
