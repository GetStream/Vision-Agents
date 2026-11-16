-- +goose Up

-- A document the business posted, or an agent directory synced, into a knowledge base.
-- The passages live in turbopuffer keyed by the source and a position; this is what lets
-- somebody list what a base was filled with and remove one, since knowing how many
-- passages a source was cut into is knowing exactly which ids to delete.
CREATE TABLE knowledge_documents (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    source TEXT NOT NULL,
    passages INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Posting the same source again replaces it, since it writes the same passage ids.
CREATE UNIQUE INDEX knowledge_documents_source_idx
    ON knowledge_documents (customer_id, namespace, source);

-- +goose Down
DROP TABLE knowledge_documents;
