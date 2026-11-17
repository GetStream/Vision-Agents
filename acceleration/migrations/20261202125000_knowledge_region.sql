-- +goose Up

-- The region each knowledge base is kept in: the one its first write through the router was
-- done in. A knowledge base stays there. Its later reads and writes go to that region, and a
-- customer whose policies now keep their work elsewhere is refused rather than having it moved.
CREATE TABLE knowledge_namespaces (
    customer_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    region TEXT NOT NULL CHECK (region ~ '^[a-z][a-z0-9-]{0,31}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, namespace)
);

-- Every knowledge base written before regions was written in us.
INSERT INTO knowledge_namespaces (customer_id, namespace, region)
SELECT customer_id, namespace, 'us' FROM knowledge_documents
UNION
SELECT customer_id, namespace, 'us' FROM knowledge_urls
ON CONFLICT DO NOTHING;

-- +goose Down
DROP TABLE knowledge_namespaces;
