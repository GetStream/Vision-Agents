-- +goose Up

-- A page the business wrote down somewhere else. It is a row rather than only passages in
-- the knowledge base because a URL is a subscription: somebody has to be able to list what
-- an agent reads, take one away again, and see when each was last fetched. The passages
-- themselves live in turbopuffer, keyed by the url, and this is what remembers how many of
-- them there are so they can be removed exactly.
CREATE TABLE knowledge_urls (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    -- pending until it has been fetched, then indexed or failed.
    state TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    passages INTEGER NOT NULL DEFAULT 0,
    -- Null until a fetch succeeded, which is what separates a page that was never read
    -- from one that was read and has since broken.
    last_indexed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

-- One row per url per knowledge base: adding the same page twice is a re-index, not a
-- second copy of it, since both would write the same passage ids anyway.
CREATE UNIQUE INDEX knowledge_urls_url_idx ON knowledge_urls (customer_id, namespace, url)
    WHERE deleted_at IS NULL;

CREATE INDEX knowledge_urls_namespace_idx ON knowledge_urls (customer_id, namespace)
    WHERE deleted_at IS NULL;

-- Which search provider an agent finds out about today with, routed the way its models are.
ALTER TABLE agent_configs ADD COLUMN search TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN search;
DROP TABLE knowledge_urls;
