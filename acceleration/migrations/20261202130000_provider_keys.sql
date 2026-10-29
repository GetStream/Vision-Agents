-- +goose Up

-- The keys customers keep with the router for their own accounts at a vendor, one per vendor and
-- region: work routed on one is billed to the customer by the vendor, and by the router as BYOK.
-- The key is sealed under this deployment's key, bound to the customer, the vendor and the
-- region, so a key copied to another row does not open. key_hint is its last four characters,
-- empty for a key too short to show any of, and is all of it that is ever read back.
CREATE TABLE provider_keys (
    customer_id TEXT NOT NULL,
    provider TEXT NOT NULL CHECK (provider ~ '^[a-z0-9][a-z0-9._-]{0,63}$'),
    region TEXT NOT NULL CHECK (region ~ '^[a-z][a-z0-9-]{0,31}$'),
    api_key_sealed BYTEA NOT NULL,
    kek_version INTEGER NOT NULL,
    key_hint TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, provider, region)
);

-- The account each knowledge base is kept in, beside its region: stream's, or the customer's own
-- at the knowledge vendor, on the key they kept for its region when the base was first written.
-- A base stays in that account. Every base written before is in stream's.
ALTER TABLE knowledge_namespaces
    ADD COLUMN account TEXT NOT NULL DEFAULT 'stream' CHECK (account IN ('stream', 'customer'));

-- A provider key put or deleted is audited as provider_key. NOT VALID, as 20261112120000 did:
-- that narrows nothing, and the table is not read again to prove what the old CHECK already did.
-- Its lock waits at most 5 s behind a running query rather than queueing every audit write.
SET LOCAL lock_timeout = '5s';
ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_resource_type_check,
    ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
        ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy',
         'price_list', 'end_user', 'limit', 'provider_key')) NOT VALID;

-- +goose Down
-- The provider_key audit rows stay, NOT VALID, rather than delete the record of who changed what.
-- A knowledge base in a customer's own account is no longer told apart: export the column first.
SET LOCAL lock_timeout = '5s';
ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_resource_type_check,
    ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
        ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy',
         'price_list', 'end_user', 'limit')) NOT VALID;
ALTER TABLE knowledge_namespaces DROP COLUMN account;
DROP TABLE provider_keys;
