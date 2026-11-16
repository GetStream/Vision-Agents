-- +goose Up

-- connector_connections is one account at one connector, owned by the app or by one of its
-- users (architecture doc, one-way doors 1 and 8, on connectors/planning). It starts from the
-- prototype's table (20260929170000_connectors.sql on codex/connector-support at cf62af0d)
-- with the design's changes:
--   - auth_scheme and tls_scheme are names in the scheme registry, checked by the store when
--     a row is written (store.CreateConnectorConnection), not by a CHECK: a new scheme is a
--     new package, and a CHECK would make it a migration as well. The prototype's auth_type
--     CHECK and auth_header column are gone; a header is the scheme's business.
--   - definition_revision pins the connector_definitions revision the connection was made
--     from. The store checks that revision exists; there is no foreign key, because a
--     built-in's row is under customer '' and a custom one under this row's customer, and
--     because a deployment a customer moves to holds only the built-in revisions its own
--     files name (store.seedBuiltin), so a key would refuse a moved connection.
--   - inputs (what the connection was created with: a shop, a region) and metadata (public
--     values captured at consent: an instance URL, a realm id) are jsonb outside the sealed
--     blob, so they can be read without the key (one-way door 2). They replace the
--     prototype's endpoint and instance: an endpoint is the manifest resolved for these
--     inputs and metadata (core.Manifest.Resolve), so storing it would be a second copy.
--   - material_sealed is core.Material sealed as one blob, and material_kek_version the key
--     version it was sealed under. They replace credential_sealed and credential_kek_version.
--     The AAD binds customer_id, id and revision, which are columns here.

CREATE TABLE connector_connections (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    -- connector_definitions.revision has the same floor.
    definition_revision INTEGER NOT NULL CHECK (definition_revision >= 1),
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL DEFAULT '',
    auth_scheme TEXT NOT NULL,
    -- NULL when the transport needs no client certificate, which is every scheme today.
    tls_scheme TEXT,
    inputs JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    label TEXT NOT NULL DEFAULT '',
    account_id TEXT NOT NULL DEFAULT '',
    -- pending, connected, needs_reauthorization or disconnected (store.Connection*). No
    -- CHECK, as on the prototype: the values are the resolver's, which T12 owns.
    status TEXT NOT NULL DEFAULT 'pending',
    granted_scopes JSONB NOT NULL DEFAULT '[]',
    -- Advances with every new material, which is sealed against it, so a stale write and a
    -- replayed blob both fail (core.Grant). It starts at 1 as on the prototype.
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
    -- Empty until a grant is saved; emptied again on delete.
    material_sealed BYTEA NOT NULL DEFAULT ''::bytea,
    -- 0 while there is no material: key versions start at 1 (auth.NewSealerWithKeyring).
    material_kek_version INTEGER NOT NULL DEFAULT 0,
    -- When the current access expires, a public fact kept outside the blob.
    expires_at TIMESTAMPTZ,
    cached_tools JSONB NOT NULL DEFAULT '[]',
    tools_digest TEXT NOT NULL DEFAULT '',
    tools_checked_at TIMESTAMPTZ,
    last_error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    -- Two owners and no third (one-way door 8). An app-owned connection names no user; a
    -- user-owned one names exactly one. The prototype had the same rule as two CHECKs.
    CONSTRAINT connector_connections_owner
        CHECK ((owner_type = 'app' AND owner_id = '') OR (owner_type = 'user' AND owner_id <> ''))
);

-- Listing one owner's live connections newest first, the cursor's order
-- (store.ConnectorConnectionsByOwner). Deleted rows are never read, so they are not indexed.
CREATE INDEX connector_connections_owner_idx
    ON connector_connections (customer_id, owner_type, owner_id, created_at DESC, id DESC)
    WHERE deleted_at IS NULL;

-- The bindings that grant an agent config tools of a connector (T20 writes them). NOT NULL
-- with a constant default adds no table rewrite on Postgres 11 and later.
ALTER TABLE agent_configs ADD COLUMN connectors JSONB NOT NULL DEFAULT '[]';

-- A data move leaves material_sealed behind, as it does every other sealed secret
-- (store.secretColumns): the key that opens it is this deployment's. This is
-- 20260926120000_data_changes.sql's function with material_sealed added to what it strips.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION record_data_change() RETURNS TRIGGER AS $$
DECLARE
    subject JSONB;
    owner TEXT;
    identity JSONB := '{}'::JSONB;
    column_name TEXT;
BEGIN
    IF TG_OP = 'DELETE' THEN
        subject := to_jsonb(OLD);
    ELSE
        subject := to_jsonb(NEW);
    END IF;

    -- TG_ARGV[0] is the column holding the customer id, or 'parent' when the row belongs
    -- to a customer through another table: TG_ARGV[1] is then the column referencing it
    -- and TG_ARGV[2] the table it references.
    IF TG_ARGV[0] = 'parent' THEN
        EXECUTE format('SELECT customer_id FROM %I WHERE id = $1', TG_ARGV[2])
            INTO owner USING subject ->> TG_ARGV[1];
    ELSE
        owner := subject ->> TG_ARGV[0];
    END IF;

    IF owner IS NULL OR owner = '' THEN
        RETURN NULL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM data_change_capture
        WHERE customer_id = owner AND expires_at > now()
    ) THEN
        RETURN NULL;
    END IF;

    FOREACH column_name IN ARRAY TG_ARGV[3:] LOOP
        identity := identity || jsonb_build_object(column_name, subject -> column_name);
    END LOOP;

    -- Credentials are left behind. A secret this deployment sealed is worth nothing to
    -- the one reading it, and an access token a customer's user granted to this
    -- deployment is not something to hand out because somebody asked for their data.
    subject := subject - 'secret_sealed' - 'access_token' - 'refresh_token'
        - 'oauth_state' - 'code_verifier' - 'material_sealed';

    INSERT INTO data_changes (customer_id, table_name, op, key, payload)
    VALUES (
        owner,
        TG_TABLE_NAME,
        lower(TG_OP),
        identity,
        CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE subject END
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION record_data_change() RETURNS TRIGGER AS $$
DECLARE
    subject JSONB;
    owner TEXT;
    identity JSONB := '{}'::JSONB;
    column_name TEXT;
BEGIN
    IF TG_OP = 'DELETE' THEN
        subject := to_jsonb(OLD);
    ELSE
        subject := to_jsonb(NEW);
    END IF;

    IF TG_ARGV[0] = 'parent' THEN
        EXECUTE format('SELECT customer_id FROM %I WHERE id = $1', TG_ARGV[2])
            INTO owner USING subject ->> TG_ARGV[1];
    ELSE
        owner := subject ->> TG_ARGV[0];
    END IF;

    IF owner IS NULL OR owner = '' THEN
        RETURN NULL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM data_change_capture
        WHERE customer_id = owner AND expires_at > now()
    ) THEN
        RETURN NULL;
    END IF;

    FOREACH column_name IN ARRAY TG_ARGV[3:] LOOP
        identity := identity || jsonb_build_object(column_name, subject -> column_name);
    END LOOP;

    subject := subject - 'secret_sealed' - 'access_token' - 'refresh_token'
        - 'oauth_state' - 'code_verifier';

    INSERT INTO data_changes (customer_id, table_name, op, key, payload)
    VALUES (
        owner,
        TG_TABLE_NAME,
        lower(TG_OP),
        identity,
        CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE subject END
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd
ALTER TABLE agent_configs DROP COLUMN connectors;
DROP TABLE connector_connections;
