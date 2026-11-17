-- +goose Up

-- connector_definitions is what a connector is, as data: the provider manifest a scheme, a
-- source and the resolver read (internal/connectors/core, manifest.go). A row is one revision
-- and is never updated. A changed manifest is the next revision, so a connection that pinned
-- an earlier one keeps reading what it was created from.
--
-- customer_id is '' for the built-ins seeded from internal/connectors/providers at startup.
-- No caller can be that customer: every auth mode refuses an empty app id
-- (internal/auth/auth.go) and api.CustomerFrom reports none for it. Any other value is the
-- customer whose custom definition the row is.
--
-- Text columns have no length, as in every other table here: Postgres stores TEXT and
-- VARCHAR(n) the same way, and how long a value a customer may send is for the API to refuse.
-- name, category and description repeat the manifest's, so a catalog can list and search
-- definitions without reading every manifest.

CREATE TABLE connector_definitions (
    customer_id TEXT NOT NULL,
    id TEXT NOT NULL,
    -- core.Manifest.Revision, which Manifest.Validate requires to be at least 1.
    revision INTEGER NOT NULL CHECK (revision >= 1),
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    manifest JSONB NOT NULL,
    -- One timestamp, because a revision is written once: the latest revision's created_at is
    -- when the definition last changed.
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, id, revision),
    -- A custom definition's id starts with custom_ and a built-in's never does. So a customer's
    -- definition cannot shadow a built-in, and an id alone says which kind it is.
    CONSTRAINT connector_definitions_custom_prefix
        CHECK ((customer_id = '') <> starts_with(id, 'custom_'))
);

-- +goose Down
DROP TABLE connector_definitions;
