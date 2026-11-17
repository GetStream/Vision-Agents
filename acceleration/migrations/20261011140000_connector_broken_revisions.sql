-- +goose Up

-- connector_broken_revisions marks a built-in connector's revisions that do not work (AI-816),
-- as a later revision's manifest declares them (core.Manifest.BrokenRevisions): Slack's
-- revisions 1 to 4 read the consent's user at $.authed_user.id, where the live token response
-- has none. The resolver refuses a connection pinned to one, so its person is asked to log in
-- again, and that consent pins the latest revision.
--
--   - A table of its own, not a column on connector_definitions: builds from before this file
--     read that table through bun, and a mark is about a revision, not part of it.
--   - Built-ins only: the seeder is the one writer (store.seedBuiltin), and a custom definition
--     is made from a request that has no broken_revisions (api.CustomConnectorRequest). A
--     built-in id never starts with custom_ (connector_definitions_custom_prefix), so the id
--     alone names a built-in.
--   - Marks only accumulate: a later revision that no longer lists one leaves it, and a mark
--     already stored keeps its first reason. Reverting a broken manifest is a new revision.
--   - No foreign key to connector_definitions: a database seeded first by the marking build
--     never stored the revisions it marks, and no connection can be pinned to those.
--   - Expand only: a new table, nothing existing is read or rewritten.

CREATE TABLE connector_broken_revisions (
    connector_id TEXT NOT NULL CHECK (NOT starts_with(connector_id, 'custom_')),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    reason TEXT NOT NULL CHECK (reason <> ''),
    -- The revision whose manifest declared the mark, always a later one.
    marked_by INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (connector_id, revision),
    CHECK (marked_by > revision)
);

-- +goose Down
DROP TABLE connector_broken_revisions;
