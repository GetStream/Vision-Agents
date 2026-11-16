-- +goose Up

-- What an organization or an app has decided about spend, data handling and prompt
-- injection. One document per scope rather than a column per setting, for the same reason
-- apps.settings is a document: the list will grow, and an absent key has to mean "no
-- opinion" so an organization's setting shows through an app that never wrote one.
CREATE TABLE policies (
    scope TEXT NOT NULL CHECK (scope IN ('organization', 'app')),
    scope_id TEXT NOT NULL,
    document JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scope, scope_id)
);

-- Which organization an app was last seen under. Behind a proxy the organization arrives
-- as a header on each request and there is no apps row to read it from, so an
-- organization's budget can only be summed over apps the router has seen it name.
CREATE TABLE app_organizations (
    app_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    seen_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX app_organizations_organization_idx ON app_organizations (organization_id);

-- A budget sums one customer's spend since the start of an interval on every check that
-- misses the cache. The existing index leads with modality, which a sum across all of
-- them cannot use.
CREATE INDEX requests_customer_spend_idx ON requests (customer_id, started_at DESC) INCLUDE (cost_micros);

-- +goose Down

DROP INDEX requests_customer_spend_idx;
DROP TABLE app_organizations;
DROP TABLE policies;
