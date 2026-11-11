-- +goose Up

-- a verification request is what an organization told us about itself to be verified: one row
-- per organization, replaced whenever it sends again.
CREATE TABLE verification_requests (
    organization_id TEXT PRIMARY KEY,
    legal_name TEXT NOT NULL,
    country TEXT NOT NULL,
    website TEXT NOT NULL,
    tax_id TEXT NOT NULL,
    use_case TEXT NOT NULL,
    calls_per_month TEXT NOT NULL,
    email TEXT NOT NULL,
    submitted_by TEXT NOT NULL DEFAULT '',
    submitted_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX verification_requests_submitted ON verification_requests (submitted_at DESC, organization_id DESC);

-- +goose Down

DROP TABLE verification_requests;
