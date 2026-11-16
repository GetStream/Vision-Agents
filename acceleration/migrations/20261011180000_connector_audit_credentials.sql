-- +goose Up

-- connector_audit_credentials names the tokens an audit row's grant event left, by
-- fingerprint (core.Fingerprint: the first 4 bytes of the token's SHA-256, in hex), so whether
-- a refresh rotated the refresh token is read here instead of from the sealed credentials
-- (AI-990). No token, and no character of one, is stored.
--
--   - A side table, not columns on connector_audit: the router before this one reads
--     connector_audit through bun.
--   - One row for an audit row whose scheme names its tokens (core.Fingerprinter); none for a
--     proxy call, a token export, a delete, or a scheme that does not.
--   - previous_* are the fingerprints before the event, empty for a first grant. rotated says
--     the refresh token the connection already had was replaced.
--   - The rows go with their audit row.
--
-- A data move does not carry these rows, as it does not carry connector_audit (store.dataTables).
CREATE TABLE connector_audit_credentials (
    audit_id TEXT PRIMARY KEY REFERENCES connector_audit (id) ON DELETE CASCADE,
    access_fingerprint TEXT NOT NULL DEFAULT '',
    previous_access_fingerprint TEXT NOT NULL DEFAULT '',
    refresh_fingerprint TEXT NOT NULL DEFAULT '',
    previous_refresh_fingerprint TEXT NOT NULL DEFAULT '',
    rotated BOOLEAN NOT NULL DEFAULT false,
    access_expires_at TIMESTAMPTZ,
    refresh_expires_at TIMESTAMPTZ
);

-- +goose Down
DROP TABLE connector_audit_credentials;
