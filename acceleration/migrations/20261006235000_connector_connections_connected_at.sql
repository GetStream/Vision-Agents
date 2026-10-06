-- +goose Up

-- connected_at is when a consent last stored the connection's credentials: when the grant
-- they belong to began (core.CredentialState.ConnectedAt). A refresh keeps it. The events
-- endpoint needs it to tell a provider's signal about an old grant from one about the grant
-- the connection holds now: Slack retries an event it got no 2xx for after 1 and 5 minutes
-- (https://docs.slack.dev/apis/events-api/, «Retries»), so a tokens_revoked can arrive after
-- the account reconnected, and it must not end the new grant (resolver.Resolver.Revoke).
-- NULL for a connection no consent connected since this column was added, which a signal
-- revokes as before.
ALTER TABLE connector_connections ADD COLUMN connected_at TIMESTAMPTZ;

-- +goose Down
ALTER TABLE connector_connections DROP COLUMN connected_at;
