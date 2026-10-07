-- +goose Up

-- connector_invocations is one row for each connector tool call a session's dispatcher ran
-- (T29, AI-856): which binding, connection and tool, how long it took, and how it failed
-- (architecture doc on connectors/planning, «Add» item 8: «Invocation log with latency_ms and
-- error_type»). The design names no other field, so the call's arguments and result are never
-- stored, for any session.
--
--   - session_id is empty for an incognito session: the row says a credential was used, and
--     nothing ties it to the conversation (session.invocationRecorder).
--   - error_type is empty for a call that answered. The five values are the subtask's
--     (subtasks.md T29), the first four from the architecture doc's item 8.
--   - The rows go with their connection when it is hard deleted (DELETE
--     /v1/agents/users/{user_id}/connections), so an offboarded user leaves no history of use.
--
-- A data move does not carry these rows (store.dataTables): they are this deployment's
-- record of what it did with credentials sealed under its own key.
CREATE TABLE connector_invocations (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connection_id TEXT NOT NULL REFERENCES connector_connections (id) ON DELETE CASCADE,
    connector_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    binding TEXT NOT NULL,
    tool TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ NOT NULL,
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    error_type TEXT NOT NULL DEFAULT '' CHECK (error_type IN
        ('', 'customer_auth', 'external_server', 'client_timeout', 'outcome_unknown', 'denied'))
);

-- One connection's calls, newest first: the list endpoint's order and cursor
-- (store.ConnectorInvocations).
CREATE INDEX connector_invocations_connection_idx
    ON connector_invocations (customer_id, connection_id, started_at DESC, id DESC);

-- connector_audit is one row for each grant a connection got, renewed or lost (T47, AI-876),
-- with the ids that tie it to what caused it, as the Observability tab of Vercel Connect
-- shows each token request. The proxy call (T44) and the token export (T45) write here too,
-- which is why action already allows them and status_code, latency_ms and target exist:
-- neither needs a migration of its own.
--
--   - There is no foreign key to connector_connections: a row outlives its connection, so a
--     deletion is on record after it. It holds no owner id and no account id, which a user
--     delete removes (architecture doc, «Add» item 9).
--   - revision is the connection's credential revision once the change committed, 0 when it
--     names none (a delete).
--   - request_id, session_id and attempt_id are the correlation ids: the API request, the
--     session whose call caused it (empty for an incognito one), the consent's attempt.
--
-- A data move does not carry these rows, for the reason connector_invocations gives.
CREATE TABLE connector_audit (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connection_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    owner_type TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN
        ('grant_created', 'grant_refreshed', 'grant_revoked', 'proxy_call', 'token_export')),
    reason TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL DEFAULT 0,
    request_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    attempt_id TEXT NOT NULL DEFAULT '',
    -- The provider's status, the time it took and the host it reached, for a proxy call.
    status_code INTEGER,
    latency_ms INTEGER,
    target TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL
);

-- The customer's audit, newest first, whole or for one connection: the list endpoint's two
-- orders and cursors (store.ConnectorAuditEvents).
CREATE INDEX connector_audit_customer_idx
    ON connector_audit (customer_id, created_at DESC, id DESC);
CREATE INDEX connector_audit_connection_idx
    ON connector_audit (customer_id, connection_id, created_at DESC, id DESC);

-- +goose Down
DROP TABLE connector_audit;
DROP TABLE connector_invocations;
