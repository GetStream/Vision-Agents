-- +goose Up

-- An approve of a connection's tools (POST /v1/agents/connections/{id}/tools/approve,
-- AI-1305) is an audit row, action tool_reapproved: it moves a tool's pin
-- (connector_tool_pins) to the schema the provider lists now, which is a trust decision as a
-- grant is.
--
--   - The CHECK on action is dropped and added again with tool_reapproved beside the five it
--     had. This is the one change here that is not expand-only, and it narrows nothing: every
--     row meets the old CHECK, whose values the new one keeps, so every row meets the new one.
--     Both statements run in goose's one transaction, so no moment allows another action.
--     Adding the CHECK reads every row under the ACCESS EXCLUSIVE lock the drop takes; how
--     many rows that is in production is unverified.
--   - tools names the approved tools: each one's name, the schema digest approved and the
--     one it was pinned at before, or none when it was not pinned for the grant. NULL for
--     every other action and for every row written before this migration. Expand-only: a
--     router built before this one inserts its own column list and reads its own through bun.

ALTER TABLE connector_audit DROP CONSTRAINT connector_audit_action_check;
ALTER TABLE connector_audit ADD CONSTRAINT connector_audit_action_check CHECK (action IN
    ('grant_created', 'grant_refreshed', 'grant_revoked', 'proxy_call', 'token_export', 'tool_reapproved'));
ALTER TABLE connector_audit ADD COLUMN tools JSONB;

-- +goose Down
-- NOT VALID: the old CHECK holds for every row written from now on and leaves the
-- tool_reapproved rows already stored in place, rather than delete a customer's audit.
ALTER TABLE connector_audit DROP COLUMN tools;
ALTER TABLE connector_audit DROP CONSTRAINT connector_audit_action_check;
ALTER TABLE connector_audit ADD CONSTRAINT connector_audit_action_check CHECK (action IN
    ('grant_created', 'grant_refreshed', 'grant_revoked', 'proxy_call', 'token_export')) NOT VALID;
