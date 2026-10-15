-- +goose Up

-- Why a connector tool call failed, beside how (error_type), for whoever reads the call log
-- (GET /v1/agents/connections/{id}/invocations, AI-1244):
--
--   - provider_status is the provider's HTTP status for the call's last answered request,
--     NULL when no request was answered: the router refused or held the call, or the provider
--     could not be reached.
--   - error_message is what the call's "connector call failed" log line says of it
--     (session.dispatcher.record): the redacted start of the provider's 4xx or 5xx body, else
--     the redacted error the call returned, at most 1 KiB. '' for a call that answered,
--     and for every row written before this migration.
--
-- Expand only: both are nullable or have a default, so a router built before this one still
-- inserts its column list, and reads its own through bun.
ALTER TABLE connector_invocations
    ADD COLUMN provider_status INTEGER,
    ADD COLUMN error_message TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE connector_invocations
    DROP COLUMN provider_status,
    DROP COLUMN error_message;
