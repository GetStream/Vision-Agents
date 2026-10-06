-- +goose Up

-- provider_unit_id is the routing key of a shared webhook: the customer's own unit at the
-- provider (core.MessageRule.ProviderUnitID, core.InboundMessage.ProviderUnitID), such as the
-- WhatsApp phone_number_id a delivery names in value.metadata. When one operator app takes
-- the events of every customer (channel.verifier.secret: operator), the events URL does not
-- name the customer, and this column does: the Router finds the one live connection with the
-- event's connector and unit (store.ConnectorConnectionByProviderUnit).
--
-- It is not account_id. account_id is the grant's identity, and one identity can be live
-- many times: the built-in slack connector's identity is team_id:user_id
-- (internal/connectors/providers/slack.yaml), and the same Slack user may connect through two
-- customers. A unit takes the events of one customer only. One WhatsApp Business account
-- lists several phone numbers (GET /<WABA_ID>/phone_numbers,
-- https://developers.facebook.com/docs/whatsapp/cloud-api/phone-numbers), so the unit is the
-- number and not the account.
--
-- NULL for every connection whose events URL names the customer, so the store writes the
-- column only through store.SetConnectorConnectionProviderUnit, which refuses any other
-- connector. It is not a secret: every delivery carries it. So record_data_change copies it
-- as it is, and an import forces it to NULL instead (store.dataTable.identity), because
-- only a consent proves a unit is the customer's.
ALTER TABLE connector_connections ADD COLUMN provider_unit_id TEXT;

-- One live connection for each unit of a connector, so a delivery is routed to one customer
-- only. Partial, because "this enforces uniqueness among the rows that satisfy the index
-- predicate, without constraining those that do not"
-- (https://www.postgresql.org/docs/current/indexes-partial.html, Example 11.3):
--   - deleted_at IS NULL: a soft-deleted connection keeps its row, and frees its unit for
--     the next connection, as channel_accounts_line_idx does for a number
--     (20261005130000_channel_accounts.sql).
--   - provider_unit_id IS NOT NULL: most connections have no unit, and the index holds
--     only the rows it routes. Unique alone would already let NULLs repeat (NULLs are
--     distinct by default, https://www.postgresql.org/docs/current/indexes-unique.html); the
--     predicate keeps them out of the index as well.
-- connector_id leads: the same unit under two connectors is two routes, since a delivery
-- arrives at one connector's events URL.
CREATE UNIQUE INDEX connector_connections_provider_unit_idx
    ON connector_connections (connector_id, provider_unit_id)
    WHERE deleted_at IS NULL AND provider_unit_id IS NOT NULL;

-- +goose Down
DROP INDEX connector_connections_provider_unit_idx;
ALTER TABLE connector_connections DROP COLUMN provider_unit_id;
