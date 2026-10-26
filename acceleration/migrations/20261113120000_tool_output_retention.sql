-- +goose NO TRANSACTION
-- +goose Up

-- tool_output_retention is how long a config's sessions keep what their tools were given
-- and gave back: none, a duration such as 30d or 12h, or empty for as long as the session
-- (store.AgentConfig.ToolOutputRetention). Empty for every config there is, so nothing
-- changes until one sets it. A constant default is a catalogue change, not a rewrite.
ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS tool_output_retention TEXT NOT NULL DEFAULT '';

-- redact_at is when a tool item's arguments and output are taken out, stamped when it is
-- written (store.AgentResponseItem.RedactAt). NULL, as every existing row is, is never.
ALTER TABLE agent_response_items ADD COLUMN IF NOT EXISTS redact_at TIMESTAMPTZ;

-- Only the items still waiting to be redacted are in the index, and redacting one takes it
-- out, so the index stays the size of what is pending however large the table grows.
-- Built concurrently, outside a transaction, so writers are not held while it builds.
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_response_items_redact_idx
    ON agent_response_items (redact_at) WHERE redact_at IS NOT NULL;

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS agent_response_items_redact_idx;
ALTER TABLE agent_response_items DROP COLUMN IF EXISTS redact_at;
ALTER TABLE agent_configs DROP COLUMN IF EXISTS tool_output_retention;
