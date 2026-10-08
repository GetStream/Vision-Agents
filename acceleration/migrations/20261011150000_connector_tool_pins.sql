-- +goose Up

-- connector_tool_pins is the schema a connection's tool was first offered with, for a tool
-- an agent config grants by name alone on a session binding (store.ToolGrant with no
-- schema_digest). A developer cannot write the digest of such a grant: it comes from each
-- person's own account, and a provider may put the person in the tool's description (Slack's
-- MCP tools name the signed-in user's id, finding F8 of the AI-816 end-to-end run). So the
-- first session that opens the connection takes the digest it finds as the one approved, and
-- later sessions offer the tool only while it still has it (trust on first use).
--
--   - A side table, not a column: the router before this one reads connector_connections
--     through bun.
--   - connected_at is the connection's connected_at when the pin was taken. A pin counts only
--     while the connection still has it: a reconnect begins a new grant, and the next session
--     pins again (store.PinConnectorTools).
--   - The rows go with their connection when it is hard deleted, and the store deletes them
--     when it soft deletes one (store.DeleteConnectorConnection).
--
-- A data move does not carry these rows (store.dataTables): the deployment a connection moves
-- to pins again on first use.
CREATE TABLE connector_tool_pins (
    connection_id TEXT NOT NULL REFERENCES connector_connections (id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    schema_digest TEXT NOT NULL,
    connected_at TIMESTAMPTZ,
    pinned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (connection_id, tool_name)
);

-- +goose Down
DROP TABLE connector_tool_pins;
