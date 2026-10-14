-- +goose Up
-- A session's custom filter is containment only (custom @> ...), which jsonb_path_ops serves
-- with a smaller index than the default jsonb_ops.
DROP INDEX agent_sessions_custom_idx;
CREATE INDEX agent_sessions_custom_idx ON agent_sessions USING GIN (custom jsonb_path_ops);

-- +goose Down
DROP INDEX agent_sessions_custom_idx;
CREATE INDEX agent_sessions_custom_idx ON agent_sessions USING GIN (custom);
