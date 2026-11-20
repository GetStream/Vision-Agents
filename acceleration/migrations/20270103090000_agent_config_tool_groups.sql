-- +goose NO TRANSACTION
-- +goose Up

-- tool_groups are the functions an SDK provides that a config's sessions may offer, each
-- group with how it is loaded and how long what its tools were given and gave back is kept
-- (store.AgentConfig.ToolGroups). Null, as on every config made before them, names none, so
-- such a config's sessions offer no function until it names one. No default, so adding the
-- column rewrites no row; the lock waits at most 5 s behind a running query, past which the
-- migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS tool_groups JSONB;
END $$;
-- +goose StatementEnd

-- What progressive_tools and tool_output_retention said of a config's connector tools moves
-- onto each of its connector bindings, as loading summary and output_retention. A binding
-- that already says either keeps what it says. Both columns stay, unread from here on, for
-- a router still running the release before.
UPDATE agent_configs AS config
SET connectors = (
    SELECT jsonb_agg(
        binding
            || CASE WHEN config.progressive_tools AND NOT binding ? 'loading'
                    THEN '{"loading": "summary"}'::jsonb ELSE '{}'::jsonb END
            || CASE WHEN config.tool_output_retention <> '' AND NOT binding ? 'output_retention'
                    THEN jsonb_build_object('output_retention', config.tool_output_retention) ELSE '{}'::jsonb END
        ORDER BY position)
    FROM jsonb_array_elements(config.connectors) WITH ORDINALITY AS bound(binding, position)
)
WHERE (config.progressive_tools OR config.tool_output_retention <> '')
  AND jsonb_typeof(config.connectors) = 'array'
  AND jsonb_array_length(config.connectors) > 0;

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_configs DROP COLUMN IF EXISTS tool_groups;
END $$;
-- +goose StatementEnd
