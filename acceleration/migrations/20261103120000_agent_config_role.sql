-- +goose Up

-- role is what an agent config is for in the app around it, beyond answering people
-- (store.AgentConfig.Role). Empty for an ordinary agent. agent_moderator marks the app's Nora,
-- the agent the chat worker hands flagged messages to for review.
ALTER TABLE agent_configs ADD COLUMN role TEXT NOT NULL DEFAULT '';

-- The dashboard marked Nora with the tag role=ai_moderator before there was a column: the
-- newest live config so tagged in each app takes the role. A test copy (tagged draft_of)
-- carries the tag of the config it copies, so it is left out, as it is from the index below.
UPDATE agent_configs SET role = 'agent_moderator'
WHERE id IN (
    SELECT DISTINCT ON (customer_id) id FROM agent_configs
    WHERE tags->>'role' = 'ai_moderator'
      AND deleted_at IS NULL
      AND coalesce(tags->>'draft_of', '') = ''
    ORDER BY customer_id, created_at DESC
);

-- One live config per app holds a role, so the worker looking for the app's Nora finds one.
-- A test copy may share its config's role, and a deleted config frees it.
CREATE UNIQUE INDEX agent_configs_role_idx ON agent_configs (customer_id, role)
    WHERE role <> '' AND deleted_at IS NULL AND coalesce(tags->>'draft_of', '') = '';

-- +goose Down
DROP INDEX agent_configs_role_idx;
ALTER TABLE agent_configs DROP COLUMN role;
