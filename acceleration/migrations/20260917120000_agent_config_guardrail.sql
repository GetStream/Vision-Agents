-- +goose Up

-- guardrail is a whole guardrail.md: frontmatter saying how to screen a turn - a
-- classifier, the customer's own webhook, or a model asked to judge - then the policy
-- itself in prose. It is stored as the file rather than as parsed columns so that what the
-- agent's directory says and what the session enforces cannot drift apart, and so that a
-- policy stays something a human reads and edits.
--
-- Empty, which every existing config is, means every turn is answered.
ALTER TABLE agent_configs ADD COLUMN guardrail TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN guardrail;
