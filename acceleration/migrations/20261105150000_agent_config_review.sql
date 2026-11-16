-- +goose Up

-- review_instructions is what the model reviewing a finished call is told to judge it by
-- (store.AgentConfig.ReviewInstructions). Empty, as every config starts, is Stream's own
-- scoring guide.
ALTER TABLE agent_configs ADD COLUMN review_instructions TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN review_instructions;
