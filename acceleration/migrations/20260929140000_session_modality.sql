-- +goose Up
ALTER TABLE agent_sessions ADD COLUMN modality text NOT NULL DEFAULT 'voice';
UPDATE agent_sessions SET modality = 'text' WHERE call_id IS NULL;
CREATE INDEX agent_sessions_modality_idx ON agent_sessions (customer_id, modality, updated_at DESC, id DESC);

-- +goose Down
DROP INDEX agent_sessions_modality_idx;
ALTER TABLE agent_sessions DROP COLUMN modality;
