-- +goose Up
ALTER TABLE agent_configs ADD COLUMN subagents JSONB NOT NULL DEFAULT '{}';
ALTER TABLE agent_configs ADD COLUMN video_source TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_configs ADD COLUMN video_max_frames INTEGER NOT NULL DEFAULT 1;
ALTER TABLE skills ADD COLUMN subagent TEXT NOT NULL DEFAULT '';
ALTER TABLE skills ADD COLUMN capture_video BOOLEAN NOT NULL DEFAULT FALSE;

-- +goose Down
ALTER TABLE skills DROP COLUMN capture_video;
ALTER TABLE skills DROP COLUMN subagent;
ALTER TABLE agent_configs DROP COLUMN video_max_frames;
ALTER TABLE agent_configs DROP COLUMN video_source;
ALTER TABLE agent_configs DROP COLUMN subagents;
