-- +goose Up

-- A skill may be a folder: SKILL.md plus the files beside it, such as references the model
-- reads when it needs them and scripts it runs in the sandbox. files maps each one's path,
-- relative to the folder, to its text.
ALTER TABLE skills ADD COLUMN files JSONB NOT NULL DEFAULT '{}';

-- +goose Down
ALTER TABLE skills DROP COLUMN files;
