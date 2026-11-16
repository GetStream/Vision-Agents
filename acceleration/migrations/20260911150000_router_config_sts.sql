-- +goose Up

-- sts is the speech-to-speech block of a router config: one native audio model that hears
-- the caller and speaks back, in place of the transcriber, the text model and the voice
-- the three blocks before it configure. A JSONB column like the others, for the same
-- reason: nothing queries inside it, and the options are the modality's own.
ALTER TABLE router_configs ADD COLUMN sts JSONB NOT NULL DEFAULT '{}';

-- +goose Down
ALTER TABLE router_configs DROP COLUMN sts;
