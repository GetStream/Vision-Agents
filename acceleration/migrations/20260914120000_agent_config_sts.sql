-- +goose Up

-- sts names one native audio model that hears the caller and speaks back, in place of the
-- transcriber, the conversation model and the voice. A config that names one is a native
-- agent: the session opens that one model and none of the three, so the cascade targets
-- on the same row are left alone rather than defaulted. Empty, which every existing
-- config is, means the cascade.
ALTER TABLE agent_configs ADD COLUMN sts TEXT NOT NULL DEFAULT '';
-- The same on the call row, for the same reason stt, tts and llm are there: what a
-- finished call actually ran with, after a session's overrides were folded in.
ALTER TABLE calls ADD COLUMN sts TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE calls DROP COLUMN sts;
ALTER TABLE agent_configs DROP COLUMN sts;
