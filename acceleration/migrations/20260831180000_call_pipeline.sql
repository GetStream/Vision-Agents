-- +goose Up

-- What the call was actually run with. A config decides these until a session overrides
-- one, so neither the config nor the request answers "what spoke on this call" on its own.
-- They are written onto the row when the agent joins, which is also what makes a finished
-- call able to say what it used after the process that ran it is gone.
--
-- These are the targets asked for rather than the provider each turn resolved to: a
-- shortcut such as llm-fast is several models, and routing fails over between them
-- mid-call. What each individual turn cost and reached lives in requests.
ALTER TABLE calls ADD COLUMN stt TEXT NOT NULL DEFAULT '';
ALTER TABLE calls ADD COLUMN tts TEXT NOT NULL DEFAULT '';
ALTER TABLE calls ADD COLUMN llm TEXT NOT NULL DEFAULT '';
ALTER TABLE calls ADD COLUMN subagent TEXT NOT NULL DEFAULT '';
ALTER TABLE calls ADD COLUMN instructions TEXT NOT NULL DEFAULT '';
-- skills carries the names offered on this call, resolved from the config's list or the
-- built-in set. The instructions behind a name live in the skill registry.
ALTER TABLE calls ADD COLUMN skills JSONB;

-- +goose Down
ALTER TABLE calls DROP COLUMN stt;
ALTER TABLE calls DROP COLUMN tts;
ALTER TABLE calls DROP COLUMN llm;
ALTER TABLE calls DROP COLUMN subagent;
ALTER TABLE calls DROP COLUMN instructions;
ALTER TABLE calls DROP COLUMN skills;
