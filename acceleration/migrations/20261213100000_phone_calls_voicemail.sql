-- +goose Up

-- A placed call an answering machine picked up ends as voicemail rather than completed.
ALTER TABLE phone_calls DROP CONSTRAINT phone_calls_outcome_check;
ALTER TABLE phone_calls ADD CONSTRAINT phone_calls_outcome_check
    CHECK (outcome IN ('completed', 'busy', 'no_answer', 'failed', 'canceled', 'unknown', 'voicemail'));

-- +goose Down

-- A router built before voicemail reads a call a machine took as what it was before: completed.
UPDATE phone_calls SET outcome = 'completed' WHERE outcome = 'voicemail';
ALTER TABLE phone_calls DROP CONSTRAINT phone_calls_outcome_check;
ALTER TABLE phone_calls ADD CONSTRAINT phone_calls_outcome_check
    CHECK (outcome IN ('completed', 'busy', 'no_answer', 'failed', 'canceled', 'unknown'));
