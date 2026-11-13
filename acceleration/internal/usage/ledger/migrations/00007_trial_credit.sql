-- +goose Up
-- A trial is the credit a new organization is given to try the product with. trial_micros is
-- what of granted_micros is trial credit, net of its revocations, so an organization is funded
-- when it has credit that is not trial credit.
ALTER TABLE credit_entries DROP CONSTRAINT credit_entries_kind_check;
ALTER TABLE credit_entries ADD CONSTRAINT credit_entries_kind_check CHECK (kind IN
    ('top_up', 'grant', 'trial', 'refund', 'dispute', 'dispute_reversal', 'adjustment'));
ALTER TABLE credit_entries DROP CONSTRAINT credit_entries_sign_check;
ALTER TABLE credit_entries ADD CONSTRAINT credit_entries_sign_check CHECK (
    (kind IN ('top_up', 'grant', 'trial', 'dispute_reversal') AND amount_micros > 0)
    OR (kind IN ('refund', 'dispute') AND amount_micros < 0)
    OR (kind = 'adjustment' AND amount_micros <> 0));
ALTER TABLE accounts ADD COLUMN trial_micros BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE accounts DROP COLUMN trial_micros;
ALTER TABLE credit_entries DROP CONSTRAINT credit_entries_sign_check;
ALTER TABLE credit_entries ADD CONSTRAINT credit_entries_sign_check CHECK (
    (kind IN ('top_up', 'grant', 'dispute_reversal') AND amount_micros > 0)
    OR (kind IN ('refund', 'dispute') AND amount_micros < 0)
    OR (kind = 'adjustment' AND amount_micros <> 0));
ALTER TABLE credit_entries DROP CONSTRAINT credit_entries_kind_check;
ALTER TABLE credit_entries ADD CONSTRAINT credit_entries_kind_check CHECK (kind IN
    ('top_up', 'grant', 'refund', 'dispute', 'dispute_reversal', 'adjustment'));
