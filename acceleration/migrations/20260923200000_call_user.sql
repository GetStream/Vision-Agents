-- +goose Up

-- Who the agent spoke to, as the client's own token named them. Empty for a call placed
-- by the customer's backend, and for telephony, where the number is the name.
ALTER TABLE calls ADD COLUMN user_id TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE calls DROP COLUMN user_id;
