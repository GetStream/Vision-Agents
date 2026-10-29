-- +goose Up
-- The agent config that answers the number's calls. Its id, not its name: a config can be
-- looked up by name, and a stored name would follow a new config made under the same one.
ALTER TABLE phone_numbers ADD COLUMN config_id TEXT;

-- +goose Down
ALTER TABLE phone_numbers DROP COLUMN config_id;
