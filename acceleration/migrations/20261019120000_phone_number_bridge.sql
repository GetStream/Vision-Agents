-- +goose Up
-- The Stream trunk an attached number's calls go to, with its credentials. Stream returns the
-- password only when the trunk is made, and a vendor that asks per call where to send a caller
-- (Telnyx) needs it then. The password is sealed, bound to the customer and the number.
ALTER TABLE phone_numbers
    ADD COLUMN IF NOT EXISTS bridge_uri text,
    ADD COLUMN IF NOT EXISTS bridge_username text,
    ADD COLUMN IF NOT EXISTS bridge_password_sealed bytea,
    ADD COLUMN IF NOT EXISTS bridge_password_kek_version integer;

-- +goose Down
ALTER TABLE phone_numbers
    DROP COLUMN IF EXISTS bridge_uri,
    DROP COLUMN IF EXISTS bridge_username,
    DROP COLUMN IF EXISTS bridge_password_sealed,
    DROP COLUMN IF EXISTS bridge_password_kek_version;
