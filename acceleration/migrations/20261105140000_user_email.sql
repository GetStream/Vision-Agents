-- +goose Up

-- email is the address the app's backend last said this user signs in with
-- (auth.EmailHeader, AI-1543), null while it has said none. Only a server-side
-- caller may say it. A connector whose manifest sets owner_email compares it with
-- the provider account on a user-owned connection's first consent.
ALTER TABLE users ADD COLUMN email TEXT;

-- +goose Down
ALTER TABLE users DROP COLUMN email;
