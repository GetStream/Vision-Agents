-- +goose Up

-- A named end user's own policy: scope user, scope_id <app>/<user>, holding a daily quota only.
-- Added NOT VALID: the old CHECK already held for every row, so none is read again. The lock
-- waits at most 5 s behind a running query.
SET LOCAL lock_timeout = '5s';
ALTER TABLE policies DROP CONSTRAINT policies_scope_check;
ALTER TABLE policies ADD CONSTRAINT policies_scope_check
    CHECK (scope IN ('organization', 'app', 'user')) NOT VALID;

-- An app's users with a policy of their own, listed by user id and found by the app's prefix.
-- Byte order, so a prefix match and a range read both use it whatever the database's collation.
CREATE INDEX policies_user_scope_idx ON policies (scope_id COLLATE "C") WHERE scope = 'user';

-- +goose Down

-- The users' rows are kept: the router before this never reads scope user, and a later Up
-- finds them again. The narrower CHECK is NOT VALID, so it holds for new rows only.
SET LOCAL lock_timeout = '5s';
DROP INDEX IF EXISTS policies_user_scope_idx;
ALTER TABLE policies DROP CONSTRAINT policies_scope_check;
ALTER TABLE policies ADD CONSTRAINT policies_scope_check
    CHECK (scope IN ('organization', 'app')) NOT VALID;
