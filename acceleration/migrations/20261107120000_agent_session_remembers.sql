-- +goose Up

-- remembers is whether the session was opened with a memory user id, so it could have taught
-- the memory store anything. Deleting one that did not skips asking the store, which counts
-- every list against the account's quota. NULL, as in every session made before it, is not
-- known, and is forgotten as before. Nullable and with no default, so adding it rewrites no
-- row.
ALTER TABLE agent_sessions ADD COLUMN remembers BOOLEAN;

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN remembers;
