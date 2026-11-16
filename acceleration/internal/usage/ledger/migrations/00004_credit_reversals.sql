-- +goose Up
-- The entry an adjustment reverses: a grant's revocation names the grant, an unrevocation the
-- revocation, and so on. Nullable, so adding it rewrites no row. The unique index lets an
-- entry be reversed once, whoever writes at the same time.
ALTER TABLE credit_entries ADD COLUMN reverses_id BIGINT REFERENCES credit_entries (id);
CREATE UNIQUE INDEX credit_entries_reverses_idx ON credit_entries (reverses_id) WHERE reverses_id IS NOT NULL;

-- +goose Down
DROP INDEX credit_entries_reverses_idx;
ALTER TABLE credit_entries DROP COLUMN reverses_id;
