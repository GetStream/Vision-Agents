-- +goose NO TRANSACTION
-- +goose Up

-- user_phone_history is how much of a linked number's past its user reads
-- (PUT /v1/agents/users/{user_id}, include_phone_history and phone_history_since): what the
-- number said before since is not the user's, NULL since is everything it ever said
-- (store.EpisodeCards, store.ThreadHistorySince).
--
--   - One row per link, keyed as user_phones is; an unlink deletes it with the link.
--   - phone is the number the row is of. A row of another number than the link's is not read:
--     a router that knows no such row (v0.6.284) may move a link to a new number, and then the
--     link's time (user_phones.updated_at) is the boundary, as with no row.
--
-- Not a column of user_phones: v0.6.284 upserts it with RETURNING *, which a new column would
-- break. A new table and a new index no deployed router reads or writes: expand-only. Outside
-- a transaction, so the index is built without stopping contact_map's writes (store.usageIndexes
-- builds it again where it was left invalid).
CREATE TABLE IF NOT EXISTS user_phone_history (
    customer_id TEXT NOT NULL,
    stream_app_pk BIGINT NOT NULL,
    user_id TEXT NOT NULL,
    phone TEXT NOT NULL,
    since TIMESTAMPTZ,
    PRIMARY KEY (customer_id, stream_app_pk, user_id),
    FOREIGN KEY (customer_id, stream_app_pk, user_id) REFERENCES user_phones ON DELETE CASCADE
);

-- A link whose boundary is before it finds the number's rows a link set aside
-- (retired:<row id>:<number>) by their number (store.joinLinked, store.detachLinked).
CREATE INDEX CONCURRENTLY IF NOT EXISTS contact_map_retired
    ON contact_map (customer_id, (regexp_replace(address, '^retired:[^:]*:', '')))
    WHERE address LIKE 'retired:%';

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS contact_map_retired;
DROP TABLE IF EXISTS user_phone_history;
