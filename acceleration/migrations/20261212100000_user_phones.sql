-- +goose NO TRANSACTION
-- +goose Up

-- user_phones is what a customer's backend says about one of its users
-- (PUT /v1/agents/users/{user_id}): the Stream user user_id of the customer's app and the phone
-- number phone are one person, so their app chat and their SMS, calls and other messages share one
-- omni-channel with each agent (store.LinkUserPhone, store.MapContact).
--
--   - One pair per Stream app: a user has one number and a number one user within
--     (customer_id, stream_app_pk). The same user id in another app of the customer is another
--     person.
--   - stream_app_pk is the app the link was made for: the pin the request acts in
--     (streamapp.Clients.Pin), 0 for the deployment's own. Not NULL, so the unique indexes hold
--     for the deployment's own app too.
--   - phone is E.164, as omnichannel.Phone reads it and contact_map keeps it. It is never part of
--     a channel id.
--
-- A new table that no deployed router reads or writes, and a new index on contact_map that no
-- deployed router depends on: expand-only. A data move does not carry the table
-- (store.dataTables), as it carries no contact_map row it would join. Outside a transaction,
-- so the index is built without stopping contact_map's writes; every statement is IF NOT
-- EXISTS. A CONCURRENTLY build that failed leaves its index invalid, which a run again skips
-- once goose has recorded the version: store.usageIndexes names the index, and each start
-- drops an invalid one and builds it again (store.ensureUsageIndexes).
CREATE TABLE IF NOT EXISTS user_phones (
    customer_id TEXT NOT NULL,
    stream_app_pk BIGINT NOT NULL,
    user_id TEXT NOT NULL CHECK (user_id <> ''),
    phone TEXT NOT NULL CHECK (phone <> ''),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (customer_id, stream_app_pk, user_id)
);

-- A number linked to another user of the same app is refused (409 phone_taken).
CREATE UNIQUE INDEX IF NOT EXISTS user_phones_phone ON user_phones (customer_id, stream_app_pk, phone);

-- A link's join finds the rows on one omni-channel of an agent (store.joinLinked's born) by
-- this, not by reading every row of the agent:
-- on the message path for a linked person's first row, and on every PUT. Until it is built,
-- those reads are slower, not wrong.
CREATE INDEX CONCURRENTLY IF NOT EXISTS contact_map_conversation
    ON contact_map (customer_id, agent_config_id, conversation_id);

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS contact_map_conversation;
DROP TABLE IF EXISTS user_phones;
