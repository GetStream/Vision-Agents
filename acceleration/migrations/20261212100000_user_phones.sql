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
-- A new table that no deployed router reads or writes: expand-only. A data move does not carry
-- it (store.dataTables), as it carries no contact_map row it would join.
CREATE TABLE user_phones (
    customer_id TEXT NOT NULL,
    stream_app_pk BIGINT NOT NULL,
    user_id TEXT NOT NULL CHECK (user_id <> ''),
    phone TEXT NOT NULL CHECK (phone <> ''),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (customer_id, stream_app_pk, user_id)
);

-- A number linked to another user of the same app is refused (409 phone_taken).
CREATE UNIQUE INDEX user_phones_phone ON user_phones (customer_id, stream_app_pk, phone);

-- +goose Down
DROP TABLE user_phones;
