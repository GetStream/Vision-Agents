-- +goose Up

-- Every end user this app has been seen acting for, rather than only the guests it minted.
--
-- 20260918120100_guest_users.sql recorded guests alone, because claiming was the only thing
-- that needed a row. An app that wants to know who its users are needs the other ones too,
-- and a kind beside the id is what keeps the claim path honest: only a guest may be claimed,
-- and an account that signed up is not one.
--
-- The key is (customer_id, id) rather than id alone. A user id is the customer's, not this
-- deployment's, so two apps naming the same user is two people, which guest_users could not
-- hold.
CREATE TABLE users (
    customer_id TEXT NOT NULL,
    -- id is the Stream user id, which the caller holds and sends back to claim.
    id TEXT NOT NULL,
    -- kind is what the credential proved them to be: guest or authenticated. An anonymous
    -- caller is never recorded, since the name it goes by is one nobody checked.
    kind TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    custom JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- claimed_by is the real user this guest turned out to be, null while they are still a
    -- guest. A guest is claimed once: a second claim naming a different user would move one
    -- person's conversations onto another's account.
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    PRIMARY KEY (customer_id, id)
);

INSERT INTO users (customer_id, id, kind, name, custom, created_at, claimed_by, claimed_at)
SELECT customer_id, id, 'guest', name, custom, created_at, claimed_by, claimed_at
FROM guest_users;

CREATE INDEX users_customer_idx ON users (customer_id, created_at DESC);
CREATE INDEX users_claimed_idx ON users (customer_id, claimed_by)
    WHERE claimed_by IS NOT NULL;

DROP TABLE guest_users;

-- +goose Down
CREATE TABLE guest_users (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    custom JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ
);

-- One row per id, since guest_users cannot hold the same guest under two apps.
INSERT INTO guest_users (id, customer_id, name, custom, created_at, claimed_by, claimed_at)
SELECT DISTINCT ON (id) id, customer_id, name, custom, created_at, claimed_by, claimed_at
FROM users
WHERE kind = 'guest'
ORDER BY id, created_at;

CREATE INDEX guest_users_customer_idx ON guest_users (customer_id, created_at DESC);
CREATE INDEX guest_users_claimed_idx ON guest_users (customer_id, claimed_by)
    WHERE claimed_by IS NOT NULL;

DROP TABLE users;
