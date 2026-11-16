-- +goose Up

-- The guest users this app handed out, and what became of them.
--
-- A guest is a real Stream user with role guest, so the chat and video sides need nothing
-- recorded here to work. This table exists for the other half of the story: somebody who
-- talked to an agent before signing up, and then signed up. Claiming moves what the guest
-- said onto the account, and that has to be something only a backend can ask for -- a page
-- allowed to claim guests could claim anybody's -- which means there has to be a row saying
-- which ids were ever guests of this customer and which have already been claimed.
CREATE TABLE guest_users (
    -- id is the Stream user id, which the caller holds and sends back to claim.
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    custom JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- claimed_by is the real user this guest turned out to be, null while they are still a
    -- guest. A guest is claimed once: a second claim naming a different user would move one
    -- person's conversations onto another's account.
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ
);

CREATE INDEX guest_users_customer_idx ON guest_users (customer_id, created_at DESC);
CREATE INDEX guest_users_claimed_idx ON guest_users (customer_id, claimed_by)
    WHERE claimed_by IS NOT NULL;

-- +goose Down
DROP TABLE guest_users;
