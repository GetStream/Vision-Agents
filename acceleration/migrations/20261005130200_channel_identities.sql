-- +goose Up

-- channel_identities is who a number is, for one agent. A phone number means nothing on its
-- own: an agent that reads a person's own calendar needs to know which end user is writing,
-- and that is settled once with a code rather than on every message. The conversation is
-- kept here too, so the next message carries on the one before rather than starting over.
CREATE TABLE channel_identities (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    address TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- One row per number per agent: the same person writing to two agents is two conversations.
CREATE UNIQUE INDEX channel_identities_address_idx
    ON channel_identities (customer_id, config_id, kind, address);

-- channel_links are the codes that tie a number to an end user. The app asks for one, shows
-- it to somebody already signed in, and the number that texts it back is theirs from then on.
CREATE TABLE channel_links (
    code TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ
);

-- +goose Down
DROP TABLE channel_links;
DROP TABLE channel_identities;
