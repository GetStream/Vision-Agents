-- +goose Up

-- contact_map is the contact map (T43, AI-883; channels.md on connectors/planning, «How the
-- agent knows it is the same person»): one person, as one agent of a customer knows them, to
-- the person's omni-channel, the agent channel that gets one episode card for each call and
-- each external thread (T41). One row per address a person is reached at.
--
--   - kind and address are how the person is known. phone: the E.164 number, the same for
--     a call, an SMS, a WhatsApp message and an iMessage from one number, so they share one
--     row and one omni-channel. slack: «<team id>:<user id>», since Slack gives no number;
--     account linking (later) points such a row at a phone row's omni-channel.
--   - conversation_id is the omni-channel's cid, agent:omni-<uuid>. It never holds the
--     number: a channel id is visible to every member of the channel and to every client
--     of the Stream app (channels.md, «Risks»).
--   - user_id is the end user the address is linked to, for an agent that must know who is
--     writing before it answers (internal/channels' link codes, channel_links). NULL until
--     linked. T62 (AI-921) moves channel_identities onto this table and fills it.
--   - stream_app_pk is the Stream app the omni-channel is in, pinned when the row is made,
--     as channel_threads pins its thread channel. NULL is the deployment's own.
--
-- A data move does not carry these rows (store.dataTables): the omni-channels live in the
-- Stream app, which a move does not carry either.
CREATE TABLE contact_map (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    agent_config_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('phone', 'slack')),
    address TEXT NOT NULL CHECK (address <> ''),
    conversation_id TEXT NOT NULL,
    user_id TEXT,
    stream_app_pk BIGINT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- One row for each address of each agent of a customer (store.MapContact): the same number
-- from an SMS and from a call is one row.
CREATE UNIQUE INDEX contact_map_address
    ON contact_map (customer_id, agent_config_id, kind, address);

-- +goose Down
DROP TABLE contact_map;
