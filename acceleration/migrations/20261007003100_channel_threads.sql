-- +goose Up

-- channel_threads links one external thread to the thread channel the channel bridge writes it
-- into in Stream Chat (T57, AI-878; channels.md on connectors/planning, «Who moves messages:
-- the channel bridge», step 4). One row per thread: a Slack thread is a channel and a
-- thread_ts.
--
--   - channel_id is the thread channel's id in the agent channel type (cid agent:<channel_id>).
--     The bridge makes it, so a message the message hook delivers is found by it alone.
--   - customer_id, connector_id, provider_unit_id and thread_key name the external thread, as
--     the verifier read it (core.InboundMessage). A provider unit can be shared: one Slack
--     workspace can install two customers' apps, so the customer is part of the key.
--   - connection_id is the app-owned connection replies are sent with, kept current by the
--     latest message on the thread.
--   - thread_parts are the thread key's named parts (core.ChannelMessage.ThreadParts), which a
--     reply template names as {thread.<name>}.
--   - stream_app_pk is the Stream app the thread channel is in: the provider app's pin
--     (connector_oauth_clients.stream_app_pk, 20261006195500). NULL is the deployment's own.
--   - last_inbound_at is when the person last wrote, for a reply window (core.ReplyValues).
--
-- A data move does not carry these rows (store.dataTables): the thread channels live in the
-- Stream app, which a move does not carry either.
CREATE TABLE channel_threads (
    channel_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    provider_unit_id TEXT NOT NULL,
    thread_key TEXT NOT NULL,
    connection_id TEXT NOT NULL,
    thread_parts JSONB NOT NULL DEFAULT '{}',
    stream_app_pk BIGINT,
    last_inbound_at TIMESTAMPTZ NOT NULL,
    -- turn_holder and turn_until are the lease on the thread's one running turn: the router
    -- that answers a message in the thread holds it until the reply is done or the lease
    -- runs out, so two routers never answer one thread at once (store.TakeChannelThreadTurn).
    -- NULL while no turn runs.
    turn_holder TEXT,
    turn_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One thread channel for each external thread (store.LinkChannelThread).
CREATE UNIQUE INDEX channel_threads_thread
    ON channel_threads (customer_id, connector_id, provider_unit_id, thread_key);

-- channel_thread_messages is every message of a thread channel already acted on, so a
-- message delivered twice is acted on once (store.ClaimChannelThreadMessage). kind says which
-- step took it, each with its own id:
--   - inbound: a provider's message the bridge took, by the provider's id for it
--     (core.InboundMessage.ProviderMessageID). Slack retries an event three times
--     (https://docs.slack.dev/apis/events-api/, «Retries»). Slack's ts is unique only within a
--     channel («the unique (per-channel) timestamp», https://docs.slack.dev/reference/events/message),
--     and a thread channel is one Slack channel's thread, so with the channel it is unique.
--   - turn: a person's message the message hook handed to the session, by its Stream Chat id,
--     since Stream may deliver a message.new more than once.
--   - reply: an agent reply the bridge sent to the external thread, by its Stream Chat id,
--     since a finished reply is told again when it is written again.
CREATE TABLE channel_thread_messages (
    channel_id TEXT NOT NULL REFERENCES channel_threads (channel_id),
    kind TEXT NOT NULL CHECK (kind IN ('inbound', 'turn', 'reply')),
    message_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (channel_id, kind, message_id)
);

-- +goose Down
DROP TABLE channel_thread_messages;
DROP TABLE channel_threads;
