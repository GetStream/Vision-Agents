-- +goose Up

-- channel_thread_inbound is the queue of inbound messages the channel bridge writes into their
-- thread channels. The events route queues a message before it answers the provider, and a
-- worker on any router writes it, so a message the provider was told was taken survives a
-- router restart. A row holds what writing the message and opening its episode card take,
-- since another router, or this one after a restart, has nothing else of the delivery.
--
--   - channel_id is the thread channel, and provider_message_id the provider's id for the
--     message (core.InboundMessage): one row per pair, so a provider's retry queues nothing.
--   - seq is the order the messages were queued in. A worker takes only the oldest queued
--     row of a thread channel (store.ClaimChannelInbound), so one thread's messages are
--     written in order and one at a time.
--   - config_id and config_name are the agent config that answers the thread when the
--     message came; author_id, text, media and raw are the message as the verifier read it.
--     media is a JSON array of core.Media; raw is the verified body.
--   - forward is the delivery's forward to the customer's event destinations that take what
--     no agent answered, for the worker to queue when it gives the message up: the delivery's
--     id, the provider's headers and when they stop verifying ({"id", "headers",
--     "headers_until"}). NULL when no destination would take the delivery.
--   - state is queued until the message is written (done) or its retries ran out (failed).
--     next_attempt_at is when a queued row is due; a worker that takes one moves it to the
--     end of its lease, so another router takes it only once that lease ran out.
--   - finished_at is when a row left queued. Done rows are kept a day, failed ones 30 days
--     (store.PruneChannelInbound).
--
-- A data move does not carry these rows (store.dataTables): they point at thread channels,
-- which a move does not carry either, and they are in flight for minutes.
CREATE TABLE channel_thread_inbound (
    seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    channel_id TEXT NOT NULL REFERENCES channel_threads (channel_id) ON DELETE CASCADE,
    provider_message_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    provider_unit_id TEXT NOT NULL,
    thread_key TEXT NOT NULL,
    config_id TEXT NOT NULL,
    config_name TEXT NOT NULL,
    author_id TEXT NOT NULL,
    text TEXT NOT NULL,
    media JSONB NOT NULL DEFAULT '[]',
    raw BYTEA NOT NULL,
    forward JSONB,
    state TEXT NOT NULL DEFAULT 'queued' CHECK (state IN ('queued', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    UNIQUE (channel_id, provider_message_id)
);

-- The oldest queued row of each thread channel, the one a worker may take
-- (store.ClaimChannelInbound, store.NextChannelInboundAt).
CREATE INDEX channel_thread_inbound_queued ON channel_thread_inbound (channel_id, seq) WHERE state = 'queued';

-- The rows past their keep (store.PruneChannelInbound).
CREATE INDEX channel_thread_inbound_finished ON channel_thread_inbound (finished_at) WHERE state <> 'queued';

-- +goose Down
DROP TABLE channel_thread_inbound;
