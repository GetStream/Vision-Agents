-- +goose Up

-- channel_thread_waiting holds a reply on an external thread that no thread channel is linked
-- to yet and that does not speak to the connection's account, for the message that links the
-- thread (AI-990 F31a). Slack retries an event whose delivery failed «nearly immediately»,
-- after 1 minute and after 5 minutes (https://docs.slack.dev/apis/events-api/, «Retries»), so
-- a reply in a mention's thread can arrive before the mention's retry links the thread. Before
-- this table the reply was answered 200 and lost.
--
--   - customer_id, connector_id, provider_unit_id and thread_key name the external thread, as
--     in channel_threads. provider_message_id is the reply's own id (core.InboundMessage).
--   - author_id, text and raw are the message as the verifier read it. raw is the verified
--     body, which the bridge reads the message from again.
--   - A row lives until the message that links its thread takes it, or for at most
--     store.channelWaitingKeep. Only a message that starts its thread takes the rows: every
--     reply in a thread comes after the message that started it, so none of them was written
--     before the bot was spoken to (AI-989).
--
-- A data move does not carry these rows (store.dataTables): they are in flight for minutes.
CREATE TABLE channel_thread_waiting (
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    provider_unit_id TEXT NOT NULL,
    thread_key TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    author_id TEXT NOT NULL,
    text TEXT NOT NULL,
    raw BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, connector_id, provider_unit_id, thread_key, provider_message_id)
);

-- The rows past their keep, which every new row drops first (store.WaitChannelThreadMessage).
CREATE INDEX channel_thread_waiting_created ON channel_thread_waiting (created_at);

-- +goose Down
DROP TABLE channel_thread_waiting;
