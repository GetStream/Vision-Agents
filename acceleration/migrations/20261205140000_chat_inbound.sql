-- +goose Up

-- chat_inbound is the queue of Stream Chat messages the router answers: a message.new of
-- an agent channel that Stream Chat delivered to the router, recorded before the router told
-- Stream it took it, so an accepted message survives a router restart. A worker on any router
-- answers it later, with nothing of the delivery's request left, so a row holds what
-- answering it takes.
--
--   - stream_app and message_id are the Stream app and its id for the message, unique within
--     the app: one row per pair, so a delivery Stream makes again queues nothing.
--   - customer_id, deployment and api_key are, with stream_app, where the delivery came from
--     as the router verified it (api.hookOrigin): the registered app's customer, empty when
--     the router does not know it; whether the deployment's own app sent it; the key whose
--     secret signed it. The app is always known: the internal endpoint reads it from the signed
--     body, and the store refuses an insert without one.
--   - body is the verified event as Stream sent it.
--   - state is queued until the message is answered (done) or its retries ran out (failed).
--     next_attempt_at is when a queued row is due; a worker that takes one moves it to the
--     end of its lease, so another router takes it only once that lease ran out.
--   - received_at is when the row was queued, and finished_at when it left queued. Done rows
--     are kept a day, failed ones 30 days (store.PruneChatInbound).
--
-- A data move does not carry these rows (store.dataTables): they are in flight for minutes.
CREATE TABLE chat_inbound (
    seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    stream_app BIGINT NOT NULL,
    message_id TEXT NOT NULL,
    customer_id TEXT NOT NULL DEFAULT '',
    deployment BOOLEAN NOT NULL,
    api_key TEXT NOT NULL DEFAULT '',
    body BYTEA NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued' CHECK (state IN ('queued', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    UNIQUE (stream_app, message_id)
);

-- The queued rows by when they are due (store.ClaimChatInbound, store.NextChatInboundAt).
CREATE INDEX chat_inbound_queued ON chat_inbound (next_attempt_at, seq) WHERE state = 'queued';

-- The rows past their keep (store.PruneChatInbound).
CREATE INDEX chat_inbound_finished ON chat_inbound (finished_at) WHERE state <> 'queued';

-- +goose Down
DROP TABLE chat_inbound;
