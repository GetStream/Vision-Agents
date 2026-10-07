-- +goose Up

-- episodes is one call, or one run of messages on one external thread, and the card it has
-- in the person's omni-channel (T41, AI-883; channels.md on connectors/planning, «The episode
-- card»). The card is one Stream Chat message, so an episode takes one place in the 200
-- messages a conversation is read back by (chatlog.transcriptLimit).
--
--   - contact_id is the contact map row of the person the episode is with; the card is in
--     its omni-channel (contact_map.conversation_id).
--   - source is what the episode came in on: call, sms, whatsapp, slack or imessage. The
--     card carries it as its source field, which is why the message hook never answers a
--     card (api.addressed).
--   - thread_channel is the cid of the channel with the raw text: the thread channel the
--     channel bridge writes a thread into, or the call channel a call's transcript is in.
--   - call_id and session_id are the Stream call and the router session of a call; NULL
--     for a thread.
--   - card_message_id is the card's Stream Chat message id, for the updates that close and
--     summarize it (T55).
--   - status is in_progress until T55 closes the episode: ended, then summarized or
--     summary_failed.
--   - stream_app_pk is the Stream app the omni-channel is in (contact_map.stream_app_pk).
--
-- A data move does not carry these rows (store.dataTables), for the reason contact_map gives.
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    contact_id TEXT NOT NULL REFERENCES contact_map (id),
    source TEXT NOT NULL CHECK (source IN ('call', 'sms', 'whatsapp', 'slack', 'imessage')),
    thread_channel TEXT NOT NULL,
    call_id TEXT,
    session_id TEXT,
    card_message_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('in_progress', 'ended', 'summarized', 'summary_failed')),
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    stream_app_pk BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((source = 'call') = (session_id IS NOT NULL))
);

-- One open episode for each thread (store.OpenEpisode): the second and third message of a
-- thread find the first one's episode, so a thread has one card until T55 closes it.
CREATE UNIQUE INDEX episodes_open_thread
    ON episodes (customer_id, thread_channel)
    WHERE status = 'in_progress' AND session_id IS NULL;

-- One episode for each call session. A call id need not be one call: the default routing
-- rule names the call after the number rung, phone-{{called_number}} (phone.Stream.CreateRoute),
-- so two calls to one number can share a call id and a call channel. Whether Stream reuses
-- the call for the second caller is unverified; the session is what a call episode is keyed
-- by either way.
CREATE UNIQUE INDEX episodes_call_session ON episodes (session_id);

-- The episodes of a person, newest first, for reading the cards (T56).
CREATE INDEX episodes_contact ON episodes (contact_id, started_at DESC);

-- +goose Down
DROP TABLE episodes;
