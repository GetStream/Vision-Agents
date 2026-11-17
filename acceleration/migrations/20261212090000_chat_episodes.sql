-- +goose Up

-- A message a person writes to an agent in the app chat (Stream Chat) opens an episode of its
-- own, under an agent config with episode_cards on (api.routeArrivingMessage):
--
--   - contact_map.kind takes stream_user: the person is the Stream user who wrote the message,
--     message.user.id, which Stream signed with the customer's own token. The address is the
--     user id.
--   - episodes.source takes chat: the card of a run of messages in one agent channel of the
--     app chat. Its thread_channel is that channel; it has no call and no session.
--
-- Each CHECK is dropped and added again with the one new value beside the old ones. It is
-- expand-only: no deployed router writes stream_user or chat, every row meets the old CHECK,
-- whose values the new one keeps, so it narrows nothing. NOT VALID, as 20261101130000 adds its
-- CHECK: the table is not read again to prove what the old CHECK already did, and the CHECK
-- still refuses an unknown value for every row written from now on. Both statements run in
-- goose's one transaction, so no moment allows another value. A lock waits at most 5 s behind
-- a running query rather than queueing every card write.
SET LOCAL lock_timeout = '5s';
ALTER TABLE contact_map
    DROP CONSTRAINT contact_map_kind_check,
    ADD CONSTRAINT contact_map_kind_check CHECK (kind IN ('phone', 'slack', 'stream_user')) NOT VALID;
ALTER TABLE episodes
    DROP CONSTRAINT episodes_source_check,
    ADD CONSTRAINT episodes_source_check CHECK (source IN ('call', 'sms', 'whatsapp', 'slack', 'imessage', 'chat')) NOT VALID;

-- +goose Down
-- The chat episodes and the stream_user contacts are deleted, then the old CHECKs restored and
-- validated. Left NOT VALID they would still be checked on every update of a row, so the idle
-- sweeper's close of an open chat episode would fail, and with it each sweep. The cards stay in
-- the omni-channels, which live in Stream Chat; episode_activity rows go with their episodes
-- (ON DELETE CASCADE).
SET LOCAL lock_timeout = '5s';
DELETE FROM episodes WHERE source = 'chat';
DELETE FROM contact_map WHERE kind = 'stream_user';
ALTER TABLE episodes
    DROP CONSTRAINT episodes_source_check,
    ADD CONSTRAINT episodes_source_check CHECK (source IN ('call', 'sms', 'whatsapp', 'slack', 'imessage'));
ALTER TABLE contact_map
    DROP CONSTRAINT contact_map_kind_check,
    ADD CONSTRAINT contact_map_kind_check CHECK (kind IN ('phone', 'slack'));
