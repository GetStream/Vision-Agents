-- +goose Up

-- channel_thread_people holds the one person a chat thread is with, as its messages name their
-- author (core.InboundMessage.AuthorID): the sender of a Linq chat whose latest message says it
-- is no group (data.chat.is_group false). The thread key of such a thread is the chat id, which
-- names nobody, so this row is what makes it one person's thread, whose session may schedule
-- tasks for them (api.threadPerson). No row for a group chat, for a chat whose message does not
-- say, for every thread linked before this table, and for every other connector: an SMS or
-- WhatsApp thread key is the person's number already, and a Slack thread is no one person's. No
-- row offers no scheduling tools.
--
-- A table of its own, not a channel_threads column: a router built before this one reads
-- channel_threads rows back whole (store.LinkChannelThread, RETURNING *) and fails on a column
-- it does not know, so a rollback would stop every thread channel. It never reads this table.
--
-- A data move does not carry these rows (store.dataTables), as it does not carry the
-- channel_threads rows they belong to.
CREATE TABLE channel_thread_people (
    channel_id TEXT PRIMARY KEY REFERENCES channel_threads (channel_id) ON DELETE CASCADE,
    chat_person TEXT NOT NULL
);

-- +goose Down
DROP TABLE channel_thread_people;
