-- +goose Up

-- channel_thread_members holds the people a Linq chat reaches as far as the bridge saw them: the
-- opt-out recipient of each author of a message in it. While one has a live opt-out the chat
-- gets no reply (AI-2015, channelbridge.silenced). A person who never wrote in it is not here.
-- A table of its own, as channel_thread_people: a router built before it reads channel_threads
-- rows back whole and would fail on a new column after a rollback. A data move does not carry
-- it (store.dataTables), as it does not carry channel_threads.
CREATE TABLE channel_thread_members (
    channel_id TEXT NOT NULL REFERENCES channel_threads (channel_id) ON DELETE CASCADE,
    member TEXT NOT NULL,
    PRIMARY KEY (channel_id, member)
);

-- +goose Down
DROP TABLE channel_thread_members;
