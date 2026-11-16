-- +goose Up

-- chat_channel_turns is the lease on the one running turn of a channel of a type an agent
-- config answers in (store.TakeChatChannelTurn), as channel_threads.turn_holder is a thread
-- channel's: two routers answering one channel at once would talk over each other. Such a
-- channel has no row of its own anywhere, so its turn is a row here, made by the turn that
-- takes it and deleted by the turn that lets go of it. A channel's id is unique only within
-- its Stream app, so the app is part of the key: 0 is the deployment's own app. A new table:
-- the previous release never reads it.
CREATE TABLE chat_channel_turns (
    stream_app_pk BIGINT NOT NULL,
    cid TEXT NOT NULL,
    turn_holder TEXT NOT NULL,
    turn_until TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (stream_app_pk, cid)
);

-- +goose Down
DROP TABLE chat_channel_turns;
