-- +goose Up

-- channel_thread is the thread channel (channel_threads.channel_id) of the SMS
-- or WhatsApp thread a task was made in, where it is delivered back to.
-- Empty for every task there is and every task made anywhere else.
ALTER TABLE scheduled_tasks ADD COLUMN channel_thread TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE scheduled_tasks DROP COLUMN channel_thread;
