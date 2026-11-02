-- +goose Up

-- keyword_answered is whether the worker that admitted a queued message answered its carrier
-- keyword (STOP, START, HELP), set before the answer is sent, so a retry of the row answers it
-- no second time (channelbridge keywords.go).
ALTER TABLE channel_thread_inbound ADD COLUMN keyword_answered BOOLEAN NOT NULL DEFAULT false;

-- +goose Down
ALTER TABLE channel_thread_inbound DROP COLUMN keyword_answered;
