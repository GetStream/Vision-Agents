-- +goose Up

-- chat_person is the provider's id of the one person a chat thread is with, as its messages
-- name their author (core.InboundMessage.AuthorID): the sender of a Linq chat whose latest
-- message says it is no group (data.chat.is_group false). The thread key of such a thread is
-- the chat id, which names nobody, so this is what makes it one person's thread, whose session
-- may schedule tasks for them (api.threadPerson). NULL for a group chat, for a chat whose
-- message does not say, for every row written before this column, and for every other
-- connector: an SMS or WhatsApp thread key is the person's number already, and a Slack thread
-- is no one person's. NULL offers no scheduling tools.
ALTER TABLE channel_threads ADD COLUMN chat_person TEXT;

-- +goose Down
ALTER TABLE channel_threads DROP COLUMN chat_person;
