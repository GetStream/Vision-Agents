-- +goose Up

-- return_url is the app's page the consents a session asks for in the chat go back to, the
-- session request's return_url (AI-1384), so a chat carried on after it ended in another
-- process, and a fork, still go back there. NULL, as in every session made before it, is the
-- dashboard. Nullable and with no default, so adding it rewrites no row.
ALTER TABLE agent_sessions ADD COLUMN return_url TEXT;

-- +goose Down
ALTER TABLE agent_sessions DROP COLUMN return_url;
