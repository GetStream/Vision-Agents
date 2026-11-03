-- +goose NO TRANSACTION
-- +goose Up

-- conversation_id is the Stream Chat conversation a call was bound to when it started, as a
-- session's row names it (agent:<channel id>). A session's row can be deleted while its call
-- row and its conversation stay, and this is what still names the channel the call's
-- transcript is in. Null for a call that started before this column, which is read as it was
-- then, and for one that kept no conversation. The column has no default, so adding it
-- rewrites no row; its lock waits at most 5 s behind a running query, past which the
-- migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE calls ADD COLUMN IF NOT EXISTS conversation_id TEXT;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE calls DROP COLUMN IF EXISTS conversation_id;
END $$;
-- +goose StatementEnd
