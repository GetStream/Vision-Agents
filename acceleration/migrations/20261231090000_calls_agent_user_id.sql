-- +goose NO TRANSACTION
-- +goose Up

-- agent_user_id is the Stream user a call's agent wrote its transcript as (session.Spec.UserID).
-- A member of the channel the transcript is in can write a line marked as the agent's too, and
-- this is what tells the agent's own lines apart. Null for a call that started before this
-- column, which is read as it was then. The column has no default, so adding it rewrites no
-- row; its lock waits at most 5 s behind a running query, past which the migration fails and
-- the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE calls ADD COLUMN IF NOT EXISTS agent_user_id TEXT;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE calls DROP COLUMN IF EXISTS agent_user_id;
END $$;
-- +goose StatementEnd
