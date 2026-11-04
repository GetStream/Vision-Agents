-- +goose NO TRANSACTION
-- +goose Up

-- A knowledge document can be a file uploaded whole, a PDF or a Word file, which is read in
-- the background rather than in the request that sent it. state is where that read is:
-- pending until the worker has written its passages, indexed once it has, failed with error
-- saying why when it could not be read. A document posted as text is indexed when it is
-- written, as every row before this column was. pages is how many pages the file had, 0 for
-- text; content_type and bytes are what was sent. object_key is the file in the knowledge
-- bucket while it waits to be read, and empty once it has been, since nothing reads it again.
-- Every column has a constant default, so adding them rewrites no row, and the CHECK is NOT
-- VALID since every row holds the default. The lock waits at most 5 s behind a running
-- query, past which the migration fails and the router's next start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE knowledge_documents
        ADD COLUMN IF NOT EXISTS state TEXT NOT NULL DEFAULT 'indexed',
        ADD COLUMN IF NOT EXISTS error TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS pages INTEGER NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS content_type TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS bytes BIGINT NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS object_key TEXT NOT NULL DEFAULT '';
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'knowledge_documents_state_check') THEN
        ALTER TABLE knowledge_documents ADD CONSTRAINT knowledge_documents_state_check
            CHECK (state IN ('pending', 'indexed', 'failed')) NOT VALID;
    END IF;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE knowledge_documents
        DROP CONSTRAINT IF EXISTS knowledge_documents_state_check,
        DROP COLUMN IF EXISTS object_key,
        DROP COLUMN IF EXISTS bytes,
        DROP COLUMN IF EXISTS content_type,
        DROP COLUMN IF EXISTS pages,
        DROP COLUMN IF EXISTS error,
        DROP COLUMN IF EXISTS state;
END $$;
-- +goose StatementEnd
