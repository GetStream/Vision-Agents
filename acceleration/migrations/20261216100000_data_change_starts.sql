-- +goose Up
-- A repeatable read or serializable write whose snapshot is older than a capture's start is
-- refused rather than lost. This is 20261214130000_data_changes_in_flight.sql's function with
-- the lock on data_change_starts added.
CREATE TABLE data_change_starts (
    one BOOLEAN PRIMARY KEY DEFAULT true CHECK (one),
    started BIGINT NOT NULL DEFAULT 0
);
INSERT INTO data_change_starts DEFAULT VALUES;

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION record_data_change() RETURNS TRIGGER AS $$
DECLARE
    subject JSONB;
    owner TEXT;
    identity JSONB := '{}'::JSONB;
    column_name TEXT;
    handed_out BIGINT;
BEGIN
    IF TG_OP = 'DELETE' THEN
        subject := to_jsonb(OLD);
    ELSE
        subject := to_jsonb(NEW);
    END IF;

    -- TG_ARGV[0] is the column holding the customer id, or 'parent' when the row belongs
    -- to a customer through another table: TG_ARGV[1] is then the column referencing it
    -- and TG_ARGV[2] the table it references.
    IF TG_ARGV[0] = 'parent' THEN
        EXECUTE format('SELECT customer_id FROM %I WHERE id = $1', TG_ARGV[2])
            INTO owner USING subject ->> TG_ARGV[1];
    ELSE
        owner := subject ->> TG_ARGV[0];
    END IF;

    IF owner IS NULL OR owner = '' THEN
        RETURN NULL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM data_change_capture
        WHERE customer_id = owner AND expires_at > now()
    ) THEN
        -- At these two levels every query reads the snapshot of the transaction's first, so a
        -- capture started since is not seen here, and the transaction may hold no id yet for
        -- Store.WaitForOpenWrites to wait on. Every start updates the one row of
        -- data_change_starts. Locking it fails the transaction with a serialization failure if
        -- a start committed after its snapshot, and makes a start that comes later wait until it
        -- ends (https://www.postgresql.org/docs/current/transaction-iso.html#XACT-REPEATABLE-READ).
        IF current_setting('transaction_isolation') IN ('repeatable read', 'serializable') THEN
            PERFORM 1 FROM data_change_starts FOR SHARE;
        END IF;
        RETURN NULL;
    END IF;

    FOREACH column_name IN ARRAY TG_ARGV[3:] LOOP
        identity := identity || jsonb_build_object(column_name, subject -> column_name);
    END LOOP;

    -- Credentials are left behind. A secret this deployment sealed is worth nothing to
    -- the one reading it, and an access token a customer's user granted to this
    -- deployment is not something to hand out because somebody asked for their data.
    subject := subject - 'secret_sealed' - 'access_token' - 'refresh_token'
        - 'oauth_state' - 'code_verifier' - 'credentials_sealed' - 'password_sealed'
        - 'bridge_password_sealed' - 'refresh_secret_hash';

    -- A change is numbered when it is written and seen when its transaction commits, so no
    -- committed row tells a reader which lower number a transaction still running holds.
    -- Before its first change, a transaction takes a shared advisory lock keyed by the
    -- highest number handed out so far, below every number it will be given, and holds it
    -- until it ends. pg_locks shows a lock as soon as it is taken, and Postgres releases it
    -- only once the commit is visible (CommitTransaction in src/backend/access/transam/xact.c:
    -- "this must be done _before_ releasing locks we hold"). Store.changeWatermark reads
    -- these locks. The key
    -- is the number's two halves in the two-integer key space; the other two-integer locks
    -- here start at 839 << 32, a number no feed reaches. The setting, local to the
    -- transaction, takes it once rather than once per row.
    IF coalesce(current_setting('data_changes.bounded', true), '') = '' THEN
        SELECT CASE WHEN is_called THEN last_value ELSE last_value - 1 END
            INTO handed_out FROM data_changes_seq_seq;
        PERFORM pg_advisory_xact_lock_shared((handed_out >> 32)::int, ((handed_out << 32) >> 32)::int);
        PERFORM set_config('data_changes.bounded', 'on', true);
    END IF;

    INSERT INTO data_changes (customer_id, table_name, op, key, payload)
    VALUES (
        owner,
        TG_TABLE_NAME,
        lower(TG_OP),
        identity,
        CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE subject END
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd

-- +goose Down
-- 20261214130000_data_changes_in_flight.sql's function, without the lock.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION record_data_change() RETURNS TRIGGER AS $$
DECLARE
    subject JSONB;
    owner TEXT;
    identity JSONB := '{}'::JSONB;
    column_name TEXT;
    handed_out BIGINT;
BEGIN
    IF TG_OP = 'DELETE' THEN
        subject := to_jsonb(OLD);
    ELSE
        subject := to_jsonb(NEW);
    END IF;

    -- TG_ARGV[0] is the column holding the customer id, or 'parent' when the row belongs
    -- to a customer through another table: TG_ARGV[1] is then the column referencing it
    -- and TG_ARGV[2] the table it references.
    IF TG_ARGV[0] = 'parent' THEN
        EXECUTE format('SELECT customer_id FROM %I WHERE id = $1', TG_ARGV[2])
            INTO owner USING subject ->> TG_ARGV[1];
    ELSE
        owner := subject ->> TG_ARGV[0];
    END IF;

    IF owner IS NULL OR owner = '' THEN
        RETURN NULL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM data_change_capture
        WHERE customer_id = owner AND expires_at > now()
    ) THEN
        RETURN NULL;
    END IF;

    FOREACH column_name IN ARRAY TG_ARGV[3:] LOOP
        identity := identity || jsonb_build_object(column_name, subject -> column_name);
    END LOOP;

    -- Credentials are left behind. A secret this deployment sealed is worth nothing to
    -- the one reading it, and an access token a customer's user granted to this
    -- deployment is not something to hand out because somebody asked for their data.
    subject := subject - 'secret_sealed' - 'access_token' - 'refresh_token'
        - 'oauth_state' - 'code_verifier' - 'credentials_sealed' - 'password_sealed'
        - 'bridge_password_sealed' - 'refresh_secret_hash';

    -- A change is numbered when it is written and seen when its transaction commits, so no
    -- committed row tells a reader which lower number a transaction still running holds.
    -- Before its first change, a transaction takes a shared advisory lock keyed by the
    -- highest number handed out so far, below every number it will be given, and holds it
    -- until it ends. pg_locks shows a lock as soon as it is taken, and Postgres releases it
    -- only once the commit is visible (CommitTransaction in src/backend/access/transam/xact.c:
    -- "this must be done _before_ releasing locks we hold"). Store.changeWatermark reads
    -- these locks. The key
    -- is the number's two halves in the two-integer key space; the other two-integer locks
    -- here start at 839 << 32, a number no feed reaches. The setting, local to the
    -- transaction, takes it once rather than once per row.
    IF coalesce(current_setting('data_changes.bounded', true), '') = '' THEN
        SELECT CASE WHEN is_called THEN last_value ELSE last_value - 1 END
            INTO handed_out FROM data_changes_seq_seq;
        PERFORM pg_advisory_xact_lock_shared((handed_out >> 32)::int, ((handed_out << 32) >> 32)::int);
        PERFORM set_config('data_changes.bounded', 'on', true);
    END IF;

    INSERT INTO data_changes (customer_id, table_name, op, key, payload)
    VALUES (
        owner,
        TG_TABLE_NAME,
        lower(TG_OP),
        identity,
        CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE subject END
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd

DROP TABLE data_change_starts;
