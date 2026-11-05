-- +goose Up
-- Every transaction that records a change holds a lock naming a number below its own, so a
-- reader of the changes never hands out a cursor past a change that can still commit. This is
-- 20261207100100_users_refresh_secret_data_changes.sql's function with that lock added.
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

-- +goose Down
-- Left as it is: the code before this reads no lock, and one shared lock per transaction
-- that records a change is all it costs.
