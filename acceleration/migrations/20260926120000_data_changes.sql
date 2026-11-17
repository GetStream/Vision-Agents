-- +goose Up

-- What a customer's rows have been doing, so a deployment they are moving to can follow
-- along after it has taken a copy. It is how a switchover happens without a window where
-- writes are lost: the new deployment imports a snapshot, replays from here until it has
-- caught up, and only then do the SDKs change URL.
--
-- payload holds the whole row as it is now, rather than the columns that changed, because
-- the other side applies it by upserting: one shape to write, and replaying the same
-- change twice lands in the same place.
CREATE TABLE data_changes (
    seq BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    op TEXT NOT NULL,
    -- key is what identifies the row, which is what a delete has instead of a payload.
    key JSONB NOT NULL,
    payload JSONB,
    -- tx is which transaction wrote this, and it is what makes tailing safe. Sequence
    -- numbers are handed out when a row is inserted and become visible when it commits,
    -- so a reader taking the highest number it can see would step over a lower one that
    -- had not committed yet. A change is only handed out once its transaction is older
    -- than every transaction still running.
    tx XID8 NOT NULL DEFAULT pg_current_xact_id(),
    at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX data_changes_customer_idx ON data_changes (customer_id, seq);
CREATE INDEX data_changes_at_idx ON data_changes (at);

-- Who is moving, and until when. Nothing is recorded for a customer without a row here,
-- which is what keeps this off the hot path: a deployment nobody is leaving pays one
-- index probe per write and stores nothing.
--
-- The row expires so that a move somebody abandoned halfway stops writing changes on its
-- own. Exporting sets it, and reading changes pushes it out again.
CREATE TABLE data_change_capture (
    customer_id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION record_data_change() RETURNS TRIGGER AS $$
DECLARE
    subject JSONB;
    owner TEXT;
    identity JSONB := '{}'::JSONB;
    column_name TEXT;
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
        - 'oauth_state' - 'code_verifier';

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
DROP FUNCTION IF EXISTS record_data_change();
DROP TABLE data_change_capture;
DROP TABLE data_changes;
