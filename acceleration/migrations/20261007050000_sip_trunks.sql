-- +goose Up

-- A customer's own SIP trunk, which calls from the numbers on it are dialled through. The
-- password is sealed by the caller under this deployment's key (password_kek_version names
-- which one), with the customer and the trunk as associated data. An empty password at
-- version 0 is a trunk whose password has to be entered again, which is how one arrives
-- from a data move: the key that sealed it belongs to the other deployment.
CREATE TABLE sip_trunks (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    transport TEXT NOT NULL CHECK (transport IN ('udp', 'tcp', 'tls')),
    username TEXT NOT NULL,
    password_sealed BYTEA NOT NULL DEFAULT ''::bytea,
    password_kek_version INTEGER NOT NULL DEFAULT 0,
    late_offer BOOLEAN NOT NULL DEFAULT false,
    codecs TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (customer_id, id)
);
CREATE INDEX sip_trunks_customer_idx ON sip_trunks (customer_id, created_at DESC);

-- A number on a customer's trunk points at it. Releasing the number clears this, so the
-- foreign key is what refuses to delete a trunk that still has numbers on it, including
-- one being added at the same moment. The reference names the customer as well, so a row
-- written by a data move cannot point at somebody else's trunk.
ALTER TABLE phone_numbers ADD COLUMN sip_trunk_id TEXT,
    ADD FOREIGN KEY (customer_id, sip_trunk_id) REFERENCES sip_trunks (customer_id, id);

-- A vendor cannot sell one number twice, but two customers can both say a number is on
-- their own trunk: only the carrier that really has it will connect the call. Keeping the
-- global constraint for those would tell the second customer that somebody holds the
-- number, and let the first keep it from them.
DROP INDEX phone_numbers_held_idx;
CREATE UNIQUE INDEX phone_numbers_held_idx ON phone_numbers (vendor, e164)
    WHERE released_at IS NULL AND vendor <> 'sip_trunk';
CREATE UNIQUE INDEX phone_numbers_trunk_held_idx ON phone_numbers (customer_id, e164)
    WHERE released_at IS NULL AND vendor = 'sip_trunk';

-- A data move leaves password_sealed behind, as it does every other sealed secret
-- (store.secretColumns). This is 20261006140000_connector_connections_credentials_columns.sql's
-- function with password_sealed added to what it strips.
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
        - 'oauth_state' - 'code_verifier' - 'credentials_sealed' - 'password_sealed';

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
        - 'oauth_state' - 'code_verifier' - 'credentials_sealed';

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
-- Two customers may hold the same number on their own trunks, which the old unique index
-- on (vendor, e164) cannot be rebuilt over.
DELETE FROM phone_numbers WHERE vendor = 'sip_trunk';
DROP INDEX phone_numbers_trunk_held_idx;
DROP INDEX phone_numbers_held_idx;
CREATE UNIQUE INDEX phone_numbers_held_idx ON phone_numbers (vendor, e164)
    WHERE released_at IS NULL;
ALTER TABLE phone_numbers DROP COLUMN sip_trunk_id;
DROP TABLE sip_trunks;
