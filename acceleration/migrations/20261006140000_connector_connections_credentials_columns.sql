-- +goose Up

-- The sealed blob is core.StoredCredentials, so its columns say credentials, not material.
-- RENAME COLUMN keeps the data and the defaults as they are. record_data_change strips the
-- sealed column from what a data move copies (20261002193000_connector_connections.sql), and
-- it names the column, so it is replaced with the new name: otherwise the renamed column
-- would leave with the rest of the row.
ALTER TABLE connector_connections RENAME COLUMN material_sealed TO credentials_sealed;
ALTER TABLE connector_connections RENAME COLUMN material_kek_version TO credentials_kek_version;

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
        - 'oauth_state' - 'code_verifier' - 'material_sealed';

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
ALTER TABLE connector_connections RENAME COLUMN credentials_kek_version TO material_kek_version;
ALTER TABLE connector_connections RENAME COLUMN credentials_sealed TO material_sealed;
