-- +goose Up

ALTER TABLE connector_connections
    ADD COLUMN auth_type TEXT NOT NULL DEFAULT 'oauth2',
    ADD COLUMN auth_header TEXT NOT NULL DEFAULT '';

ALTER TABLE connector_connections
    ADD CONSTRAINT connector_connections_auth_header_check
    CHECK ((auth_type = 'api_key' AND auth_header <> '') OR (auth_type <> 'api_key' AND auth_header = ''));

CREATE TABLE connector_definitions (
    customer_id TEXT NOT NULL,
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'Custom',
    description TEXT NOT NULL DEFAULT '',
    endpoint TEXT NOT NULL,
    auth_type TEXT NOT NULL CHECK (auth_type IN ('oauth2', 'none', 'bearer', 'api_key')),
    auth_header TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, id),
    CHECK ((auth_type = 'api_key' AND auth_header <> '') OR (auth_type <> 'api_key' AND auth_header = ''))
);

-- +goose Down
DROP TABLE connector_definitions;
ALTER TABLE connector_connections DROP CONSTRAINT connector_connections_auth_header_check;
ALTER TABLE connector_connections DROP COLUMN auth_header;
ALTER TABLE connector_connections DROP COLUMN auth_type;
