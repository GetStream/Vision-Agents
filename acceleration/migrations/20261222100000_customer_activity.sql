-- +goose Up

-- One row per app, area of the API, client and UTC day: how many requests the app made there
-- that day, how many were refused and with what the last time, and when the first and last
-- arrived. internal/activity sums them in memory and writes each one at most once a minute.
-- The organization is beside the app rather than in the key, so a day of an organization is
-- read by the index.
CREATE TABLE customer_activity (
    organization_id TEXT NOT NULL DEFAULT '',
    app_id TEXT NOT NULL,
    area TEXT NOT NULL,
    client TEXT NOT NULL,
    day DATE NOT NULL,
    requests BIGINT NOT NULL DEFAULT 0,
    refused BIGINT NOT NULL DEFAULT 0,
    last_refusal_code TEXT,
    last_refusal_at TIMESTAMPTZ,
    first_at TIMESTAMPTZ NOT NULL,
    last_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (app_id, area, client, day)
);

CREATE INDEX customer_activity_organization_day ON customer_activity (organization_id, day);

-- +goose Down
DROP TABLE customer_activity;
