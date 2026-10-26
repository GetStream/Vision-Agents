-- +goose Up

-- One row per warning sent that an organization's credit is running out (T6 C): the first
-- router to write it sends the event, and the rest see the row and send nothing. granted_micros
-- is part of the key, so a top-up or a grant re-arms the warning for the next time credit runs
-- low.
CREATE TABLE balance_notices (
    organization_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('balance_low', 'balance_exhausted')),
    granted_micros BIGINT NOT NULL,
    balance_micros BIGINT NOT NULL,
    threshold_micros BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (organization_id, kind, granted_micros)
);

-- +goose Down
DROP TABLE balance_notices;
