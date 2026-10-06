-- +goose Up
-- Which customers app mode still writes into the deployment's own app because they have
-- registered none, so the fallback can be turned off knowing who it would leave nowhere.
-- Written at most once a minute per customer.
CREATE TABLE stream_fallback_uses (
    customer_id TEXT PRIMARY KEY,
    first_at TIMESTAMPTZ NOT NULL,
    last_at TIMESTAMPTZ NOT NULL,
    uses BIGINT NOT NULL
);

-- +goose Down
DROP TABLE stream_fallback_uses;
