-- +goose Up
-- Each Stream hook delivery acted on, so one Stream sends again is answered once. Kept a day,
-- which is longer than Stream retries for.
CREATE TABLE hook_deliveries (
    key TEXT PRIMARY KEY,
    seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX hook_deliveries_seen_at ON hook_deliveries (seen_at);

-- +goose Down
DROP TABLE hook_deliveries;
