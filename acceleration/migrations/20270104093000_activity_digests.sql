-- +goose Up

-- One row per UTC day whose activity digest was posted to the operator's channel, so the
-- deployment posts each day's once whatever its routers and restarts. internal/limitnotice
-- writes the row before it posts and posts only if its write made the row: posted_at stays
-- empty until the post succeeds, and a row claimed long enough ago and never posted is
-- claimed again, as cost_anomaly_notices are.
CREATE TABLE IF NOT EXISTS activity_digests (
    day DATE PRIMARY KEY,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    posted_at TIMESTAMPTZ
);

-- +goose Down
DROP TABLE activity_digests;
