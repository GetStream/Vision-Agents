-- +goose Up

-- One row per organization and UTC day on which its spend jumped past its own baseline, so the
-- operator hears of each such day once. internal/limitnotice writes the row before it posts:
-- posted_at stays empty until the post succeeds, and a row claimed long enough ago and never
-- posted is claimed again, as limit_notices are. spent_micros and baseline_micros are what the
-- check saw: that day's spend so far and the median day before it.
CREATE TABLE cost_anomaly_notices (
    organization_id TEXT NOT NULL,
    day DATE NOT NULL,
    spent_micros BIGINT NOT NULL,
    baseline_micros BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    posted_at TIMESTAMPTZ,
    PRIMARY KEY (organization_id, day)
);

CREATE INDEX cost_anomaly_notices_unposted ON cost_anomaly_notices (claimed_at) WHERE posted_at IS NULL;

-- +goose Down
DROP TABLE cost_anomaly_notices;
