-- +goose Up

-- One row per organization and kind of money limit it was first refused at, or would have been
-- in shadow, so the operator hears of it once. The router that writes the row claims it and
-- posts it; posted_at stays empty until the post succeeds, and a row claimed long enough ago
-- and never posted is claimed again.
CREATE TABLE limit_notices (
    organization_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('no_credit', 'spend_hard_limit', 'spend_limit')),
    app_id TEXT NOT NULL DEFAULT '',
    tier TEXT NOT NULL DEFAULT '',
    enforced BOOLEAN NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    posted_at TIMESTAMPTZ,
    PRIMARY KEY (organization_id, kind)
);

CREATE INDEX limit_notices_unposted ON limit_notices (claimed_at) WHERE posted_at IS NULL;

-- +goose Down
DROP TABLE limit_notices;
