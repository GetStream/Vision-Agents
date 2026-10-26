-- +goose Up
-- One unit of work as a producer shipped it, once per usage id.
CREATE TABLE usage_events (
    usage_id UUID PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    organization_id TEXT NOT NULL CHECK (organization_id <> ''),
    organization_tier TEXT NOT NULL CHECK (organization_tier <> ''),
    app_id TEXT NOT NULL CHECK (app_id <> ''),
    api_key TEXT NOT NULL CHECK (api_key <> ''),
    user_id TEXT NOT NULL DEFAULT '',
    usage_categories TEXT[] NOT NULL DEFAULT '{}',
    agent_id TEXT NOT NULL DEFAULT '',
    call_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    meter TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    purpose TEXT NOT NULL DEFAULT '',
    quantities JSONB NOT NULL DEFAULT '{}'::jsonb,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_micros BIGINT NOT NULL CHECK (cost_micros >= 0),
    base_cost_micros BIGINT NOT NULL DEFAULT 0 CHECK (base_cost_micros >= 0),
    price_version TEXT NOT NULL DEFAULT '',
    billable BOOLEAN NOT NULL,
    success BOOLEAN NOT NULL,
    error_code TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '{}'::jsonb,
    source TEXT NOT NULL
);
CREATE INDEX usage_events_organization_idx ON usage_events (organization_id, occurred_at DESC);
-- Usage by key, user or category; a user id is only unique within its app.
CREATE INDEX usage_events_key_idx ON usage_events (organization_id, api_key, occurred_at DESC);
CREATE INDEX usage_events_user_idx ON usage_events (organization_id, app_id, user_id, occurred_at DESC)
    WHERE user_id <> '';
CREATE INDEX usage_events_categories_idx ON usage_events USING gin (usage_categories)
    WHERE usage_categories <> '{}';

-- What one subject spent in one window of one interval: the running total every limit is
-- checked against, so a check reads one row and never sums events. Every event counts for
-- every subject it names in every interval, limit or not.
CREATE TABLE usage_windows (
    organization_id TEXT NOT NULL,
    subject TEXT NOT NULL CHECK (subject IN
        ('organization', 'app', 'api_key', 'user', 'category', 'app_category')),
    app_id TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    interval TEXT NOT NULL CHECK (interval IN ('daily', 'weekly', 'biweekly', 'monthly')),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    -- Never negative: the mirror's script compares totals as unsigned decimals.
    spent_micros BIGINT NOT NULL DEFAULT 0 CHECK (spent_micros >= 0),
    events BIGINT NOT NULL DEFAULT 0 CHECK (events >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (organization_id, subject, app_id, api_key, user_id, category, interval, window_start)
);
CREATE INDEX usage_windows_end_idx ON usage_windows (window_end);

-- +goose Down
DROP TABLE usage_windows;
DROP TABLE usage_events;
