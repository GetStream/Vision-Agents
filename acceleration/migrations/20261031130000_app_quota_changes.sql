-- +goose Up

-- app_quota_changes is every change Stream staff made to what one end user of an app may
-- spend in a day: who made it, why, when, and the numbers before and after. A NULL number
-- is one left unset, which is the router's default. Rows are only ever appended.
CREATE TABLE app_quota_changes (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    changed_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    previous_messages_per_user_per_day BIGINT,
    previous_tokens_per_user_per_day BIGINT,
    new_messages_per_user_per_day BIGINT,
    new_tokens_per_user_per_day BIGINT,
    changed_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX app_quota_changes_customer_idx ON app_quota_changes (customer_id, changed_at DESC, id DESC);

-- +goose Down
DROP TABLE app_quota_changes;
