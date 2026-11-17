-- +goose Up

-- telephony_suspensions is an app Stream staff stopped from using the phone, for abuse or
-- fraud: while its row is here it buys no numbers, places no calls or transfers, answers no
-- incoming call and runs no campaign. One row per app; lifting the suspension deletes it.
CREATE TABLE telephony_suspensions (
    customer_id TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    suspended_by TEXT NOT NULL,
    suspended_at TIMESTAMPTZ NOT NULL
);

-- +goose Down
DROP TABLE telephony_suspensions;
