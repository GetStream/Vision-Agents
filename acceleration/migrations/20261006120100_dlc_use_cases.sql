-- +goose Up

-- dlc_use_cases are what an app sends text and calls for, one 10DLC campaign each. Stream
-- reviews a use case before it goes to the vendor, and the vendor approves it before its
-- numbers may send. The 10DLC fields are columns; the RCS, WhatsApp, iMessage and voice
-- fields are kept in channels, stored for review and not yet sent anywhere.
CREATE TABLE dlc_use_cases (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    -- status is draft, submitted, changes_requested, rejected, vendor_pending, approved or
    -- vendor_rejected.
    status TEXT NOT NULL,
    use_case_type TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    message_flow TEXT NOT NULL DEFAULT '',
    message_samples TEXT[] NOT NULL DEFAULT '{}',
    help_message TEXT NOT NULL DEFAULT '',
    opt_out_message TEXT NOT NULL DEFAULT '',
    opt_in_message TEXT NOT NULL DEFAULT '',
    embedded_links BOOLEAN NOT NULL DEFAULT FALSE,
    embedded_phone BOOLEAN NOT NULL DEFAULT FALSE,
    age_gated BOOLEAN NOT NULL DEFAULT FALSE,
    direct_lending BOOLEAN NOT NULL DEFAULT FALSE,
    channels JSONB NOT NULL DEFAULT '{}',
    vendor TEXT NOT NULL DEFAULT '',
    vendor_campaign_id TEXT NOT NULL DEFAULT '',
    vendor_status TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    submitted_at TIMESTAMPTZ,
    approved_at TIMESTAMPTZ
);

-- One default per app: it is the campaign every number not assigned to another one sends as.
CREATE UNIQUE INDEX dlc_use_cases_default_idx ON dlc_use_cases (customer_id) WHERE is_default;
CREATE INDEX dlc_use_cases_customer_idx ON dlc_use_cases (customer_id, created_at DESC, id DESC);
-- The staff queue pages through every app's use cases in one status.
CREATE INDEX dlc_use_cases_status_idx ON dlc_use_cases (status, updated_at, id);

-- dlc_review_logs is every move a use case made, and who made it: the app submitting,
-- Stream staff reviewing, or the vendor answering. Nothing is ever updated or deleted here.
CREATE TABLE dlc_review_logs (
    id TEXT PRIMARY KEY,
    use_case_id TEXT NOT NULL REFERENCES dlc_use_cases (id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL,
    -- actor is app, staff or vendor, and actor_name who exactly, where known.
    actor TEXT NOT NULL,
    actor_name TEXT NOT NULL DEFAULT '',
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    vendor_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    submitted_at TIMESTAMPTZ,
    approved_at TIMESTAMPTZ
);

CREATE INDEX dlc_review_logs_use_case_idx ON dlc_review_logs (use_case_id, created_at, id);

-- A number sends as the use case it is assigned to, and as the app's default when it is
-- assigned to none.
ALTER TABLE phone_numbers ADD COLUMN dlc_use_case_id TEXT;

-- +goose Down
ALTER TABLE phone_numbers DROP COLUMN dlc_use_case_id;
DROP TABLE dlc_review_logs;
DROP TABLE dlc_use_cases;
