-- +goose Up
-- What one tier string decides. The rows are data: this migration writes none, and the
-- service adds the tiers of its tier file it lacks at start. The 'default' row answers for
-- any tier string with no row of its own.
CREATE TABLE tiers (
    name TEXT PRIMARY KEY CHECK (name <> ''),
    -- closed: refuse billable work while the balance cannot be read; open: admit it.
    fail_mode TEXT NOT NULL DEFAULT 'closed' CHECK (fail_mode IN ('open', 'closed')),
    limits JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_by TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One organization's credit, spend, status and its own settings. Every organization has one,
-- the row written the first time anything changes it: until then it is active with nothing
-- granted or spent. Balance is granted_micros - spent_micros, both running totals.
CREATE TABLE accounts (
    organization_id TEXT PRIMARY KEY CHECK (organization_id <> ''),
    -- suspended: everything is refused, free work included.
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended')),
    status_reason TEXT NOT NULL DEFAULT '',
    -- The organization's own value for a tier's settings, each laid over its tier's.
    limit_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,
    granted_micros BIGINT NOT NULL DEFAULT 0,
    spent_micros BIGINT NOT NULL DEFAULT 0 CHECK (spent_micros >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every change to an organization's credit, written once under its idempotency key.
CREATE TABLE credit_entries (
    id BIGSERIAL PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES accounts (organization_id),
    kind TEXT NOT NULL CHECK (kind IN
        ('top_up', 'grant', 'refund', 'dispute', 'dispute_reversal', 'adjustment')),
    amount_micros BIGINT NOT NULL, -- signed; the kind fixes the sign
    reference TEXT NOT NULL DEFAULT '', -- the writer's, never read here
    idempotency_key TEXT NOT NULL UNIQUE,
    note TEXT NOT NULL DEFAULT '',
    actor TEXT NOT NULL DEFAULT '',
    actor_name TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT credit_entries_sign_check CHECK (
        (kind IN ('top_up', 'grant', 'dispute_reversal') AND amount_micros > 0)
        OR (kind IN ('refund', 'dispute') AND amount_micros < 0)
        OR (kind = 'adjustment' AND amount_micros <> 0))
);
CREATE INDEX credit_entries_organization_idx
    ON credit_entries (organization_id, created_at DESC, id DESC);

-- Every management write: who, on which account (none for a tier), and what it was before
-- and after.
CREATE TABLE manage_audit (
    id BIGSERIAL PRIMARY KEY,
    actor TEXT NOT NULL,
    actor_name TEXT NOT NULL DEFAULT '',
    organization_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '', -- a suspension's or a restore's
    before JSONB NOT NULL DEFAULT '{}'::jsonb,
    after JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- An organization's history, newest first.
CREATE INDEX manage_audit_organization_idx
    ON manage_audit (organization_id, created_at DESC, id DESC);

-- +goose Down
DROP TABLE manage_audit;
DROP TABLE credit_entries;
DROP TABLE accounts;
DROP TABLE tiers;
