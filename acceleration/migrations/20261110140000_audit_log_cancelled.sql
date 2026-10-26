-- +goose Up

-- A scheduled price change can be cancelled through the management API (T1 C), once, before
-- it takes effect, and the cancel is audited: audit_log's action CHECK gains cancelled.
--
-- The CHECK is dropped and added again with the new value beside the ones it had, as
-- 20261110120000 did for resource_type and source. That narrows nothing: every row meets the
-- old CHECK, whose values the new one keeps. It is added NOT VALID, so the table is not read
-- again to prove what the old CHECK already did, and the CHECK still refuses an unknown value
-- for every row written from now on. goose runs it all in one transaction, so no moment allows
-- another value. A router built before this one writes none of the new value.

ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_action_check,
    ADD CONSTRAINT audit_log_action_check CHECK (action IN
        ('created', 'updated', 'deleted', 'synced', 'cancelled')) NOT VALID;

-- +goose Down
-- NOT VALID: the old CHECK holds for every row written from now on and leaves the cancelled
-- rows already stored in place, rather than delete the record of who cancelled what.
ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_action_check,
    ADD CONSTRAINT audit_log_action_check CHECK (action IN
        ('created', 'updated', 'deleted', 'synced')) NOT VALID;
