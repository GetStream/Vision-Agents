-- +goose Up

-- The management API (/v1/manage, T2 A) audits what it changes: Stream's staff, or a
-- self-hosted router's operator, call it with the manage key and name their actor.
--
--   - resource_type gains price_list: the deployment's price list, which is no customer's.
--     Its rows have an empty customer_id, so no app's audit query lists them.
--   - source gains manage: a change made through the management API. No client header can
--     claim it; only the manage key does (store.AuditSource).
--
-- Both CHECKs are dropped and added again with the new value beside the ones they had. That
-- narrows nothing: every row meets the old CHECKs, whose values the new ones keep. They are
-- added NOT VALID, as 20261101130000 adds its CHECK: the table is not read again to prove
-- what the old CHECKs already did, and the CHECKs still refuse an unknown value for every
-- row written from now on. goose runs it all in one transaction, so no moment allows
-- another value. A router built before this one writes none of the new values.

ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_resource_type_check,
    ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
        ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy',
         'price_list')) NOT VALID,
    DROP CONSTRAINT audit_log_source_check,
    ADD CONSTRAINT audit_log_source_check CHECK (source IN
        ('dashboard', 'cli', 'sdk', 'api', 'manage')) NOT VALID;

-- +goose Down
-- NOT VALID: the old CHECKs hold for every row written from now on and leave the price_list
-- and manage rows already stored in place, rather than delete the record of who changed what.
ALTER TABLE audit_log
    DROP CONSTRAINT audit_log_resource_type_check,
    ADD CONSTRAINT audit_log_resource_type_check CHECK (resource_type IN
        ('agent_config', 'skill', 'knowledge', 'knowledge_url', 'router_config', 'plugin', 'policy')) NOT VALID,
    DROP CONSTRAINT audit_log_source_check,
    ADD CONSTRAINT audit_log_source_check CHECK (source IN
        ('dashboard', 'cli', 'sdk', 'api')) NOT VALID;
