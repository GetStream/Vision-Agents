-- +goose Up

-- A campaign becomes an outbound dialer: a sequence of steps to try each contact through,
-- the hours it may ring, and for each contact where it stands in that sequence.
-- sequence and calling_hours are NULL on a draft until the API fills them.
ALTER TABLE campaigns
    ADD COLUMN sequence JSONB,
    ADD COLUMN calling_hours JSONB,
    ADD COLUMN default_country TEXT NOT NULL DEFAULT '',
    ADD COLUMN start_at TIMESTAMPTZ,
    ADD COLUMN launched_at TIMESTAMPTZ,
    ADD COLUMN stopped_at TIMESTAMPTZ,
    ADD COLUMN consent_at TIMESTAMPTZ,
    ADD COLUMN type TEXT NOT NULL DEFAULT 'standard';

ALTER TABLE campaign_contacts RENAME COLUMN state TO stage;
ALTER TABLE campaign_contacts RENAME COLUMN attempts TO attempt;
ALTER TABLE campaign_contacts RENAME COLUMN error TO reason;

ALTER TABLE campaign_contacts
    ADD COLUMN variables JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN time_zone TEXT NOT NULL DEFAULT '',
    ADD COLUMN step_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN next_attempt_at TIMESTAMPTZ,
    ADD COLUMN leased_by TEXT NOT NULL DEFAULT '',
    ADD COLUMN leased_until TIMESTAMPTZ,
    ADD COLUMN outcome TEXT NOT NULL DEFAULT '',
    ADD COLUMN converted BOOLEAN NOT NULL DEFAULT false;

-- A campaign that was running keeps ringing as before: one step, one attempt at once, every
-- hour of every day. A contact that was pending or mid-call is rung again.
UPDATE campaigns SET state = 'completed' WHERE state = 'finished';

UPDATE campaigns SET
    sequence = jsonb_build_object(
        'steps', jsonb_build_array(jsonb_build_object(
            'id', 'step_1',
            'config_id', config_id,
            'attempts', jsonb_build_array(jsonb_build_object('delay_minutes', 0)),
            'outcomes', '[]'::jsonb,
            'other', jsonb_build_object('next', jsonb_build_object('end', 'unsuccessful')),
            'no_response', jsonb_build_object('next', jsonb_build_object('end', 'unsuccessful')))),
        'double_dial', false),
    calling_hours = jsonb_build_object(
        'windows', jsonb_build_object(
            'monday', '[{"start":"00:00","end":"24:00"}]'::jsonb,
            'tuesday', '[{"start":"00:00","end":"24:00"}]'::jsonb,
            'wednesday', '[{"start":"00:00","end":"24:00"}]'::jsonb,
            'thursday', '[{"start":"00:00","end":"24:00"}]'::jsonb,
            'friday', '[{"start":"00:00","end":"24:00"}]'::jsonb,
            'saturday', '[{"start":"00:00","end":"24:00"}]'::jsonb,
            'sunday', '[{"start":"00:00","end":"24:00"}]'::jsonb),
        'time_zone', jsonb_build_object('mode', 'fixed', 'fallback', 'UTC', 'fixed', 'UTC'),
        'excluded_dates', '[]'::jsonb);

UPDATE campaign_contacts SET stage = CASE stage
    WHEN 'pending' THEN 'enrolled'
    WHEN 'calling' THEN 'enrolled'
    WHEN 'done' THEN 'completed'
    ELSE stage END;

UPDATE campaign_contacts SET step_id = 'step_1';
UPDATE campaign_contacts SET next_attempt_at = now() WHERE stage = 'enrolled';
ALTER TABLE campaign_contacts ALTER COLUMN stage SET DEFAULT 'enrolled';

DROP INDEX campaign_contacts_pending_idx;
CREATE UNIQUE INDEX campaign_contacts_number_idx ON campaign_contacts (campaign_id, to_number);
CREATE INDEX campaign_contacts_due_idx ON campaign_contacts (campaign_id, stage, next_attempt_at);
CREATE INDEX campaign_contacts_lease_idx ON campaign_contacts (leased_until) WHERE stage = 'in_progress';

-- One dial of one contact. A contact has several over a campaign, and the result and the
-- outcome of each is kept here rather than overwritten on the contact.
CREATE TABLE campaign_attempts (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns (id) ON DELETE CASCADE,
    contact_id TEXT NOT NULL REFERENCES campaign_contacts (id) ON DELETE CASCADE,
    step_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    dial INTEGER NOT NULL DEFAULT 1,
    call_id TEXT NOT NULL DEFAULT '',
    vendor_call_id TEXT NOT NULL DEFAULT '',
    -- result is answered, unanswered, invalid_number, refused, deferred, paused or
    -- abandoned, and empty while the call is happening.
    result TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    converted BOOLEAN NOT NULL DEFAULT false,
    -- classify_state is empty, pending or done: whether the outcome of an answered call
    -- still has to be read from its transcript.
    classify_state TEXT NOT NULL DEFAULT '',
    classify_after TIMESTAMPTZ,
    classify_tries INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ
);

CREATE INDEX campaign_attempts_contact_idx ON campaign_attempts (contact_id, started_at);
CREATE INDEX campaign_attempts_classify_idx ON campaign_attempts (classify_after)
    WHERE classify_state = 'pending';

-- +goose Down
DROP TABLE campaign_attempts;

DROP INDEX campaign_contacts_lease_idx;
DROP INDEX campaign_contacts_due_idx;
DROP INDEX campaign_contacts_number_idx;

UPDATE campaign_contacts SET stage = CASE stage
    WHEN 'enrolled' THEN 'pending'
    WHEN 'in_progress' THEN 'pending'
    WHEN 'awaiting_outcome' THEN 'done'
    WHEN 'completed' THEN 'done'
    WHEN 'rejected' THEN 'failed'
    WHEN 'stopped' THEN 'pending'
    ELSE stage END;
UPDATE campaigns SET state = CASE state
    WHEN 'completed' THEN 'finished'
    WHEN 'scheduled' THEN 'draft'
    WHEN 'stopped' THEN 'paused'
    ELSE state END;

ALTER TABLE campaign_contacts
    DROP COLUMN converted,
    DROP COLUMN outcome,
    DROP COLUMN leased_until,
    DROP COLUMN leased_by,
    DROP COLUMN next_attempt_at,
    DROP COLUMN step_id,
    DROP COLUMN time_zone,
    DROP COLUMN variables;
ALTER TABLE campaign_contacts ALTER COLUMN stage SET DEFAULT 'pending';
ALTER TABLE campaign_contacts RENAME COLUMN reason TO error;
ALTER TABLE campaign_contacts RENAME COLUMN attempt TO attempts;
ALTER TABLE campaign_contacts RENAME COLUMN stage TO state;
CREATE INDEX campaign_contacts_pending_idx ON campaign_contacts (campaign_id, seq)
    WHERE state = 'pending';

ALTER TABLE campaigns
    DROP COLUMN type,
    DROP COLUMN consent_at,
    DROP COLUMN stopped_at,
    DROP COLUMN launched_at,
    DROP COLUMN start_at,
    DROP COLUMN default_country,
    DROP COLUMN calling_hours,
    DROP COLUMN sequence;
