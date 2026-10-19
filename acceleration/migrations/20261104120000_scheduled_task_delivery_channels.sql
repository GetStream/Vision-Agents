-- +goose Up

-- quiet_hours is the span of the user's local day an occurrence is held
-- through rather than delivered into, as {start, end} in "22:00" form, read
-- in the task's own zone. Null, as in every task made before it, is no hold.
ALTER TABLE scheduled_tasks ADD COLUMN quiet_hours JSONB;

-- postponed_until is when the hold on a run ends, recorded when the worker
-- holds the occurrence (quiet hours), so when the task is claimed again the
-- occurrence's age is counted from the hold's end and not from when it came
-- due: a reminder held overnight is delivered in the morning, not missed.
ALTER TABLE scheduled_task_runs ADD COLUMN postponed_until TIMESTAMPTZ;

-- from_address is who a delivered message came from. A line's deliveries
-- were only ever its dedup record; with it, the last time a person wrote is
-- readable, which is what WhatsApp's 24-hour window is counted from.
ALTER TABLE channel_deliveries ADD COLUMN from_address TEXT NOT NULL DEFAULT '';

-- thread is where a message is sent to reach this person again: the chat,
-- for iMessage, whose replies are posted to a conversation rather than to a
-- person. Empty for the channels that reply to the address itself.
ALTER TABLE channel_identities ADD COLUMN thread TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE channel_identities DROP COLUMN thread;
ALTER TABLE channel_deliveries DROP COLUMN from_address;
ALTER TABLE scheduled_task_runs DROP COLUMN postponed_until;
ALTER TABLE scheduled_tasks DROP COLUMN quiet_hours;
