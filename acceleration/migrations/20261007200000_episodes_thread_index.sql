-- +goose Up

-- The episodes of one thread channel of a customer, in the order they started, for reading
-- the cards (T56 and T42, AI-885): store.EpisodeCards ends a card's lines where the next
-- episode in its channel starts, which without this scans every newer episode of the
-- customer. A call channel holds every caller's calls (episodes_call_session), so this is
-- the one index that finds the next one. episodes_open_thread is on in_progress thread
-- episodes only, and episodes_contact is by person. A new index on an existing table: no
-- row is rewritten.
CREATE INDEX episodes_thread ON episodes (customer_id, thread_channel, started_at);

-- +goose Down
DROP INDEX episodes_thread;
