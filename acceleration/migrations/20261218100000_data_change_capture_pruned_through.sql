-- +goose Up
-- The highest number of a customer's changes pruning dropped: a cursor below it lost a change, so
-- it is refused (Store.Changes). 0 while none of theirs was pruned.
ALTER TABLE data_change_capture ADD COLUMN pruned_through BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE data_change_capture DROP COLUMN pruned_through;
