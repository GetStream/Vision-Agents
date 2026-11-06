-- +goose Up
-- A change number taken when a customer's capture was turned on again after it expired, above
-- every one handed out before: nothing written while it was off was recorded, so a cursor below
-- it is refused (Store.Changes). 0 for a capture that never expired.
ALTER TABLE data_change_capture ADD COLUMN recorded_since BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE data_change_capture DROP COLUMN recorded_since;
