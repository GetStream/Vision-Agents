-- +goose Up
-- The routing rule an attached number was given, beside its trunk, so releasing or
-- re-attaching the number removes both from the Stream app they were made in.
ALTER TABLE phone_numbers ADD COLUMN stream_route_id TEXT;

-- +goose Down
ALTER TABLE phone_numbers DROP COLUMN stream_route_id;
