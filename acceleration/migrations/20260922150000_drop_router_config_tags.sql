-- +goose Up

-- A router config says where to route, not who to bill. Cost labels belong to the request
-- that is being billed, which carries its own, so the ones held here were a default
-- nobody had a reason to set.
ALTER TABLE router_configs DROP COLUMN tags;

-- +goose Down
ALTER TABLE router_configs ADD COLUMN tags JSONB NOT NULL DEFAULT '{}';
