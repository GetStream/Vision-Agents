-- +goose Up
-- Which Stream app a session, call, number or call leg was made in. Work is finished in the
-- app it was started in, wherever its customer acts now. NULL is the deployment's own app,
-- which is where everything written before apps had identities was made.
ALTER TABLE agent_sessions ADD COLUMN stream_app_pk BIGINT;
ALTER TABLE calls ADD COLUMN stream_app_pk BIGINT;
ALTER TABLE phone_numbers ADD COLUMN stream_app_pk BIGINT;
ALTER TABLE call_resources ADD COLUMN stream_app_pk BIGINT;

-- +goose Down
ALTER TABLE call_resources DROP COLUMN stream_app_pk;
ALTER TABLE phone_numbers DROP COLUMN stream_app_pk;
ALTER TABLE calls DROP COLUMN stream_app_pk;
ALTER TABLE agent_sessions DROP COLUMN stream_app_pk;
