-- +goose Up

-- provider_headers_until is when the provider's own signature headers a forward carries stop
-- verifying at a receiver that checks them (AI-924): the provider's signed timestamp plus the
-- manifest's channel.verifier.max_age. For Slack that is X-Slack-Request-Timestamp plus 5
-- minutes, the same 5 minutes Slack Bolt refuses a request after
-- (requestTimestampMaxDeltaMin = 5, bolt-js src/receivers/verify-request.ts). An attempt sent
-- after it carries Content-Type alone of the provider's headers, so the receiver verifies it
-- with webhook-signature (eventforward.Forwarder.send). NULL when the verifier signs no
-- timestamp, and on every row queued before this column.
ALTER TABLE connector_event_deliveries ADD COLUMN provider_headers_until TIMESTAMPTZ;

-- +goose Down
ALTER TABLE connector_event_deliveries DROP COLUMN provider_headers_until;
