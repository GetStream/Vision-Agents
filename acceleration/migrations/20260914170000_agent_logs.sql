-- +goose Up
CREATE TABLE agent_logs (
 id BIGSERIAL PRIMARY KEY,
 customer_id TEXT NOT NULL,
 config_id TEXT NOT NULL DEFAULT '', agent_id TEXT NOT NULL DEFAULT '',
 session_id TEXT NOT NULL DEFAULT '', user_id TEXT NOT NULL DEFAULT '',
 source TEXT NOT NULL, severity TEXT NOT NULL, event_type TEXT NOT NULL,
 message TEXT NOT NULL, occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 ingested_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 details JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX agent_logs_customer_order ON agent_logs(customer_id,id DESC);
CREATE INDEX agent_logs_config_order ON agent_logs(customer_id,config_id,id DESC);
CREATE INDEX agent_logs_error_order ON agent_logs(customer_id,severity,id DESC);
-- Writers lock per customer before allocating an ID: resume order follows commit order.
-- Provider recording already runs off the conversation path.
-- +goose StatementBegin
CREATE FUNCTION log_agent_request() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE cfg TEXT; sid TEXT;
BEGIN
 IF NEW.agent_id IS NULL OR NEW.agent_id = '' THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended(NEW.customer_id, 731));
 SELECT config_id,id INTO cfg,sid FROM calls WHERE customer_id=NEW.customer_id AND agent_id=NEW.agent_id AND (NEW.call_id IS NULL OR call_id=NEW.call_id) AND started_at <= NEW.started_at ORDER BY started_at DESC LIMIT 1;
 INSERT INTO agent_logs(customer_id,config_id,agent_id,session_id,source,severity,event_type,message,occurred_at,details)
 VALUES(NEW.customer_id,coalesce(cfg,''),NEW.agent_id,coalesce(sid,''),'system',CASE WHEN NEW.success THEN 'info' ELSE 'error' END,
 'provider_request',NEW.modality || ' request ' || CASE WHEN NEW.success THEN 'completed' ELSE 'failed' END,NEW.started_at,
 jsonb_build_object('provider',NEW.provider,'model',NEW.model,'request_id',NEW.id::text,'duration_ms',NEW.latency_ms,'input_tokens',NEW.input_tokens,'output_tokens',NEW.output_tokens,'cost_micros',NEW.cost_micros));
 RETURN NEW;
END $$;
-- +goose StatementEnd
CREATE TRIGGER agent_request_log AFTER INSERT ON requests FOR EACH ROW EXECUTE FUNCTION log_agent_request();
-- +goose Down
DROP TRIGGER agent_request_log ON requests;
DROP FUNCTION log_agent_request();
DROP TABLE agent_logs;
