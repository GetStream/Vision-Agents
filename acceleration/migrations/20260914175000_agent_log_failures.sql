-- +goose Up
ALTER TABLE requests ADD COLUMN error_message TEXT;
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION log_agent_request() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE cfg TEXT; sid TEXT;
BEGIN
 IF NEW.agent_id IS NULL OR NEW.agent_id = '' THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended(NEW.customer_id, 731));
 SELECT config_id,id INTO cfg,sid FROM calls WHERE customer_id=NEW.customer_id AND agent_id=NEW.agent_id AND (NEW.call_id IS NULL OR call_id=NEW.call_id) AND started_at <= NEW.started_at ORDER BY started_at DESC LIMIT 1;
 INSERT INTO agent_logs(customer_id,config_id,agent_id,session_id,source,severity,event_type,message,occurred_at,details)
 VALUES(NEW.customer_id,coalesce(cfg,''),NEW.agent_id,coalesce(sid,''),'system',CASE WHEN NEW.success THEN 'info' ELSE 'error' END,
 'provider_request',NEW.modality || ' request ' || CASE WHEN NEW.success THEN 'completed' ELSE 'failed' END,NEW.started_at,
 jsonb_build_object('provider',NEW.provider,'model',NEW.model,'request_id',NEW.id::text,'duration_ms',NEW.latency_ms,'input_tokens',NEW.input_tokens,'output_tokens',NEW.output_tokens,'cost_micros',NEW.cost_micros,'error_code',NEW.error_code,'error_message',NEW.error_message));
 RETURN NEW;
END $$;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION log_agent_request() RETURNS trigger LANGUAGE plpgsql AS $$
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

ALTER TABLE requests DROP COLUMN error_message;
