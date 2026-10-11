-- +goose Up

-- A language model a customer serves themselves, behind an OpenAI-compatible endpoint: a
-- fine-tune on a host of open weights, or vLLM on their own GPUs. A session names it as
-- custom/<name>, and routing reaches it the way it reaches a model in the catalogue.
--
-- The key is sealed under this deployment's key, bound to the customer, so a row copied to
-- another customer does not open.
CREATE TABLE custom_models (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    api_key_sealed BYTEA,
    kek_version INTEGER NOT NULL DEFAULT 0,
    context_window BIGINT NOT NULL DEFAULT 0,
    input_modalities TEXT[] NOT NULL DEFAULT '{}',
    per_million_input_tokens DOUBLE PRECISION NOT NULL DEFAULT 0,
    per_million_output_tokens DOUBLE PRECISION NOT NULL DEFAULT 0,
    trains_on_data TEXT NOT NULL DEFAULT '',
    retention TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX custom_models_name_idx ON custom_models (customer_id, name);
CREATE INDEX custom_models_customer_idx ON custom_models (customer_id, created_at DESC, id DESC);

-- +goose Down
DROP TABLE custom_models;
