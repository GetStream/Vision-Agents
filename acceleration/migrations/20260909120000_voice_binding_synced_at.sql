-- +goose Up

-- synced_at is when this provider last came back with a voice we can speak in. It is not
-- updated_at: that moves again when a binding goes back to pending, so a row being touched
-- says nothing about whether the provider currently has the voice.
--
-- Null means never, which is what a pending or failed binding is.
ALTER TABLE voice_bindings ADD COLUMN synced_at TIMESTAMPTZ;

-- +goose Down
ALTER TABLE voice_bindings DROP COLUMN synced_at;
