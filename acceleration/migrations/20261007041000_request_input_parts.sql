-- +goose Up

-- What an LLM prompt was made of: its instructions, the conversation's words, the tools it
-- was offered, the tools it used, images and video. No provider reports this, so the router
-- estimates it from the request and scales it to the provider's input count, which is why
-- the six sum to input_tokens. Rows from before, and from the other modalities, read zero.
ALTER TABLE requests ADD COLUMN input_instruction_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN input_message_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN input_tool_definition_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN input_tool_use_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN input_image_tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN input_video_tokens BIGINT NOT NULL DEFAULT 0;

-- +goose Down
ALTER TABLE requests DROP COLUMN input_video_tokens;
ALTER TABLE requests DROP COLUMN input_image_tokens;
ALTER TABLE requests DROP COLUMN input_tool_use_tokens;
ALTER TABLE requests DROP COLUMN input_tool_definition_tokens;
ALTER TABLE requests DROP COLUMN input_message_tokens;
ALTER TABLE requests DROP COLUMN input_instruction_tokens;
