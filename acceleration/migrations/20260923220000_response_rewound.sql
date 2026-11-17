-- +goose Up

-- When a rewind went back past this response. A rewound turn is kept rather than deleted,
-- because what was said still happened, but it is no longer part of the conversation.
ALTER TABLE agent_responses ADD COLUMN rewound_at TIMESTAMPTZ;

-- +goose Down
ALTER TABLE agent_responses DROP COLUMN rewound_at;
