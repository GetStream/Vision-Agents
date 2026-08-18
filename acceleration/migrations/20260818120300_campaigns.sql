-- +goose Up

-- A campaign is a list of people to ring and one agent to ring them with. It is stored
-- rather than held in memory because the list outlives any one run of the process: a
-- campaign paused on Friday is the same campaign on Monday, minus whoever was reached.
CREATE TABLE campaigns (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    -- config_id is the agent that makes the calls. Without one the calls are made by an
    -- agent with nothing to say, so it is required in practice and checked in the API.
    config_id TEXT NOT NULL DEFAULT '',
    -- from_number is one of the customer's own, which is what the person sees.
    from_number TEXT NOT NULL DEFAULT '',
    -- concurrency is how many of these calls may be happening at once.
    concurrency INTEGER NOT NULL DEFAULT 1,
    -- state is draft, running, paused or finished.
    state TEXT NOT NULL DEFAULT 'draft',
    tags JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE INDEX campaigns_customer_idx ON campaigns (customer_id, created_at DESC);
CREATE INDEX campaigns_tags_idx ON campaigns USING GIN (tags);

-- One person to ring, and what became of ringing them.
CREATE TABLE campaign_contacts (
    id TEXT PRIMARY KEY,
    -- seq is the order they are rung in, which is the order they were added. A timestamp
    -- cannot say it: a whole list arrives in one insert and shares one created_at.
    seq BIGSERIAL NOT NULL,
    campaign_id TEXT NOT NULL REFERENCES campaigns (id) ON DELETE CASCADE,
    to_number TEXT NOT NULL,
    -- instructions are what this agent should say to this person, added to whatever the
    -- config already says. It is why a campaign is not one call repeated.
    instructions TEXT NOT NULL DEFAULT '',
    -- state is pending, calling, done or failed.
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    -- call_id is the calls row this contact became, so a contact leads to a recording,
    -- a transcript and a review.
    call_id TEXT,
    vendor_call_id TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX campaign_contacts_campaign_idx ON campaign_contacts (campaign_id, seq);
CREATE INDEX campaign_contacts_pending_idx ON campaign_contacts (campaign_id, seq)
    WHERE state = 'pending';

-- +goose Down
DROP TABLE campaign_contacts;
DROP TABLE campaigns;
