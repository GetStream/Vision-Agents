-- +goose Up

-- One leg of a phone call: what call minutes are billed from, and what a limit on concurrent
-- calls counts. A placed call and each transfer into it are an outbound leg each, recorded when
-- the vendor takes the dial and started when the person's phone joins the Stream call; a call
-- to one of the app's numbers is an inbound one, started when it arrives. A leg ends when its
-- phone leaves the call, or when the call ends. Kept in the shard that opened it.
CREATE TABLE call_legs (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- Who the leg's minutes are billed to and counted against, from the leg's owner when it
    -- opened: a key's outbound call keeps its key; an inbound one is phone:inbound.
    organization_id TEXT NOT NULL DEFAULT '',
    organization_tier TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    usage_categories TEXT[] NOT NULL DEFAULT '{}',
    -- The Stream app the call is in. NULL is the deployment's own app.
    stream_app_pk BIGINT,
    call_type TEXT NOT NULL,
    call_id TEXT NOT NULL,
    -- Stream's session of the call the leg is in, and the user session of the leg's phone in
    -- it, which is what a participant leaving names. Empty until known.
    stream_session_id TEXT NOT NULL DEFAULT '',
    participant_session TEXT NOT NULL DEFAULT '',
    vendor TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    local_number TEXT NOT NULL,
    remote_number TEXT NOT NULL,
    -- The ISO 3166-1 alpha-2 country the leg is priced by, empty when unknown.
    country TEXT NOT NULL DEFAULT '',
    agent_id TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- When the vendor took the dial, or the call arrived; when the phone joined (NULL while
    -- it rings); when the leg ended, and what ended it: left, ended, swept or unanswered.
    dialed_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    end_reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX call_legs_call_idx ON call_legs (call_type, call_id) WHERE ended_at IS NULL;
-- The calls an organization has up, and the legs open longest, which the sweep ends.
CREATE INDEX call_legs_open_idx ON call_legs (organization_id) WHERE ended_at IS NULL AND started_at IS NOT NULL;
CREATE INDEX call_legs_open_started_idx ON call_legs (started_at) WHERE ended_at IS NULL;
CREATE INDEX call_legs_ringing_idx ON call_legs (dialed_at) WHERE ended_at IS NULL AND started_at IS NULL;
-- The legs that ended lately, which the sweep checks were billed.
CREATE INDEX call_legs_ended_idx ON call_legs (ended_at) WHERE started_at IS NOT NULL;
-- A call that arrives is one leg per Stream session, however often or late the event that
-- announces it is delivered, the leg ended or not.
CREATE UNIQUE INDEX call_legs_inbound_idx ON call_legs (call_type, call_id, COALESCE(stream_app_pk, 0), stream_session_id)
    WHERE direction = 'inbound';
-- A phone that joins is one leg, however often the event that says so is delivered.
CREATE UNIQUE INDEX call_legs_participant_idx ON call_legs (call_type, call_id, COALESCE(stream_app_pk, 0), participant_session)
    WHERE participant_session <> '';

-- A phone's join or leave of a call that no leg was waiting for when it was delivered. Stream
-- does not order its call events: a placed call's phone can join before the vendor's answer
-- writes its leg, a caller before the session_started that opens theirs, and a phone's leave
-- can come before its join. The leg that opens later takes its phone's join from here and
-- starts when the phone joined, and a join whose leave came first ends the leg it connects.
-- Kept in the shard that heard it, and deleted by the leg sweep a day on.
CREATE TABLE call_participants (
    id BIGSERIAL PRIMARY KEY,
    -- The app the hook came from, and whether it was the deployment's own, whose legs may
    -- name no app: what AppScope reads off the hook.
    stream_app_pk BIGINT,
    unpinned BOOLEAN NOT NULL DEFAULT false,
    call_type TEXT NOT NULL,
    call_id TEXT NOT NULL,
    participant_session TEXT NOT NULL,
    -- The number off the phone's sip-<number> participant id, empty when it named none.
    number TEXT NOT NULL DEFAULT '',
    joined_at TIMESTAMPTZ,
    left_at TIMESTAMPTZ,
    recorded_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX call_participants_idx ON call_participants (call_type, call_id, COALESCE(stream_app_pk, 0), participant_session);
CREATE INDEX call_participants_recorded_idx ON call_participants (recorded_at);

-- +goose Down
DROP TABLE call_participants;
DROP TABLE call_legs;
