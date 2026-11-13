-- +goose NO TRANSACTION
-- +goose Up

-- team is the team a session's usage is counted under, the one usage_categories already holds
-- as team:<name>, kept as a column so a list narrowed to one team is served by an index. A
-- session counted under no team has none. The constant default rewrites no row; the lock waits
-- at most 5 s behind a running query, past which the migration fails and the router's next
-- start tries again.
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS team TEXT NOT NULL DEFAULT '';
END $$;
-- +goose StatementEnd

-- The sessions from before take the team their usage categories name.
UPDATE agent_sessions
SET team = substring(category FROM 6)
FROM (
    SELECT asn.id, min(c) AS category
    FROM agent_sessions asn, unnest(asn.usage_categories) AS c
    WHERE c LIKE 'team:%'
    GROUP BY asn.id
) AS teams
WHERE agent_sessions.id = teams.id AND agent_sessions.team = '';

-- Built without stopping the writes of sessions.
CREATE INDEX CONCURRENTLY IF NOT EXISTS agent_sessions_team_idx
    ON agent_sessions (customer_id, team, updated_at DESC, id DESC);

-- +goose Down
DROP INDEX CONCURRENTLY IF EXISTS agent_sessions_team_idx;
-- +goose StatementBegin
DO $$
BEGIN
    SET LOCAL lock_timeout = '5s';
    ALTER TABLE agent_sessions DROP COLUMN IF EXISTS team;
END $$;
-- +goose StatementEnd
