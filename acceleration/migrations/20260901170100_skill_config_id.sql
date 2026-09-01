-- +goose Up

-- A skill now belongs to one agent config rather than to the customer. Two agents that
-- both need the same kind of work have one each, so editing what "explain" means for one
-- leaves the other alone.
ALTER TABLE skills ADD COLUMN config_id TEXT NOT NULL DEFAULT '';

-- The old index made a name unique per customer, which is what stops a copy per config
-- being inserted below. Uniqueness moves down to the config with the skill.
DROP INDEX IF EXISTS skills_name_idx;

-- Every config that named a shared skill gets its own copy of it, because that is what
-- the sharing meant. A skill nothing named is not copied, and is dropped below: it was
-- reachable only by a config naming it, and no config did.
INSERT INTO skills (id, customer_id, config_id, name, description, instructions,
                    deadline_ms, created_at, updated_at)
SELECT md5(random()::text || clock_timestamp()::text),
       shared.customer_id,
       config.id,
       shared.name,
       shared.description,
       shared.instructions,
       shared.deadline_ms,
       shared.created_at,
       now()
FROM skills AS shared
JOIN agent_configs AS config
  ON config.customer_id = shared.customer_id
 AND config.deleted_at IS NULL
 AND config.skills ? shared.name
WHERE shared.deleted_at IS NULL
  AND shared.config_id = '';

DELETE FROM skills WHERE config_id = '';

CREATE UNIQUE INDEX skills_name_idx ON skills (customer_id, config_id, name)
    WHERE deleted_at IS NULL;

-- +goose Down
DROP INDEX IF EXISTS skills_name_idx;
ALTER TABLE skills DROP COLUMN config_id;
CREATE UNIQUE INDEX skills_name_idx ON skills (customer_id, name)
    WHERE deleted_at IS NULL;
