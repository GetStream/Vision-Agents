-- +goose Up

-- A skill is a kind of work worth handing to the slower model. There is nothing behind one
-- but a better model answering under different instructions, which is why the row is three
-- strings and a deadline.
--
-- They live here rather than in the config that uses them so several configs can share a
-- skill, and so editing what "explain" means changes it everywhere at once.
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    -- description is the one line the fast model sees, and is the whole of how it decides
    -- when to hand something over.
    description TEXT NOT NULL,
    -- instructions is the full prompt, which only the subagent sees.
    instructions TEXT NOT NULL,
    -- deadline_ms is how long the work may run before it is abandoned. Zero leaves the
    -- harness's own default.
    deadline_ms BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

-- A config names a skill, so two a customer can still reach cannot share a name.
CREATE UNIQUE INDEX skills_name_idx ON skills (customer_id, name)
    WHERE deleted_at IS NULL;

-- +goose Down
DROP TABLE skills;
