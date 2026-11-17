-- +goose Up

-- Where `router replicate` stands, here, in copying an app from another deployment: the cursor
-- of the source's changes this deployment holds everything up to, so a later run carries on from
-- it rather than copying the export over what was done here since. The cursor is empty while a
-- copy is underway, which a run that stopped partway leaves behind. read_at is when the source
-- was read for it: the source keeps recording the app's changes only for its retention after a
-- read.
CREATE TABLE data_move_cursors (
    customer_id TEXT NOT NULL,
    source TEXT NOT NULL,
    cursor BIGINT,
    read_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (customer_id, source)
);

-- +goose Down
DROP TABLE data_move_cursors;
