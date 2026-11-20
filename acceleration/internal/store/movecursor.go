package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

// MoveCursor is where a copy of an app from another deployment stands here.
type MoveCursor struct {
	// Cursor is the source's cursor this deployment holds every change up to, nil while a copy
	// is underway.
	Cursor *int64
	// ReadAt is when the source was read for it.
	ReadAt time.Time
}

// dataMoveCursor is MoveCursor as data_move_cursors holds it.
type dataMoveCursor struct {
	bun.BaseModel `bun:"table:data_move_cursors,alias:dmc"`

	CustomerID string    `bun:"customer_id,pk"`
	Source     string    `bun:"source,pk"`
	Cursor     *int64    `bun:"cursor"`
	ReadAt     time.Time `bun:"read_at,notnull"`
}

// MoveCursor answers where the copy of a customer from a source stands here, nil when none was
// ever started.
func (s *Store) MoveCursor(ctx context.Context, customerID, source string) (*MoveCursor, error) {
	var saved dataMoveCursor
	err := s.db.NewSelect().Model(&saved).Where("customer_id = ? AND source = ?", customerID, source).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("store: move cursor: %w", err)
	}
	return &MoveCursor{Cursor: saved.Cursor, ReadAt: saved.ReadAt}, nil
}

// SaveMoveCursor records where the copy of a customer from a source stands here: the cursor the
// source was read up to at readAt, or nil for a copy that has started and not finished. A cursor
// never moves back: of two runs at once, the one that read less cannot undo the other.
func (s *Store) SaveMoveCursor(ctx context.Context, customerID, source string, cursor *int64, readAt time.Time) error {
	_, err := s.db.NewInsert().Model(&dataMoveCursor{CustomerID: customerID, Source: source, Cursor: cursor, ReadAt: readAt}).
		On("CONFLICT (customer_id, source) DO UPDATE").
		Set("cursor = CASE WHEN EXCLUDED.cursor IS NULL THEN NULL ELSE GREATEST(dmc.cursor, EXCLUDED.cursor) END").
		Set("read_at = CASE WHEN EXCLUDED.cursor IS NULL THEN EXCLUDED.read_at ELSE GREATEST(dmc.read_at, EXCLUDED.read_at) END").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save move cursor: %w", err)
	}
	return nil
}

// HoldingTable names the first table an export carries that holds a row of the customer's in the
// shared database, empty when none does. A table kept in the shards is left out: it holds the
// customer's own traffic here, and a shard's pod reads only its own shard, so counting it would
// make the answer depend on which pod runs. A table that belongs to a customer through a parent
// is left out too: its rows are written only under a parent of the customer's.
func (s *Store) HoldingTable(ctx context.Context, customerID string) (string, error) {
	for _, table := range dataTables {
		if table.customer == "" || table.shard {
			continue
		}
		where := table.customer + " = ?"
		if table.filter != "" {
			where += " AND " + table.filter
		}
		var holds bool
		query := "SELECT EXISTS (SELECT 1 FROM " + table.name + " WHERE " + where + ")"
		if err := s.db.QueryRowContext(ctx, query, customerID).Scan(&holds); err != nil {
			return "", fmt.Errorf("store: holds %s: %w", table.name, err)
		}
		if holds {
			return table.name, nil
		}
	}
	return "", nil
}
