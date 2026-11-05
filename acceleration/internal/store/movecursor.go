package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// MoveCursor is where a copy of an app from another deployment stands here.
type MoveCursor struct {
	// Cursor is the source's cursor this deployment holds every change up to, nil while a copy
	// is underway.
	Cursor *int64
	// ReadAt is when the source was read for it.
	ReadAt time.Time
}

// MoveCursor answers where the copy of a customer from a source stands here, nil when none was
// ever started.
func (s *Store) MoveCursor(ctx context.Context, customerID, source string) (*MoveCursor, error) {
	var saved MoveCursor
	err := s.db.QueryRowContext(ctx, "SELECT cursor, read_at FROM data_move_cursors WHERE customer_id = ? AND source = ?",
		customerID, source).Scan(&saved.Cursor, &saved.ReadAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("store: move cursor: %w", err)
	}
	return &saved, nil
}

// SaveMoveCursor records where the copy of a customer from a source stands here: the cursor the
// source was read up to at readAt, or nil for a copy that has started and not finished.
func (s *Store) SaveMoveCursor(ctx context.Context, customerID, source string, cursor *int64, readAt time.Time) error {
	_, err := s.db.ExecContext(ctx, `
INSERT INTO data_move_cursors (customer_id, source, cursor, read_at) VALUES (?, ?, ?, ?)
ON CONFLICT (customer_id, source) DO UPDATE SET cursor = EXCLUDED.cursor, read_at = EXCLUDED.read_at`,
		customerID, source, cursor, readAt)
	if err != nil {
		return fmt.Errorf("store: save move cursor: %w", err)
	}
	return nil
}

// HoldsCustomer says whether any table an export carries holds a row of the customer's here.
// A table that belongs to a customer through a parent is left out: its rows are written only
// under a parent of the customer's.
func (s *Store) HoldsCustomer(ctx context.Context, customerID string) (bool, error) {
	for _, table := range dataTables {
		if table.customer == "" {
			continue
		}
		where := table.customer + " = ?"
		if table.filter != "" {
			where += " AND " + table.filter
		}
		var holds bool
		query := "SELECT EXISTS (SELECT 1 FROM " + table.name + " WHERE " + where + ")"
		if err := s.sideOf(table).QueryRowContext(ctx, query, customerID).Scan(&holds); err != nil {
			return false, fmt.Errorf("store: holds %s: %w", table.name, err)
		}
		if holds {
			return true, nil
		}
	}
	return false, nil
}
