package api

import (
	"encoding/base64"
	"encoding/json"
	"errors"
)

var errBadCursor = errors.New("cursor is not one this list handed out")

// encodeCursor writes the last position of a page as the opaque cursor the next request
// hands back.
func encodeCursor(position any) *string {
	// A position is a struct of times, strings and numbers, which always marshals.
	raw, _ := json.Marshal(position)
	encoded := base64.RawURLEncoding.EncodeToString(raw)
	return &encoded
}

// decodeCursor reads a cursor back. An absent or empty one is the first page, and nil.
func decodeCursor[T any](value *string) (*T, error) {
	if value == nil || *value == "" {
		return nil, nil
	}
	raw, err := base64.RawURLEncoding.DecodeString(*value)
	if err != nil {
		return nil, errBadCursor
	}
	var position T
	if err := json.Unmarshal(raw, &position); err != nil {
		return nil, errBadCursor
	}
	return &position, nil
}

// page drops the row past the limit that the queries fetch, reporting whether it was there.
func page[T any](rows []T, limit int) ([]T, bool) {
	if len(rows) > limit {
		return rows[:limit], true
	}
	return rows, false
}
