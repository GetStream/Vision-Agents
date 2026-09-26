package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// defaultChangeLimit and maxChangeLimit bound one page of changes. A page is held in
// memory on both sides, and a row can be a whole agent response, so the ceiling is there
// to keep a caller from asking for more than either end can hold.
const (
	defaultChangeLimit = 500
	maxChangeLimit     = 1000
)

// dataLine is one line of an export.
//
// Rows, and then a last line naming the cursor their changes carry on from. It is last
// rather than first because it is also what says the export finished: a stream that died
// halfway has no cursor, and importing one without noticing would leave somebody with
// half their data and no way to tell. A line carrying a change rather than a row is what
// replaying onto the other deployment sends, so one endpoint takes both and a move is the
// same request twice.
type dataLine struct {
	Cursor   *int64            `json:"cursor,omitempty"`
	Customer string            `json:"customer,omitempty"`
	At       *time.Time        `json:"at,omitempty"`
	Table    string            `json:"table,omitempty"`
	Row      json.RawMessage   `json:"row,omitempty"`
	Change   *store.DataChange `json:"change,omitempty"`
}

// exportData writes everything the calling app has, as one JSON object per line.
func (s *Server) exportData(w http.ResponseWriter, r *http.Request) {
	customerID, ok := s.dataMover(w, r)
	if !ok {
		return
	}

	// Recording starts before the snapshot is read rather than after, so a write that
	// lands while the export is being streamed is in one of the two and not in neither.
	if err := s.store.StartDataCapture(r.Context(), customerID, s.dataRetention); err != nil {
		s.logger.Error("could not start recording changes", "customer", customerID, "error", err)
		writeError(w, http.StatusServiceUnavailable, "changes cannot be recorded, so an export would be a copy nothing could catch up from")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	encoder := json.NewEncoder(w)

	// A failure before the first line is still an error document. One after it is a
	// truncated stream, which the missing cursor line on the end is what says.
	written := int64(0)
	cursor, err := s.store.ExportCustomer(r.Context(), customerID, func(table string, row json.RawMessage) error {
		written++
		return encoder.Encode(dataLine{Table: table, Row: row})
	})
	if err != nil {
		if written == 0 {
			writeError(w, http.StatusServiceUnavailable, "the export could not be read: "+err.Error())
			return
		}
		s.logger.Error("export stopped partway", "customer", customerID, "error", err)
		return
	}

	now := time.Now().UTC()
	if err := encoder.Encode(dataLine{Cursor: &cursor, Customer: customerID, At: &now}); err != nil {
		s.logger.Debug("export was not finished", "customer", customerID, "error", err)
		return
	}
	s.logger.Info("exported a customer", "customer", customerID, "rows", written, "cursor", cursor)
}

// importData writes an export, or a batch of changes, into this deployment.
func (s *Server) importData(w http.ResponseWriter, r *http.Request) {
	customerID, ok := s.dataMover(w, r)
	if !ok {
		return
	}

	var cursor *int64
	rows := int64(0)
	tables := map[string]int64{}

	decoder := json.NewDecoder(r.Body)
	for {
		var line dataLine
		if err := decoder.Decode(&line); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			writeError(w, http.StatusBadRequest, "this is not an export: "+err.Error())
			return
		}

		switch {
		case line.Change != nil:
			if err := s.store.ApplyChanges(r.Context(), customerID, []store.DataChange{*line.Change}); err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			rows++
			tables[line.Change.Table]++
		case line.Table != "":
			if err := s.store.ImportRow(r.Context(), customerID, line.Table, line.Row); err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			rows++
			tables[line.Table]++
		case line.Cursor != nil:
			cursor = line.Cursor
		}
	}

	s.logger.Info("imported a customer", "customer", customerID, "rows", rows)
	writeJSON(w, map[string]any{"rows": rows, "tables": tables, "cursor": cursor})
}

// listDataChanges returns what has happened to the calling app's rows since a cursor.
func (s *Server) listDataChanges(w http.ResponseWriter, r *http.Request) {
	customerID, ok := s.dataMover(w, r)
	if !ok {
		return
	}

	after, err := strconv.ParseInt(nonEmpty(r.URL.Query().Get("after"), "0"), 10, 64)
	if err != nil || after < 0 {
		writeError(w, http.StatusBadRequest, "after is the cursor the last page ended at")
		return
	}
	limit := defaultChangeLimit
	if raw := r.URL.Query().Get("limit"); raw != "" {
		limit, err = strconv.Atoi(raw)
		if err != nil || limit < 1 || limit > maxChangeLimit {
			writeError(w, http.StatusBadRequest, "limit is between 1 and "+strconv.Itoa(maxChangeLimit))
			return
		}
	}

	// Following the changes is the other half of the move, so it says the customer is
	// still going rather than letting the recording expire under them.
	if err := s.store.StartDataCapture(r.Context(), customerID, s.dataRetention); err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}

	changes, cursor, err := s.store.Changes(r.Context(), customerID, after, limit)
	if errors.Is(err, store.ErrChangesExpired) {
		writeError(w, http.StatusGone, "the changes since that cursor are no longer kept: export again")
		return
	}
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}

	writeJSON(w, map[string]any{
		"changes": changes,
		"cursor":  cursor,
		// Fewer changes than were asked for means the end of what has happened so far,
		// which is the moment a switchover is safe.
		"caught_up": len(changes) < limit,
	})
}

// dataMover answers who is moving their data, and writes the refusal itself when nobody
// may.
//
// Three refusals, in the order they stop being guesses. A caller nobody authenticated has
// no data here. A device holding a token its own backend minted may hold a conversation
// and may not walk off with the customer's account, which is the ordinary server-side
// rule. And a deployment running without authentication must refuse outright: there the
// tenant is a header, so serving an export would mean handing any caller any customer's
// data for the price of naming them.
func (s *Server) dataMover(w http.ResponseWriter, r *http.Request) (string, bool) {
	customerID, known := CustomerFrom(r.Context())
	if !known {
		writeError(w, http.StatusUnauthorized, "this operation needs a customer")
		return "", false
	}
	if s.refuseClientSide(w, r) {
		return "", false
	}
	if s.authMode == auth.NoAuth {
		writeError(w, http.StatusForbidden,
			"moving data is refused while this router runs without authentication: "+
				"the customer is whatever the caller says it is, so an export would be anybody's")
		return "", false
	}
	if s.store == nil {
		writeError(w, http.StatusServiceUnavailable, "there is no database here to move")
		return "", false
	}
	return customerID, true
}

func nonEmpty(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(value)
}
