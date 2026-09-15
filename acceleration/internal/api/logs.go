package api

import (
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"net/http"
	"strconv"
	"strings"
	"time"
)

func logCursor(id int64) string {
	return base64.RawURLEncoding.EncodeToString([]byte(strconv.FormatInt(id, 10)))
}
func readLogCursor(value string) (int64, error) {
	if value == "" {
		return 0, nil
	}
	raw, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil {
		return 0, err
	}
	id, err := strconv.ParseInt(string(raw), 10, 64)
	if id < 0 {
		return 0, fmt.Errorf("negative cursor")
	}
	return id, err
}
func (s *Server) logFilter(w http.ResponseWriter, r *http.Request) (store.LogFilter, bool) {
	customer, ok := CustomerFrom(r.Context())
	if !ok {
		writeError(w, 401, "customer required")
		return store.LogFilter{}, false
	}
	if s.refuseClientSide(w, r) {
		return store.LogFilter{}, false
	}
	if s.store == nil {
		writeError(w, 503, "logging storage unavailable")
		return store.LogFilter{}, false
	}
	q := r.URL.Query()
	f := store.LogFilter{CustomerID: customer, ConfigID: q.Get("config_id"), SessionID: q.Get("session_id"), UserID: q.Get("user_id"), Search: q.Get("q"), Severity: q.Get("severity"), Limit: 250}
	if f.Severity != "" && f.Severity != "error" && f.Severity != "info" {
		writeError(w, 400, "invalid severity")
		return f, false
	}
	if len(f.Search) > 256 {
		writeError(w, 400, "search too long")
		return f, false
	}
	if q.Get("source") != "" {
		for _, v := range strings.Split(q.Get("source"), ",") {
			if v != "user" && v != "agent" && v != "tool" && v != "system" {
				writeError(w, 400, "invalid source")
				return f, false
			}
			f.Sources = append(f.Sources, v)
		}
	}
	for key, target := range map[string]*time.Time{"from": &f.From, "to": &f.To} {
		if q.Get(key) != "" {
			v, err := time.Parse(time.RFC3339Nano, q.Get(key))
			if err != nil {
				writeError(w, 400, "invalid time")
				return f, false
			}
			*target = v
		}
	}
	if !f.To.IsZero() && !f.From.IsZero() && !f.To.After(f.From) {
		writeError(w, 400, "invalid time range")
		return f, false
	}
	if q.Get("limit") != "" {
		v, err := strconv.Atoi(q.Get("limit"))
		if err != nil || v < 1 || v > 250 {
			writeError(w, 400, "limit must be 1 to 250")
			return f, false
		}
		f.Limit = v
	}
	return f, true
}
func logJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(value)
}
func (s *Server) listAgentLogs(w http.ResponseWriter, r *http.Request) {
	f, ok := s.logFilter(w, r)
	if !ok {
		return
	}
	before, err := readLogCursor(r.URL.Query().Get("cursor"))
	if err != nil {
		writeError(w, 400, "invalid cursor")
		return
	}
	high, err := s.store.LogHighWater(r.Context(), f.CustomerID)
	if err != nil {
		writeError(w, 503, "log storage unavailable")
		return
	}
	f.Before = high + 1
	if before > 0 && before < f.Before {
		f.Before = before
	}
	limit := f.Limit
	f.Limit++
	rows, err := s.store.AgentLogs(r.Context(), f)
	if err != nil {
		writeError(w, 503, "log storage unavailable")
		return
	}
	more := len(rows) > limit
	if more {
		rows = rows[:limit]
	}
	next := ""
	if more {
		next = logCursor(rows[len(rows)-1].ID)
	}
	logJSON(w, map[string]any{"items": rows, "has_more": more, "next_cursor": next, "resume_cursor": logCursor(high), "dropped_logs": s.store.LogDrops.Load(), "coverage": "Structured logs are recorded from logging enablement. Earlier session history may contain additional events."})
}
func (s *Server) getAgentLog(w http.ResponseWriter, r *http.Request) {
	f, ok := s.logFilter(w, r)
	if !ok {
		return
	}
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id < 1 {
		writeError(w, 400, "invalid log id")
		return
	}
	entry, err := s.store.AgentLog(r.Context(), f.CustomerID, id)
	if errors.Is(err, sql.ErrNoRows) {
		writeError(w, 404, "log not found")
		return
	}
	if err != nil {
		writeError(w, 503, "log storage unavailable")
		return
	}
	logJSON(w, entry)
}
func (s *Server) streamAgentLogs(w http.ResponseWriter, r *http.Request) {
	f, ok := s.logFilter(w, r)
	if !ok {
		return
	}
	cursor := r.Header.Get("Last-Event-ID")
	if cursor == "" {
		cursor = r.URL.Query().Get("cursor")
	}
	after, err := readLogCursor(cursor)
	if err != nil || cursor == "" {
		writeError(w, 400, "resume cursor required")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("X-Accel-Buffering", "no")
	controller := http.NewResponseController(w)
	send := func(event string, id int64, data any) error {
		_ = controller.SetWriteDeadline(time.Now().Add(10 * time.Second))
		raw, _ := json.Marshal(data)
		if _, err := fmt.Fprintf(w, "id: %s\nevent: %s\ndata: %s\n\n", logCursor(id), event, raw); err != nil {
			return err
		}
		return controller.Flush()
	}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		high, err := s.store.LogHighWater(r.Context(), f.CustomerID)
		if err != nil {
			return
		}
		if high < after {
			_ = send("reset", high, map[string]string{"message": "Log history changed; reload to resume."})
			return
		}
		f.After = after
		f.Before = high + 1
		f.Forward = true
		f.Limit = 250
		rows, err := s.store.AgentLogs(r.Context(), f)
		if err != nil {
			return
		}
		if len(rows) > 0 {
			after = rows[len(rows)-1].ID
			if err := send("logs", after, rows); err != nil {
				return
			}
		}
		if len(rows) == 250 {
			continue
		}
		after = high
		if err := send("checkpoint", after, map[string]any{"dropped_logs": s.store.LogDrops.Load()}); err != nil {
			return
		}
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
		}
	}
}
