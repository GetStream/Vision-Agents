// Package turbopuffertest serves turbopuffer's API over HTTP for the suites that need a
// turbopuffer to talk to and have no key. Every answer is shaped as turbopuffer's API
// reference documents it (read 2026-10-23; see ../testdata/README.md), billing object
// included, and none was recorded from a real call. It keeps what is written to it, so a
// passage upserted can be fetched back.
package turbopuffertest

import (
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// MinimumQueried is what turbopuffer bills a query on a namespace smaller than it: 1.28 GB.
const MinimumQueried = 1_280_000_000

// Server is a turbopuffer in process.
type Server struct {
	*httptest.Server

	mu sync.Mutex
	// namespaces are the passages each namespace holds, by id.
	namespaces map[string]map[string]row
	// sizes overrides what a namespace's metadata reports it holds.
	sizes map[string]int64
	// unreadable namespaces answer their metadata with a 500.
	unreadable map[string]bool
	// writesLeft is how many more writes are taken before every write answers a 500; below
	// zero takes them all.
	writesLeft int
}

type row struct {
	ID     string `json:"id"`
	Text   string `json:"text"`
	Source string `json:"source"`
}

// bytes is the row's logical size as turbopuffer bills it: the bytes of each value.
func (r row) bytes() int64 { return int64(len(r.ID) + len(r.Text) + len(r.Source)) }

// Start serves a turbopuffer until the test ends.
func Start(t testing.TB) *Server {
	t.Helper()
	server := &Server{
		namespaces: map[string]map[string]row{},
		sizes:      map[string]int64{},
		unreadable: map[string]bool{},
		writesLeft: -1,
	}
	server.Server = httptest.NewServer(http.HandlerFunc(server.serve))
	t.Cleanup(server.Close)
	return server
}

// Holds says a namespace exists and holds bytes, whatever was written to it: its metadata
// reports them, and a query on it is billed as searching them.
func (s *Server) Holds(namespace string, bytes int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.namespaces[namespace] == nil {
		s.namespaces[namespace] = map[string]row{}
	}
	s.sizes[namespace] = bytes
}

// Unreadable makes a namespace's metadata answer a 500, as turbopuffer having a bad moment
// does.
func (s *Server) Unreadable(namespace string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.unreadable[namespace] = true
}

// RefuseWritesAfter takes n more writes and answers every one after them with a 500, as a
// write that fails partway through a large ingest does.
func (s *Server) RefuseWritesAfter(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.writesLeft = n
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	path := r.URL.Path
	switch {
	case r.Method == http.MethodGet && path == "/v1/namespaces":
		s.list(w, r)
	case r.Method == http.MethodGet && strings.HasPrefix(path, "/v1/namespaces/") && strings.HasSuffix(path, "/metadata"):
		s.metadata(w, strings.TrimSuffix(strings.TrimPrefix(path, "/v1/namespaces/"), "/metadata"))
	case r.Method == http.MethodPost && strings.HasPrefix(path, "/v2/namespaces/") && strings.HasSuffix(path, "/query"):
		s.query(w, r, strings.TrimSuffix(strings.TrimPrefix(path, "/v2/namespaces/"), "/query"))
	case r.Method == http.MethodPost && strings.HasPrefix(path, "/v2/namespaces/"):
		s.write(w, r, strings.TrimPrefix(path, "/v2/namespaces/"))
	default:
		http.NotFound(w, r)
	}
}

func (s *Server) list(w http.ResponseWriter, r *http.Request) {
	prefix, cursor := r.URL.Query().Get("prefix"), r.URL.Query().Get("cursor")
	size, err := strconv.Atoi(r.URL.Query().Get("page_size"))
	if err != nil || size <= 0 {
		size = 100
	}
	var names []string
	for name := range s.namespaces {
		if strings.HasPrefix(name, prefix) && name > cursor {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	type namespace struct {
		ID string `json:"id"`
	}
	page := struct {
		Namespaces []namespace `json:"namespaces"`
		NextCursor string      `json:"next_cursor,omitempty"`
	}{Namespaces: []namespace{}}
	for _, name := range names[:min(size, len(names))] {
		page.Namespaces = append(page.Namespaces, namespace{ID: name})
	}
	if len(names) > size {
		page.NextCursor = names[size-1]
	}
	answer(w, page)
}

func (s *Server) metadata(w http.ResponseWriter, namespace string) {
	rows, ok := s.namespaces[namespace]
	if !ok {
		http.Error(w, `{"status":"error","error":"namespace not found"}`, http.StatusNotFound)
		return
	}
	if s.unreadable[namespace] {
		http.Error(w, `{"status":"error","error":"internal error"}`, http.StatusInternalServerError)
		return
	}
	size, sized := s.sizes[namespace]
	if !sized {
		for _, stored := range rows {
			size += stored.bytes()
		}
	}
	answer(w, map[string]any{
		"approx_logical_bytes": size,
		"approx_row_count":     len(rows),
		"created_at":           "2026-10-01T09:30:45Z",
		"updated_at":           "2026-10-22T17:12:14Z",
		"encryption":           map[string]string{"mode": "default"},
		"index":                map[string]string{"status": "up-to-date"},
	})
}

// query answers a BM25 query with the namespace's passages that hold a word of it, and a
// fetch with the ids its filter names, billed the namespace's size or the minimum, whichever
// is greater, and the bytes returned.
func (s *Server) query(w http.ResponseWriter, r *http.Request, namespace string) {
	rows, ok := s.namespaces[namespace]
	if !ok {
		http.Error(w, `{"status":"error","error":"namespace not found"}`, http.StatusNotFound)
		return
	}
	var asked struct {
		RankBy  []any `json:"rank_by"`
		Filters []any `json:"filters"`
		Limit   int   `json:"limit"`
	}
	if err := json.NewDecoder(r.Body).Decode(&asked); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var matched []row
	var size int64
	for _, id := range slices.Sorted(maps.Keys(rows)) {
		stored := rows[id]
		size += stored.bytes()
		if matches(stored, asked.RankBy, asked.Filters) {
			matched = append(matched, stored)
		}
	}
	if held, sized := s.sizes[namespace]; sized {
		size = held
	}
	if asked.Limit > 0 && len(matched) > asked.Limit {
		matched = matched[:asked.Limit]
	}
	var returned int64
	answered := make([]map[string]any, 0, len(matched))
	for _, found := range matched {
		returned += found.bytes()
		answered = append(answered, map[string]any{"$dist": 1.0, "id": found.ID, "text": found.Text, "source": found.Source})
	}
	answer(w, map[string]any{
		"rows": answered,
		"billing": map[string]int64{
			"billable_logical_bytes_queried":  max(size, MinimumQueried),
			"billable_logical_bytes_returned": returned,
		},
		"performance": map[string]any{"cache_temperature": "hot", "server_total_ms": 3, "approx_namespace_size": len(rows)},
	})
}

// matches is whether a passage answers the query: one of its words for a BM25 rank, its id
// among those an In filter names.
func matches(stored row, rankBy, filters []any) bool {
	if len(filters) == 3 && filters[1] == "In" {
		ids, _ := filters[2].([]any)
		return slices.Contains(ids, any(stored.ID))
	}
	if len(rankBy) == 3 && rankBy[1] == "BM25" {
		text, _ := rankBy[2].(string)
		for _, word := range strings.Fields(strings.ToLower(text)) {
			if strings.Contains(strings.ToLower(stored.Text), word) {
				return true
			}
		}
		return false
	}
	return true
}

// write upserts and deletes, billed the logical bytes of what was upserted and the ids of
// what was deleted.
func (s *Server) write(w http.ResponseWriter, r *http.Request, namespace string) {
	if s.writesLeft == 0 {
		http.Error(w, `{"status":"error","error":"internal error"}`, http.StatusInternalServerError)
		return
	}
	if s.writesLeft > 0 {
		s.writesLeft--
	}
	var asked struct {
		UpsertRows []row    `json:"upsert_rows"`
		Deletes    []string `json:"deletes"`
	}
	if err := json.NewDecoder(r.Body).Decode(&asked); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	rows, ok := s.namespaces[namespace]
	if !ok {
		if len(asked.UpsertRows) == 0 {
			http.Error(w, `{"status":"error","error":"namespace not found"}`, http.StatusNotFound)
			return
		}
		rows = map[string]row{}
		s.namespaces[namespace] = rows
	}
	var written int64
	for _, upserted := range asked.UpsertRows {
		rows[upserted.ID] = upserted
		written += upserted.bytes()
	}
	deleted := 0
	for _, id := range asked.Deletes {
		written += int64(len(id))
		if _, held := rows[id]; held {
			delete(rows, id)
			deleted++
		}
	}
	response := map[string]any{
		"rows_affected": len(asked.UpsertRows) + deleted,
		"billing":       map[string]int64{"billable_logical_bytes_written": written},
		"performance":   map[string]int{"server_total_ms": 5},
	}
	if len(asked.UpsertRows) > 0 {
		response["rows_upserted"] = len(asked.UpsertRows)
	}
	if len(asked.Deletes) > 0 {
		response["rows_deleted"] = deleted
	}
	answer(w, response)
}

func answer(w http.ResponseWriter, body any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(body)
}
