package telnyx

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	fakeKey     = "KEY-secret-value"
	fakeAccount = "00000000-0000-0000-0000-000000000000"
	fakeAppID   = "1234567890"
	fakeSID     = "v3:placed"
)

// fakeTelnyx answers the three TeXML call requests the adapter makes, as Telnyx's API
// reference documents them, and records every request.
type fakeTelnyx struct {
	server *httptest.Server

	mu        sync.Mutex
	requests  []string // "METHOD path"
	placed    []map[string]string
	hungUp    []url.Values
	statuses  []string // the life of the placed call; each read moves on, the last stays
	failWith  int
	notFound  int  // reads of the placed call that answer 404 before it becomes readable
	noSID     bool // the reply to a placed call names no call sid
	badTimes  bool // the call's times come in a format the adapter cannot read
	callsPath string
}

func newFakeTelnyx() *fakeTelnyx {
	f := &fakeTelnyx{statuses: []string{"completed"}, callsPath: "/v2/texml/Accounts/" + fakeAccount + "/Calls"}
	f.server = httptest.NewServer(http.HandlerFunc(f.handle))
	return f
}

func (f *fakeTelnyx) handle(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, r.Method+" "+r.URL.Path)
	if r.Header.Get("Authorization") != "Bearer "+fakeKey {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	if f.failWith != 0 {
		w.WriteHeader(f.failWith)
		_, _ = w.Write([]byte(`{"errors":[{"detail":"the fake is failing"}]}`))
		return
	}
	rest, ok := strings.CutPrefix(r.URL.Path, f.callsPath)
	switch {
	case !ok:
		w.WriteHeader(http.StatusNotFound)
	case rest == "" && r.Method == http.MethodPost:
		f.place(w, r)
	case rest == "/"+fakeSID && r.Method == http.MethodPost:
		f.hangup(w, r)
	case rest == "/"+fakeSID && r.Method == http.MethodGet:
		f.read(w)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (f *fakeTelnyx) place(w http.ResponseWriter, r *http.Request) {
	var body map[string]string
	_ = json.NewDecoder(r.Body).Decode(&body)
	f.placed = append(f.placed, body)
	w.WriteHeader(http.StatusOK)
	if f.noSID {
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "queued"})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]string{"call_sid": fakeSID, "status": "queued"})
}

func (f *fakeTelnyx) hangup(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	f.hungUp = append(f.hungUp, r.PostForm)
	f.statuses = []string{"completed"}
	_ = json.NewEncoder(w).Encode(map[string]string{"sid": fakeSID, "status": "completed"})
}

func (f *fakeTelnyx) read(w http.ResponseWriter) {
	if f.notFound > 0 {
		f.notFound--
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"errors":[{"detail":"not found yet"}]}`))
		return
	}
	start := time.Date(2026, 10, 21, 9, 0, 0, 0, time.UTC)
	rec := map[string]any{
		"sid": fakeSID, "status": f.statuses[0],
		"start_time": start.Format(time.RFC1123Z), "end_time": nil,
	}
	if f.statuses[0] == "completed" {
		rec["end_time"] = start.Add(25 * time.Second).Format(time.RFC1123Z)
	}
	if f.badTimes {
		rec["start_time"] = "yesterday"
	}
	if len(f.statuses) > 1 {
		f.statuses = f.statuses[1:]
	}
	_ = json.NewEncoder(w).Encode(rec)
}

func (f *fakeTelnyx) paths() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.requests...)
}

func (f *fakeTelnyx) hangups() []url.Values {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]url.Values(nil), f.hungUp...)
}

func (f *fakeTelnyx) setStatuses(statuses ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.statuses = statuses
}
