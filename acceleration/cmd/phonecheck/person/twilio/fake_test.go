package twilio

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
	fakeSID   = "AC00000000000000000000000000000000"
	fakeToken = "secret-value"
)

// fakeCall is one call on the fake. Each read of it moves to its next status; the last stays.
type fakeCall struct {
	SID      string
	Statuses []string
	Start    time.Time
}

func (c *fakeCall) status() string { return c.Statuses[0] }

func (c *fakeCall) advance() {
	if len(c.Statuses) > 1 {
		c.Statuses = c.Statuses[1:]
	}
}

// fakeTwilio answers the few Calls requests the adapter makes and records every request.
type fakeTwilio struct {
	server *httptest.Server

	mu        sync.Mutex
	requests  []string // "METHOD path"
	placed    []url.Values
	queries   []url.Values
	completed []string
	calls     map[string]*fakeCall
	listed    []string // sids the Calls list returns
	// placedStatuses is the life of a call the adapter places.
	placedStatuses []string
	failWith       int
}

func newFakeTwilio() *fakeTwilio {
	f := &fakeTwilio{calls: map[string]*fakeCall{}, placedStatuses: []string{"completed"}}
	f.server = httptest.NewServer(http.HandlerFunc(f.handle))
	return f
}

func (f *fakeTwilio) add(sid string, start time.Time, statuses ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls[sid] = &fakeCall{SID: sid, Statuses: statuses, Start: start}
}

// list makes the Calls list return these calls.
func (f *fakeTwilio) list(sids ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.listed = sids
}

func (f *fakeTwilio) handle(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, r.Method+" "+r.URL.Path)
	if user, pass, ok := r.BasicAuth(); !ok || user != fakeSID || pass != fakeToken {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	if f.failWith != 0 {
		w.WriteHeader(f.failWith)
		_, _ = w.Write([]byte(`{"message":"the fake is failing"}`))
		return
	}
	_ = r.ParseForm()
	prefix := "/2010-04-01/Accounts/" + fakeSID + "/Calls"
	rest, ok := strings.CutPrefix(r.URL.Path, prefix)
	switch {
	case !ok:
		w.WriteHeader(http.StatusNotFound)
	case rest == ".json" && r.Method == http.MethodPost:
		f.place(w, r.PostForm)
	case rest == ".json":
		f.queries = append(f.queries, r.URL.Query())
		f.writeList(w)
	case r.Method == http.MethodPost:
		f.complete(w, strings.TrimSuffix(strings.TrimPrefix(rest, "/"), ".json"))
	default:
		sid := strings.TrimSuffix(strings.TrimPrefix(rest, "/"), ".json")
		f.writeCall(w, f.calls[sid])
	}
}

func (f *fakeTwilio) place(w http.ResponseWriter, form url.Values) {
	f.placed = append(f.placed, form)
	sid := "CAplaced"
	f.calls[sid] = &fakeCall{SID: sid, Statuses: f.placedStatuses, Start: time.Now()}
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(map[string]string{"sid": sid})
}

func (f *fakeTwilio) complete(w http.ResponseWriter, sid string) {
	f.completed = append(f.completed, sid)
	if c := f.calls[sid]; c != nil {
		c.Statuses = []string{"completed"}
	}
	_ = json.NewEncoder(w).Encode(map[string]string{"sid": sid, "status": "completed"})
}

func (f *fakeTwilio) writeList(w http.ResponseWriter) {
	records := []map[string]any{}
	for _, sid := range f.listed {
		records = append(records, record(f.calls[sid]))
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"calls": records})
}

func (f *fakeTwilio) writeCall(w http.ResponseWriter, c *fakeCall) {
	if c == nil {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	body := record(c)
	c.advance()
	_ = json.NewEncoder(w).Encode(body)
}

func record(c *fakeCall) map[string]any {
	rec := map[string]any{
		"sid": c.SID, "status": c.status(),
		"start_time": c.Start.UTC().Format(time.RFC1123Z), "end_time": nil,
	}
	if c.status() == "completed" {
		rec["end_time"] = c.Start.Add(25 * time.Second).UTC().Format(time.RFC1123Z)
	}
	return rec
}

func (f *fakeTwilio) paths() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.requests...)
}

func (f *fakeTwilio) completedSIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.completed...)
}
