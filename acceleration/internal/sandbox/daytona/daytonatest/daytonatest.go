// Package daytonatest serves Daytona's API over HTTP for the suites that need a Daytona to talk
// to and have no key: sandboxes made, read, listed by label, started, stopped and deleted, the
// toolbox code runs in, and the analytics API's usage reports. Every answer is shaped as
// Daytona's Go SDK (v0.211.2) and its analytics API's OpenAPI document read them, and none was
// recorded from a real call. A sandbox keeps its state, so code sent to one Daytona stopped is
// refused as Daytona's toolbox refuses it.
package daytonatest

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// Organization is the Daytona organization every sandbox here belongs to.
const Organization = "org-test"

// The states a sandbox goes through here, as Daytona names them.
const (
	Started   = "started"
	Stopped   = "stopped"
	Destroyed = "destroyed"
)

// Box is a sandbox as Daytona keeps it.
type Box struct {
	ID     string
	State  string
	Labels map[string]string
	// CPU, Memory and Disk are what it reserves, in vCPUs and GiB.
	CPU, Memory, Disk float64
	Class             string
	CreatedAt         time.Time
	// AutoStop and AutoDelete are the intervals it was created with, in minutes; nil is unset.
	AutoStop, AutoDelete *float64
}

// Server is a Daytona in process.
type Server struct {
	*httptest.Server

	mu    sync.Mutex
	boxes map[string]*Box
	// gone are the sandboxes Daytona no longer knows of: a read of one is a 404.
	gone map[string]bool
	// sequence names the next sandbox made.
	sequence int
	// creations is the body of every request for a sandbox, calls every request to the
	// sandbox API as "<method> <path>", ran the code of every run with the sandbox it ran in,
	// and deleted every sandbox deleted through the API.
	creations []map[string]any
	calls     []string
	ran       []string
	deleted   []string
	// keys is the bearer key every request to the sandbox API carried.
	keys []string
	// files is what each sandbox's file system holds, by path.
	files map[string]map[string][]byte
	// writes is what every code run leaves in its sandbox's file system, by path.
	writes map[string][]byte
	// refuseDeletes answers every delete with a 500.
	refuseDeletes bool
	// usage is the analytics API's per-sandbox answer, and periods each sandbox's periods.
	usage   []byte
	periods map[string][]byte
	// asked is every analytics request, as its path and query.
	asked []string
}

// Start serves a Daytona until the test ends.
func Start(t testing.TB) *Server {
	t.Helper()
	server := &Server{boxes: map[string]*Box{}, gone: map[string]bool{}, files: map[string]map[string][]byte{},
		periods: map[string][]byte{}}
	server.Server = httptest.NewServer(http.HandlerFunc(server.serve))
	t.Cleanup(server.Close)
	return server
}

// APIURL is the sandbox API, as DAYTONA_API_URL names it.
func (s *Server) APIURL() string { return s.URL + "/api" }

// Holds puts a sandbox there as though it had been made earlier, by anybody.
func (s *Server) Holds(box Box) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if box.State == "" {
		box.State = Started
	}
	if box.Class == "" {
		box.Class = "container"
	}
	if box.CreatedAt.IsZero() {
		box.CreatedAt = time.Now().UTC()
	}
	box.Labels = maps.Clone(box.Labels)
	s.boxes[box.ID] = &box
}

// Writes has every code run from now on leave data at path in its sandbox, as code that
// writes a file does.
func (s *Server) Writes(path string, data []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.writes == nil {
		s.writes = map[string][]byte{}
	}
	s.writes[path] = data
}

// Stop stops a sandbox, as Daytona's auto-stop does.
func (s *Server) Stop(id string) { s.set(id, Stopped) }

// SetState puts a sandbox in any state Daytona names, such as error or build_failed.
func (s *Server) SetState(id, state string) { s.set(id, state) }

// Gone deletes a sandbox and forgets it, as Daytona's auto-delete does: a read of it is a 404.
func (s *Server) Gone(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.boxes, id)
	s.gone[id] = true
}

func (s *Server) set(id, state string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if box, ok := s.boxes[id]; ok {
		box.State = state
	}
}

// Sandbox is the sandbox id as Daytona keeps it now, and false once it is gone.
func (s *Server) Sandbox(id string) (Box, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	box, ok := s.boxes[id]
	if !ok {
		return Box{}, false
	}
	copied := *box
	copied.Labels = maps.Clone(box.Labels)
	return copied, true
}

// Created is every sandbox made through the API, in order.
func (s *Server) Created() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	created := make([]string, 0, s.sequence)
	for i := 1; i <= s.sequence; i++ {
		created = append(created, fmt.Sprintf("sandbox-%d", i))
	}
	return created
}

// Creations is the body of every request for a sandbox, in order.
func (s *Server) Creations() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.creations)
}

// Calls is every request to the sandbox API, as "<method> <path>", in order.
func (s *Server) Calls() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.calls)
}

// Keys is the key every request to the sandbox API was made with, in order.
func (s *Server) Keys() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.keys)
}

// Ran is the code of every run, as "<sandbox>: <code>", in order.
func (s *Server) Ran() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.ran)
}

// Deleted is every sandbox deleted through the API, in order.
func (s *Server) Deleted() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.deleted)
}

// Files is what a sandbox's file system holds, by path.
func (s *Server) Files(id string) map[string][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return maps.Clone(s.files[id])
}

// RefuseDeletes has every delete answer a 500, as Daytona failing to delete does, or not.
func (s *Server) RefuseDeletes(refuse bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refuseDeletes = refuse
}

// Bills is what the analytics API answers for the organization's usage per sandbox: a JSON
// array of its SandboxUsage objects.
func (s *Server) Bills(usage []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.usage = usage
}

// Periods is what the analytics API answers for one sandbox's usage periods: a JSON array of
// its UsagePeriod objects.
func (s *Server) Periods(id string, periods []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.periods[id] = periods
}

// Asked is every analytics request, as its path and query, in order.
func (s *Server) Asked() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.asked)
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	path := r.URL.Path
	switch {
	case strings.HasPrefix(path, "/api/socket.io"):
		w.WriteHeader(http.StatusNotFound)
	case path == "/api/config":
		answer(w, http.StatusOK, map[string]any{"analyticsApiUrl": s.URL + "/analytics"})
	case strings.HasPrefix(path, "/analytics/"):
		s.analytics(w, r)
	case strings.HasPrefix(path, "/api/sandbox"):
		s.calls = append(s.calls, r.Method+" "+path)
		s.keys = append(s.keys, strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
		s.sandboxAPI(w, r, strings.Trim(strings.TrimPrefix(path, "/api/sandbox"), "/"))
	case strings.HasPrefix(path, "/toolbox/"):
		id, rest, _ := strings.Cut(strings.TrimPrefix(path, "/toolbox/"), "/")
		s.toolbox(w, r, id, rest)
	default:
		answer(w, http.StatusNotFound, map[string]any{"message": "no such route " + path})
	}
}

// sandboxAPI serves /api/sandbox and what is under it. The caller holds mu.
func (s *Server) sandboxAPI(w http.ResponseWriter, r *http.Request, rest string) {
	id, action, _ := strings.Cut(rest, "/")
	switch {
	case r.Method == http.MethodPost && rest == "":
		s.create(w, r)
	case r.Method == http.MethodGet && rest == "":
		s.list(w, r)
	case s.gone[id] || s.boxes[id] == nil:
		answer(w, http.StatusNotFound, map[string]any{"message": "sandbox " + id + " not found", "statusCode": 404})
	case r.Method == http.MethodGet && action == "":
		answer(w, http.StatusOK, s.sandbox(s.boxes[id]))
	case r.Method == http.MethodPost && action == "start":
		s.boxes[id].State = Started
		answer(w, http.StatusOK, s.sandbox(s.boxes[id]))
	case r.Method == http.MethodPost && action == "stop":
		s.boxes[id].State = Stopped
		answer(w, http.StatusOK, s.sandbox(s.boxes[id]))
	case r.Method == http.MethodDelete && action == "":
		if s.refuseDeletes {
			answer(w, http.StatusInternalServerError, map[string]any{"message": "could not delete the sandbox"})
			return
		}
		s.deleted = append(s.deleted, id)
		s.boxes[id].State = Destroyed
		answer(w, http.StatusOK, s.sandbox(s.boxes[id]))
	default:
		answer(w, http.StatusNotFound, map[string]any{"message": "no such route"})
	}
}

// create makes a sandbox of what the request asked for, or Daytona's default size. The caller
// holds mu.
func (s *Server) create(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	body := map[string]any{}
	_ = json.Unmarshal(raw, &body)
	s.creations = append(s.creations, body)
	s.sequence++
	box := &Box{ID: fmt.Sprintf("sandbox-%d", s.sequence), State: Started, Labels: map[string]string{}, CPU: 1, Memory: 1,
		Disk: 3, Class: "container", CreatedAt: time.Now().UTC()}
	if labels, ok := body["labels"].(map[string]any); ok {
		for name, value := range labels {
			box.Labels[name], _ = value.(string)
		}
	}
	for field, size := range map[string]*float64{"cpu": &box.CPU, "memory": &box.Memory, "disk": &box.Disk} {
		if asked, ok := body[field].(float64); ok && asked > 0 {
			*size = asked
		}
	}
	if minutes, ok := body["autoStopInterval"].(float64); ok {
		box.AutoStop = &minutes
	}
	if minutes, ok := body["autoDeleteInterval"].(float64); ok {
		box.AutoDelete = &minutes
	}
	s.boxes[box.ID] = box
	answer(w, http.StatusCreated, s.sandbox(box))
}

// list answers the sandboxes whose labels hold every label asked for, leaving out the deleted
// ones as Daytona does, in one page. The caller holds mu.
func (s *Server) list(w http.ResponseWriter, r *http.Request) {
	wanted := map[string]string{}
	if labels := r.URL.Query().Get("labels"); labels != "" {
		_ = json.Unmarshal([]byte(labels), &wanted)
	}
	items := []map[string]any{}
	for _, id := range slices.Sorted(maps.Keys(s.boxes)) {
		box := s.boxes[id]
		matches := box.State != Destroyed
		for name, value := range wanted {
			matches = matches && box.Labels[name] == value
		}
		if matches {
			items = append(items, s.sandbox(box))
		}
	}
	answer(w, http.StatusOK, map[string]any{"items": items, "nextCursor": nil})
}

// sandbox is box as the sandbox API answers it. The caller holds mu.
func (s *Server) sandbox(box *Box) map[string]any {
	answered := map[string]any{"id": box.ID, "organizationId": Organization, "name": box.ID, "user": "daytona",
		"env": map[string]string{}, "labels": box.Labels, "public": false, "networkBlockAll": false, "target": "us",
		"cpu": box.CPU, "gpu": 0, "memory": box.Memory, "disk": box.Disk, "state": box.State,
		"sandboxClass": box.Class, "createdAt": box.CreatedAt.Format(time.RFC3339Nano),
		"toolboxProxyUrl": s.URL + "/toolbox"}
	if box.AutoStop != nil {
		answered["autoStopInterval"] = *box.AutoStop
	}
	if box.AutoDelete != nil {
		answered["autoDeleteInterval"] = *box.AutoDelete
	}
	return answered
}

// toolbox serves a sandbox's toolbox: code runs, commands and files. A sandbox that is not
// started refuses all of it, and one deleted is not found. The caller holds mu.
func (s *Server) toolbox(w http.ResponseWriter, r *http.Request, id, rest string) {
	box, ok := s.boxes[id]
	if s.gone[id] || ok && box.State == Destroyed {
		answer(w, http.StatusNotFound, map[string]any{"message": "sandbox " + id + " not found", "statusCode": 404})
		return
	}
	if !ok || box.State != Started {
		answer(w, http.StatusBadRequest, map[string]any{"message": "sandbox " + id + " is not started"})
		return
	}
	if s.files[id] == nil {
		s.files[id] = map[string][]byte{}
	}
	switch rest {
	case "process/code-run":
		var body struct {
			Code string `json:"code"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		s.ran = append(s.ran, id+": "+body.Code)
		maps.Copy(s.files[id], s.writes)
		answer(w, http.StatusOK, map[string]any{"result": "ran\n", "exitCode": 0})
	case "process/execute":
		answer(w, http.StatusOK, map[string]any{"result": "", "exitCode": 0})
	case "files/upload-v2":
		file, _, err := r.FormFile("file")
		if err != nil {
			answer(w, http.StatusBadRequest, map[string]any{"message": err.Error()})
			return
		}
		data, _ := io.ReadAll(file)
		name := r.URL.Query().Get("path")
		s.files[id][name] = data
		answer(w, http.StatusOK, map[string]any{"name": name, "path": name, "type": "file"})
	case "files/download":
		data, ok := s.files[id][r.URL.Query().Get("path")]
		if !ok {
			answer(w, http.StatusNotFound, map[string]any{"message": "no such file"})
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(data)
	default:
		answer(w, http.StatusNotFound, map[string]any{"message": "no such toolbox route " + rest})
	}
}

// analytics serves the analytics API's two usage reports for the organization. The caller
// holds mu.
func (s *Server) analytics(w http.ResponseWriter, r *http.Request) {
	s.asked = append(s.asked, r.URL.Path+"?"+r.URL.RawQuery)
	query := r.URL.Query()
	if _, err := time.Parse(time.RFC3339, query.Get("from")); err != nil {
		answer(w, http.StatusBadRequest, map[string]any{"error": "from is not RFC 3339"})
		return
	}
	if _, err := time.Parse(time.RFC3339, query.Get("to")); err != nil {
		answer(w, http.StatusBadRequest, map[string]any{"error": "to is not RFC 3339"})
		return
	}
	rest, ok := strings.CutPrefix(r.URL.Path, "/analytics/organization/"+Organization+"/")
	if !ok {
		answer(w, http.StatusForbidden, map[string]any{"error": "not this key's organization"})
		return
	}
	var body []byte
	switch {
	case rest == "usage/sandbox":
		body = s.usage
	case strings.HasPrefix(rest, "sandbox/") && strings.HasSuffix(rest, "/usage"):
		body = s.periods[strings.TrimSuffix(strings.TrimPrefix(rest, "sandbox/"), "/usage")]
	default:
		answer(w, http.StatusNotFound, map[string]any{"error": "no such report"})
		return
	}
	if body == nil {
		body = []byte("[]")
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(body)
}

func answer(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
