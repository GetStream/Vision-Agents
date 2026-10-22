//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
)

// quotaSpent is what the fake mem0 answers while it is down. Synthetic: it stands for a
// vendor's own words, which name the vendor and the account and must stay in the log.
const quotaSpent = `{"detail":"mem0 org acme-test: monthly memory operations quota exhausted"}`

// MemoryUnavailableSuite runs the router against the mem0 client pointed at a stand-in for
// mem0's API that can be taken down, so a memory store that refuses is a real 429 from a
// real HTTP server, read by the client production uses.
type MemoryUnavailableSuite struct {
	RouterSuite

	mem0   *flakyMem0
	logged *lockedLog
}

func TestMemoryUnavailableSuite(t *testing.T) {
	runSuite(t, new(MemoryUnavailableSuite))
}

func (s *MemoryUnavailableSuite) SetupSuite() {
	s.mem0 = newFlakyMem0(s.T())
	// The key is given, so MEM0_API_KEY in the environment is never read.
	store, err := mem0.New(mem0.Options{APIKey: "test-key-" + s.T().Name(), BaseURL: s.mem0.server.URL})
	s.Require().NoError(err)
	s.memoryStore = store
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *MemoryUnavailableSuite) SetupTest() {
	s.useFixture("standard")
	s.mem0.down.Store(false)
}

func (s *MemoryUnavailableSuite) TestADeleteTheMemoryStoreRefusesKeepsTheSession() {
	opened := s.client.createSession(textSession(nil))
	s.mem0.remember(s.customerID(), opened.Id)
	s.mem0.down.Store(true)

	status, body := s.client.call(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil)

	s.Require().Equal(http.StatusServiceUnavailable, status)
	answer := answered(s.T(), body)
	s.Equal("memory_unavailable", answer.Code)
	s.Equal(ErrorTypeUnavailable, answer.Type)
	s.Equal("The session was not deleted: what it remembered could not be removed right now. "+
		"Try again later.", answer.Message)
	s.NotContains(strings.ToLower(string(body)), "mem0")
	s.NotContains(string(body), "quota")
	s.assertReaches(s.client, opened.Id)
	s.True(s.mem0.holds(s.customerID(), opened.Id))
}

func (s *MemoryUnavailableSuite) TestTheMemoryStoresRefusalIsLoggedAsAnError() {
	opened := s.client.createSession(textSession(nil))
	s.mem0.down.Store(true)

	s.Require().Equal(http.StatusServiceUnavailable,
		s.client.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil, nil))

	line := s.servedLine(http.MethodDelete, "/v1/agents/sessions/"+opened.Id)
	s.Contains(line, "level=ERROR")
	s.Contains(line, "status=503")
	s.Contains(line, "mem0:")
	s.Contains(line, "429 Too Many Requests")
	s.Contains(line, "monthly memory operations quota exhausted")
}

func (s *MemoryUnavailableSuite) TestTheSameDeleteSucceedsOnceTheMemoryStoreAnswers() {
	opened := s.client.createSession(textSession(nil))
	s.mem0.remember(s.customerID(), opened.Id)
	s.mem0.down.Store(true)
	s.Require().Equal(http.StatusServiceUnavailable,
		s.client.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil, nil))

	s.mem0.down.Store(false)

	s.client.deleteSession(opened.Id)
	s.assertDoesNotReach(s.client, opened.Id)
	s.False(s.mem0.holds(s.customerID(), opened.Id))
}

func (s *MemoryUnavailableSuite) TestTruncatingWhileTheMemoryStoreRefusesIsUnavailable() {
	user := s.utils.uuid()
	s.mem0.down.Store(true)

	status, body := s.serverClient.call(http.MethodDelete, "/v1/agents/users/"+user+"/memories", nil)

	s.Require().Equal(http.StatusServiceUnavailable, status)
	answer := answered(s.T(), body)
	s.Equal("memory_unavailable", answer.Code)
	s.Equal("What was remembered could not be removed right now. Try again later.", answer.Message)
	s.Contains(s.servedLine(http.MethodDelete, "/v1/agents/users/"+user+"/memories"), "quota exhausted")
}

func (s *MemoryUnavailableSuite) TestForgettingASessionWhileTheMemoryStoreRefusesIsUnavailable() {
	opened := s.client.createSession(textSession(nil))
	s.mem0.remember(s.customerID(), opened.Id)
	s.mem0.down.Store(true)

	status, body := s.serverClient.call(http.MethodDelete, "/v1/agents/sessions/"+opened.Id+"/memories", nil)

	s.Require().Equal(http.StatusServiceUnavailable, status)
	answer := answered(s.T(), body)
	s.Equal("memory_unavailable", answer.Code)
	s.Equal("What was remembered could not be removed right now. Try again later.", answer.Message)
	s.NotContains(string(body), "quota")
	s.True(s.mem0.holds(s.customerID(), opened.Id))
}

// servedLine is the one access log line of the request to path.
func (s *MemoryUnavailableSuite) servedLine(method, path string) string {
	var found []string
	for line := range strings.Lines(s.logged.String()) {
		if strings.Contains(line, `msg="served a request"`) && strings.Contains(line, " method="+method+" ") &&
			strings.Contains(line, " path="+path+" ") {
			found = append(found, line)
		}
	}
	s.Require().Len(found, 1, "one access log line for %s %s", method, path)
	return found[0]
}

// flakyMem0 is the part of mem0's API the router deletes through: a list by filter and a
// batch delete by id. Down, it answers every call with a 429, as mem0 does once an account's
// quota is spent.
type flakyMem0 struct {
	server *httptest.Server
	down   atomic.Bool

	mu   sync.Mutex
	kept []keptMem0
}

type keptMem0 struct {
	ID    string
	AppID string
	RunID string
}

func newFlakyMem0(t *testing.T) *flakyMem0 {
	fake := &flakyMem0{}
	fake.server = httptest.NewServer(http.HandlerFunc(fake.serve))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *flakyMem0) remember(appID, runID string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.kept = append(f.kept, keptMem0{ID: appID + "/" + runID, AppID: appID, RunID: runID})
}

func (f *flakyMem0) holds(appID, runID string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.ContainsFunc(f.kept, func(kept keptMem0) bool { return kept.AppID == appID && kept.RunID == runID })
}

func (f *flakyMem0) serve(w http.ResponseWriter, r *http.Request) {
	if f.down.Load() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(quotaSpent))
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.Method + " " + r.URL.Path {
	case "POST /v3/memories/":
		var listed struct {
			Filters map[string]any `json:"filters"`
		}
		if err := json.NewDecoder(r.Body).Decode(&listed); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		results := []map[string]string{}
		for _, kept := range f.kept {
			if kept.AppID == listed.Filters["app_id"] && (listed.Filters["run_id"] == nil || kept.RunID == listed.Filters["run_id"]) {
				results = append(results, map[string]string{"id": kept.ID})
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
	case "DELETE /v1/batch/":
		var batch struct {
			Memories []struct {
				MemoryID string `json:"memory_id"`
			} `json:"memories"`
		}
		if err := json.NewDecoder(r.Body).Decode(&batch); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		for _, deleted := range batch.Memories {
			f.kept = slices.DeleteFunc(f.kept, func(kept keptMem0) bool { return kept.ID == deleted.MemoryID })
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"message": "deleted"})
	case "POST /v3/memories/search/":
		_ = json.NewEncoder(w).Encode(map[string]any{"results": []any{}})
	case "POST /v3/memories/add/":
		_ = json.NewEncoder(w).Encode(map[string]string{"event_id": "queued", "status": "PENDING"})
	default:
		http.NotFound(w, r)
	}
}
