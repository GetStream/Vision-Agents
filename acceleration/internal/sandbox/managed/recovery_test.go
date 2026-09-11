package managed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/research"
	"github.com/daytona/clients/sdk-go/pkg/daytona"
	"github.com/daytona/clients/sdk-go/pkg/types"
	"github.com/stretchr/testify/require"
)

// Exercise the official SDK against a stateful Daytona/worker HTTP peer.
func TestStoppedWorkspaceRecoversInPlaceAndRefreshesPreview(t *testing.T) {
	var mu sync.Mutex
	state, token := "stopped", "fresh-preview"
	ready, failStart := false, true
	starts, creates, launches := 0, 0, 0
	var server *httptest.Server
	sandbox := func() map[string]any {
		return map[string]any{"id": "kept", "organizationId": "org", "name": "support", "user": "root", "env": map[string]string{}, "public": false, "networkBlockAll": false, "target": "us", "cpu": 2, "gpu": 0, "memory": 4, "disk": 10, "state": state, "toolboxProxyUrl": server.URL + "/toolbox", "labels": map[string]string{}}
	}
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(r.URL.Path, "socket.io"):
			w.WriteHeader(404)
		case r.URL.Path == "/api/sandbox/kept" && r.Method == "GET":
			_ = json.NewEncoder(w).Encode(sandbox())
		case r.URL.Path == "/api/sandbox":
			creates++
			w.WriteHeader(500)
		case r.URL.Path == "/api/sandbox/kept/start":
			starts++
			if failStart {
				w.WriteHeader(503)
				_, _ = w.Write([]byte(`{"message":"temporarily unavailable"}`))
				return
			}
			state = "started"
			_ = json.NewEncoder(w).Encode(sandbox())
		case strings.HasSuffix(r.URL.Path, "/preview-url"):
			_ = json.NewEncoder(w).Encode(map[string]any{"sandboxId": "kept", "url": server.URL, "token": token})
		case strings.HasSuffix(r.URL.Path, "/process/session/stream-research-worker") && r.Method == "DELETE":
			w.WriteHeader(204)
		case strings.HasSuffix(r.URL.Path, "/process/session") && r.Method == "POST":
			_, _ = w.Write([]byte(`{}`))
		case strings.HasSuffix(r.URL.Path, "/process/session/stream-research-worker/exec"):
			launches++
			ready = true
			_, _ = w.Write([]byte(`{"cmdId":"worker"}`))
		case r.URL.Path == "/health":
			if state != "started" || !ready {
				w.WriteHeader(503)
			} else if r.Header.Get("X-Daytona-Preview-Token") != token {
				w.WriteHeader(401)
			} else {
				w.WriteHeader(204)
			}
		case r.URL.Path == "/research":
			require.Equal(t, "Bearer worker", r.Header.Get("Authorization"))
			require.Equal(t, token, r.Header.Get("X-Daytona-Preview-Token"))
			if !ready {
				w.WriteHeader(503)
				return
			}
			_ = json.NewEncoder(w).Encode(research.Frame{Result: &research.Result{Status: "answered"}})
		case strings.HasSuffix(r.URL.Path, "/process/execute"):
			_, _ = w.Write([]byte(`{"exitCode":0,"result":""}`))
		default:
			t.Errorf("unexpected SDK request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	client, err := daytona.NewClientWithConfig(&types.DaytonaConfig{APIKey: "test", APIUrl: server.URL + "/api"})
	require.NoError(t, err)
	defer client.Close(context.Background())
	box, err := client.Get(t.Context(), "kept")
	require.NoError(t, err)
	w := &Workspace{client: client, box: box, Profile: research.Profile{Name: "support", Repositories: []research.Repository{{ID: "react", Product: "chat", SDK: "react", Revision: strings.Repeat("a", 40)}}}, token: "worker", link: &types.PreviewLink{URL: server.URL, Token: "expired"}, life: context.Background(), slots: make(chan struct{}, 9), exclusive: make(chan struct{}, 1)}
	first := w.Research(t.Context(), query(), func(research.Progress) {})
	require.Equal(t, "workspace_resume_failed", first.Code)
	require.Equal(t, "kept", w.box.ID, "failed resume must retain resource identity")
	mu.Lock()
	failStart = false
	mu.Unlock()
	var phases []string
	result := w.Research(t.Context(), query(), func(p research.Progress) { phases = append(phases, p.Phase) })
	require.Equal(t, "answered", result.Status, "%+v", result)
	require.Contains(t, phases, "recovering_workspace")
	require.Contains(t, phases, "starting_worker")
	require.Equal(t, strings.Repeat("a", 40), w.Profile.Repositories[0].Revision)
	require.Equal(t, "kept", w.box.ID)
	// A warm request reuses the recovered process; heartbeat repairs another lost worker.
	require.Equal(t, "answered", w.Research(t.Context(), query(), func(research.Progress) {}).Status)
	mu.Lock()
	require.Equal(t, 2, starts)
	require.Zero(t, creates)
	require.Equal(t, 1, launches)
	ready = false
	mu.Unlock()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	w.heartbeat(ctx)
	require.Equal(t, "answered", w.Research(t.Context(), query(), func(research.Progress) {}).Status)
	mu.Lock()
	require.Equal(t, 2, launches)
	require.Equal(t, 2, starts)
	mu.Unlock()
}
