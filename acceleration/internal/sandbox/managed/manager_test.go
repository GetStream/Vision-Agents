package managed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/GetStream/Vision-Agents/acceleration/internal/research"
)

func testWorkspace(t *testing.T, h http.HandlerFunc) *Workspace {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h(w, r)
	}))
	t.Cleanup(server.Close)
	return &Workspace{Profile: research.Profile{Name: "support", CustomerID: "acme", AgentID: "bot", Repositories: []research.Repository{{ID: "react", Product: "chat", SDK: "react"}}}, link: &types.PreviewLink{URL: server.URL, Token: "preview"}, token: "worker", slots: make(chan struct{}, 9), exclusive: make(chan struct{}, 1), life: context.Background()}
}
func query() research.Request {
	return research.Request{Product: "chat", SDK: "react", Question: "Where is the message view?", RepositoryIDs: []string{"react"}}
}
func TestProfileOwnership(t *testing.T) {
	w := testWorkspace(t, func(http.ResponseWriter, *http.Request) {})
	m := &Manager{Profiles: map[string]*Workspace{"support": w}}
	for _, owner := range [][2]string{{"other", "bot"}, {"acme", "other"}, {"", ""}} {
		if _, err := m.Find("support", owner[0], owner[1]); err == nil {
			t.Fatal(owner)
		}
	}
	if found, err := m.Find("support", "acme", "bot"); err != nil || found != w {
		t.Fatal(err)
	}
}
func TestQueueCancellationAndReuse(t *testing.T) {
	entered := make(chan struct{}, 16)
	release := make(chan struct{})
	var mu sync.Mutex
	active, maxActive := 0, 0
	w := testWorkspace(t, func(out http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer worker" || r.Header.Get("X-Daytona-Preview-Token") != "preview" {
			t.Error("missing authentication")
		}
		mu.Lock()
		active++
		maxActive = max(maxActive, active)
		mu.Unlock()
		defer func() { mu.Lock(); active--; mu.Unlock() }()
		entered <- struct{}{}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		_ = json.NewEncoder(out).Encode(research.Frame{Result: &research.Result{Status: "answered"}})
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	result := make(chan research.Result, 1)
	go func() { result <- w.Research(ctx, query(), func(research.Progress) {}) }()
	<-entered
	queued, cancelQueued := context.WithCancel(ctx)
	ready := make(chan struct{}, 8)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := w.Research(queued, query(), func(p research.Progress) {
				if p.Phase == "queued" {
					ready <- struct{}{}
				}
			})
			if got.Code != "queue_cancelled" {
				t.Error(got)
			}
		}()
	}
	for range 8 {
		<-ready
	}
	got := w.Research(ctx, query(), func(research.Progress) {})
	if got.Code != "queue_full" {
		t.Fatal(got)
	}
	cancelQueued()
	wg.Wait()
	close(release)
	if (<-result).Status != "answered" {
		t.Fatal("first failed")
	}
	if got = w.Research(ctx, query(), func(research.Progress) {}); got.Status != "answered" {
		t.Fatal(got)
	}
	mu.Lock()
	defer mu.Unlock()
	if maxActive != 1 {
		t.Fatal("research overlapped", maxActive)
	}
}
func TestActiveCancellationAndShutdown(t *testing.T) {
	entered := make(chan struct{})
	ended := make(chan struct{})
	w := testWorkspace(t, func(out http.ResponseWriter, r *http.Request) {
		out.Header().Set("Content-Type", "application/x-ndjson")
		_ = json.NewEncoder(out).Encode(research.Frame{Progress: &research.Progress{Phase: "researching"}})
		out.(http.Flusher).Flush()
		close(entered)
		<-r.Context().Done()
		close(ended)
	})
	life, stop := context.WithCancel(context.Background())
	w.life = life
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	result := make(chan research.Result)
	go func() { result <- w.Research(ctx, query(), func(research.Progress) {}) }()
	<-entered
	stop()
	select {
	case <-ended:
	case <-ctx.Done():
		t.Fatal("worker not cancelled")
	}
	select {
	case got := <-result:
		if got.Status != "research_failed" {
			t.Fatal(got)
		}
	case <-ctx.Done():
		t.Fatal("research hung")
	}
}
