//go:build integration

package managed

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/research"
)

// TestLiveDaytona creates real billable resources and removes them on completion.
// Run explicitly with -tags integration and the operator profile path.
func TestLiveDaytona(t *testing.T) {
	if os.Getenv("DAYTONA_API_KEY") == "" {
		t.Skip("unblock: configure DAYTONA_API_KEY for real Daytona acceptance")
	}
	path := os.Getenv("RESEARCH_TEST_PROFILE")
	if path == "" {
		t.Fatal("RESEARCH_TEST_PROFILE is required")
	}
	t.Setenv("RESEARCH_DEPLOYMENT_ID", fmt.Sprintf("stream-support-integration-%d", time.Now().UnixNano()))
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Minute)
	defer cancel()
	m, err := Start(ctx, path)
	if m != nil {
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Profiles) != 1 {
		t.Fatal("expected exactly one support workspace")
	}
	for _, w := range m.Profiles {
		originalID := w.box.ID
		t.Logf("startup: %v", w.Timings)
		for _, r := range w.Profile.Repositories {
			if r.Revision == "" {
				t.Fatal("revision missing")
			}
			t.Logf("%s %s", r.ID, r.Revision)
			q := "Name one public client type or function defined in this SDK and cite its declaration."
			if r.ID == "stream-chat-react" {
				q = "What does useChatContext return?"
			} else if r.ID == "stream-chat-swiftui" {
				q = "Where is ChatChannelListView defined?"
			}
			result := w.Research(ctx, research.Request{Product: r.Product, SDK: r.SDK, RepositoryIDs: []string{r.ID}, Question: q}, func(research.Progress) {})
			if result.Status != "answered" || len(result.Citations) == 0 {
				t.Errorf("%s failed: %+v", r.ID, result)
			}
			t.Logf("%s research_ms=%d citations=%d", r.ID, result.ElapsedMS, len(result.Citations))
			if w.box.ID != originalID {
				t.Fatal("workspace was replaced between requests")
			}
		}
		if _, err = m.Find(w.Profile.Name, "unauthorized", w.Profile.AgentID); err == nil {
			t.Fatal("cross-customer profile accepted")
		}
	}
}

// TestLiveDaytonaResume exercises the inactivity fallback without waiting five minutes.
func TestLiveDaytonaResume(t *testing.T) {
	if os.Getenv("DAYTONA_API_KEY") == "" {
		t.Skip("unblock: configure DAYTONA_API_KEY")
	}
	path := os.Getenv("RESEARCH_TEST_PROFILE")
	if path == "" {
		t.Fatal("RESEARCH_TEST_PROFILE is required")
	}
	t.Setenv("RESEARCH_DEPLOYMENT_ID", fmt.Sprintf("stream-support-resume-%d", time.Now().UnixNano()))
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer cancel()
	m, err := Start(ctx, path)
	if m != nil {
		t.Cleanup(func() {
			if e := m.Close(); e != nil {
				t.Error(e)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range m.Profiles {
		// Exclude the heartbeat to reproduce a stopped VM at request admission.
		w.exclusive <- struct{}{}
		id := w.box.ID
		revisions := fmt.Sprint(w.Profile.Repositories)
		if err := w.box.Stop(ctx); err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		if err := w.ensureReady(ctx, func(p research.Progress) { t.Log(p.Phase) }); err != nil {
			t.Fatal(err)
		}
		if w.box.ID != id || fmt.Sprint(w.Profile.Repositories) != revisions {
			t.Fatal("recovery replaced the VM or revisions")
		}
		for _, repo := range w.Profile.Repositories {
			r, err := w.command(ctx, "test -d "+quote(research.Root+"/"+repo.ID))
			if err != nil {
				t.Fatal(r, err)
			}
		}
		if !w.healthy(ctx) {
			t.Fatal("resumed worker is not healthy")
		}
		t.Logf("same VM resumed in %s; pinned repositories preserved", time.Since(started).Round(time.Millisecond))
		<-w.exclusive
		result := w.Research(ctx, research.Request{Product: "chat", SDK: "react", RepositoryIDs: []string{"stream-chat-react"}, Question: "What does useChatContext return?"}, func(research.Progress) {})
		t.Logf("resumed research: status=%s code=%s elapsed_ms=%d citations=%d", result.Status, result.Code, result.ElapsedMS, len(result.Citations))
		if result.Code == "worker_unavailable" || result.Code == "worker_disconnected" || result.Code == "worker_endpoint_unavailable" {
			t.Fatal(result.Code)
		}
	}
}
