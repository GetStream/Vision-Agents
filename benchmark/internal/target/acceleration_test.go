package target

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadPackContract(t *testing.T) {
	root := findTestRoot(t)
	cases := []struct {
		pack  string
		tools int
		first string
	}{
		{"telecom", 8, "verify_account"},
		{"restaurant", 5, "check_availability"},
		{"healthcare", 6, "verify_identity"},
	}
	for _, tc := range cases {
		instructions, tools, err := LoadPackContract(root, tc.pack)
		if err != nil {
			t.Fatal(err)
		}
		if instructions == "" {
			t.Fatalf("%s: empty instructions", tc.pack)
		}
		if len(tools) != tc.tools {
			t.Fatalf("%s: got %d tools", tc.pack, len(tools))
		}
		if tools[0].Name != tc.first {
			t.Fatalf("%s: first tool %s", tc.pack, tools[0].Name)
		}
	}
}

func TestAccelEventsURL(t *testing.T) {
	got, err := AccelEventsURL("http://127.0.0.1:8080", "sess-1")
	if err != nil {
		t.Fatal(err)
	}
	if got != "ws://127.0.0.1:8080/v1/agents/sessions/sess-1/events" {
		t.Fatalf("got %s", got)
	}
	got, err = AccelEventsURL("https://router.example", "abc")
	if err != nil {
		t.Fatal(err)
	}
	if got != "wss://router.example/v1/agents/sessions/abc/events" {
		t.Fatalf("got %s", got)
	}
}

func TestCallWorldTool(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/session/tools/check_outage" {
			t.Fatalf("path %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"outage":false}`))
	}))
	t.Cleanup(srv.Close)

	out, fail := CallWorldTool(t.Context(), srv.URL, "check_outage", "{}")
	if fail != "" {
		t.Fatal(fail)
	}
	if out != `{"outage":false}` {
		t.Fatalf("got %s", out)
	}
}

func findTestRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := wd
	for {
		if _, err := os.Stat(filepath.Join(dir, "scenarios", "restaurant")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("scenarios not found")
		}
		dir = parent
	}
}

func TestLiveKitRejectsUnreachableWorldURL(t *testing.T) {
	root := findTestRoot(t)
	t.Setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
	t.Setenv("LIVEKIT_API_KEY", "key")
	t.Setenv("LIVEKIT_API_SECRET", "secret")

	target := &LiveKit{Root: root, Pack: "healthcare", AgentName: "assistant-164d", WorldURL: "http://127.0.0.1:8090"}
	_, err := target.Prepare(context.Background())
	if err == nil {
		t.Fatal("a loopback world URL is unreachable for a remote worker and must fail before the call")
	}
	if !strings.Contains(err.Error(), "--world-url") {
		t.Fatalf("error does not point at the fix: %v", err)
	}

	target = &LiveKit{Root: root, Pack: "healthcare", AgentName: "assistant-164d", WorldURL: "http://10.0.0.4:8090"}
	stop, err := target.Prepare(context.Background())
	if err != nil {
		t.Fatalf("routable world URL rejected: %v", err)
	}
	stop()
}

func TestLiveKitSpawnIgnoresAmbientAgentName(t *testing.T) {
	t.Setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
	t.Setenv("LIVEKIT_API_KEY", "key")
	t.Setenv("LIVEKIT_API_SECRET", "secret")
	t.Setenv("LIVEKIT_AGENT_NAME", "assistant-164d")

	target := &LiveKit{Root: t.TempDir(), Pack: "healthcare", WorldURL: "http://127.0.0.1:8090", Spawn: true,
		Instructions: "be brief", Tools: []AccelTool{{Name: "verify_identity"}}}
	if _, err := target.Prepare(context.Background()); err == nil {
		t.Fatal("spawning from a directory without the worker should fail")
	}
	if target.AgentName != LiveKitWorkerAgentName {
		t.Fatalf("spawned worker dispatched to %q instead of the reference worker", target.AgentName)
	}
}
