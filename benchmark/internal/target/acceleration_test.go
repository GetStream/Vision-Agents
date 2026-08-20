package target

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
		{"healthcare", 5, "verify_identity"},
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
