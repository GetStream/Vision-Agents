package edge

import (
	"net/url"
	"strings"
	"testing"
)

// credentials are a well-formed key and secret. Nothing here reaches Stream: creating a
// client and minting a token are both local.
//
// The demo deployment is cleared as well, since a developer with EXAMPLE_BASE_URL set would
// otherwise be running a different test from CI.
func credentials(t *testing.T) *Edge {
	t.Helper()
	t.Setenv(MonitorBaseURLEnv, "")

	transport, err := New(Options{APIKey: "key123", APISecret: "secret123"})
	if err != nil {
		t.Fatal(err)
	}
	return transport
}

func TestCredentialsAreRequiredBeforeAnythingIsAttempted(t *testing.T) {
	t.Setenv(APIKeyEnv, "")
	t.Setenv(APISecretEnv, "")

	if _, err := New(Options{}); err == nil {
		t.Fatal("a call cannot be created without them")
	}
}

func TestTheMonitorLinkCarriesTheCallAndATokenToJoinItWith(t *testing.T) {
	transport := credentials(t)

	link, err := transport.MonitorURL(Call{ID: "call-1", Type: "default"}, User{ID: "watcher"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(link, DefaultMonitorBaseURL+"/join/call-1?") {
		t.Fatalf("the link is %s", link)
	}

	query, err := url.Parse(link)
	if err != nil {
		t.Fatal(err)
	}
	values := query.Query()
	if values.Get("api_key") != "key123" {
		t.Errorf("the browser is told the key is %q", values.Get("api_key"))
	}
	if values.Get("token") == "" {
		t.Error("without a token the browser cannot join")
	}
	if values.Get("skip_lobby") != "true" {
		t.Error("a monitor should land in the call rather than in the lobby")
	}
	if values.Get("user_name") != "watcher" {
		t.Errorf("the watcher appears as %q, want their id when they have no name", values.Get("user_name"))
	}
}

func TestTheMonitorLinkPointsAtWhicheverDemoIsDeployed(t *testing.T) {
	t.Setenv(MonitorBaseURLEnv, "https://demo.example.com/")

	transport, err := New(Options{APIKey: "key123", APISecret: "secret123"})
	if err != nil {
		t.Fatal(err)
	}

	link, err := transport.MonitorURL(Call{ID: "call-1"}, User{ID: "watcher", Name: "Ada"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(link, "https://demo.example.com/join/call-1?") {
		t.Errorf("the link is %s", link)
	}
}

func TestThereIsNoLinkToACallThatDoesNotExist(t *testing.T) {
	transport := credentials(t)

	if _, err := transport.MonitorURL(Call{}, User{ID: "watcher"}); err == nil {
		t.Fatal("a conversation held in writing has no call to watch")
	}
	if _, err := transport.MonitorURL(Call{ID: "call-1"}, User{}); err == nil {
		t.Fatal("somebody has to be watching")
	}
}
