package stream

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestABackendReadsTheEnvironmentForWhateverItWasNotTold(t *testing.T) {
	t.Setenv(URLEnv, "https://acceleration.example.com/")
	t.Setenv(CustomerEnv, "acme")

	resolved, err := Backend{}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if resolved.URL != "https://acceleration.example.com" {
		t.Errorf("the url is %q, want the trailing slash gone", resolved.URL)
	}
	if resolved.CustomerID != "acme" {
		t.Errorf("the customer is %q", resolved.CustomerID)
	}
}

func TestWhatIsPassedInBeatsWhatTheEnvironmentSays(t *testing.T) {
	t.Setenv(URLEnv, "https://acceleration.example.com")
	t.Setenv(CustomerEnv, "acme")

	resolved, err := Backend{URL: "http://localhost:9999", CustomerID: "other"}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if resolved.URL != "http://localhost:9999" || resolved.CustomerID != "other" {
		t.Errorf("resolved to %s as %s", resolved.URL, resolved.CustomerID)
	}
}

func TestABackendNobodyIsBilledForIsRefused(t *testing.T) {
	t.Setenv(URLEnv, "http://localhost:8080")
	t.Setenv(CustomerEnv, "")

	if _, err := (Backend{}).Resolve(); err == nil {
		t.Fatal("a request with no customer would be work nobody pays for")
	}
}

func TestASocketURLFollowsWhetherTheRouterIsEncrypted(t *testing.T) {
	for url, want := range map[string]string{
		"https://acceleration.example.com": "wss://acceleration.example.com/v1/agents/sessions/s1/events",
		"http://localhost:8080":            "ws://localhost:8080/v1/agents/sessions/s1/events",
	} {
		backend := Backend{URL: url}
		if got := backend.SocketURL("/v1/agents/sessions/s1/events"); got != want {
			t.Errorf("%s became %s, want %s", url, got, want)
		}
	}
}

func TestEveryRequestCarriesWhoIsBeingBilled(t *testing.T) {
	seen := make(chan string, 1)
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen <- r.Header.Get(CustomerHeader)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`[]`))
	}))
	defer router.Close()

	client, err := Backend{URL: router.URL, CustomerID: "acme"}.Client()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.ListSkillsWithResponse(t.Context(), nil); err != nil {
		t.Fatal(err)
	}

	if got := <-seen; got != "acme" {
		t.Errorf("the router was told %q", got)
	}
}
