package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// Catches a Gmail list or proxy send that does not name the user it is for: the router would
// then list every user's connections, or refuse the user's own connection.
func TestGmailCallsNameTheUser(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]string{}
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen[r.Method+" "+r.URL.Path] = r.Header.Get("X-Stream-User-Id")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "/proxy/") {
			_, _ = w.Write([]byte(`{"id":"m1"}`))
			return
		}
		_, _ = w.Write([]byte(`{"items":[{"id":"c1","status":"connected"}],"has_more":false}`))
	}))
	defer router.Close()
	t.Setenv("STREAM_ACCELERATION_URL", router.URL)
	t.Setenv("STREAM_ACCELERATION_CUSTOMER_ID", "acme")

	if _, err := gmailConnections(t.Context(), "anya"); err != nil {
		t.Fatal(err)
	}
	if _, err := proxy(t.Context(), "anya", "c1", "gmail/v1/users/me/messages/send", map[string]string{"raw": "x"}); err != nil {
		t.Fatal(err)
	}
	for _, request := range []string{"GET /v1/agents/connections", "POST /v1/agents/connections/c1/proxy/gmail/v1/users/me/messages/send"} {
		if got := seen[request]; got != "anya" {
			t.Errorf("%s named %q, want anya", request, got)
		}
	}
}
