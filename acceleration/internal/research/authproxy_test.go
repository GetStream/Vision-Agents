package research

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The credential must reach Cursor's key exchange and nothing else, because anything
// that can talk to the loopback proxy is by definition inside the research sandbox.
func TestTheCredentialIsAttachedOnlyToTheKeyExchange(t *testing.T) {
	const credential = "real-cursor-credential"
	var seen string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		_, _ = w.Write([]byte("upstream ok"))
	}))
	defer upstream.Close()
	cursorAPI = upstream.URL
	defer func() { cursorAPI = "https://api2.cursor.sh" }()

	proxy, err := StartAuthProxy(credential)
	if err != nil {
		t.Fatal(err)
	}
	defer proxy.Close()

	for _, c := range []struct {
		name, path, offered, want string
	}{
		{"the exchange presenting the placeholder is upgraded", authExchangePath, "Bearer " + CredentialPlaceholder, "Bearer " + credential},
		{"an exchange presenting anything else is not", authExchangePath, "Bearer guessed-token", "Bearer guessed-token"},
		{"the placeholder is not upgraded on other routes", "/aiserver.v1.AiService/AvailableModels", "Bearer " + CredentialPlaceholder, "Bearer " + CredentialPlaceholder},
		{"an exchanged session token passes through", "/aiserver.v1.AiService/AvailableModels", "Bearer session-token", "Bearer session-token"},
		{"an unauthenticated request stays unauthenticated", authExchangePath, "", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			seen = ""
			request, err := http.NewRequest("POST", proxy.Endpoint()+c.path, strings.NewReader("{}"))
			if err != nil {
				t.Fatal(err)
			}
			if c.offered != "" {
				request.Header.Set("Authorization", c.offered)
			}
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			body, _ := io.ReadAll(response.Body)
			_ = response.Body.Close()
			if seen != c.want {
				t.Fatalf("upstream saw %q, want %q", seen, c.want)
			}
			if strings.Contains(string(body), credential) {
				t.Fatal("the credential was echoed back into the sandbox")
			}
		})
	}
}

func TestTheProxyIsReachableOnlyFromTheSandboxItself(t *testing.T) {
	proxy, err := StartAuthProxy("credential")
	if err != nil {
		t.Fatal(err)
	}
	defer proxy.Close()
	if !strings.HasPrefix(proxy.Endpoint(), "http://127.0.0.1:") {
		t.Fatalf("endpoint %q is not loopback-only", proxy.Endpoint())
	}
}

func TestAWorkerWithoutACredentialDoesNotStart(t *testing.T) {
	if _, err := StartAuthProxy(""); err == nil {
		t.Fatal("a proxy with no credential would send Cursor an empty bearer")
	}
}
