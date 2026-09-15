package stream

import (
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"
)

// admit reads a request the way the authenticating proxy in front of a hosted router does:
// the app is named by a header, the only scheme on offer is a JWT, and the token is signed
// with that app's secret. It returns the user the token speaks for, which is empty for one
// that speaks for the app itself.
//
// It is written out here rather than imported because the proxy lives in another repository
// and this is the contract this SDK has to keep. A change there that this does not follow
// should fail here rather than in production.
func admit(t *testing.T, header http.Header, secret string) (string, error) {
	t.Helper()
	if header.Get(APIKeyHeader) == "" {
		return "", errNoAPIKey
	}
	if header.Get(AuthTypeHeader) != AuthTypeJWT {
		return "", errAuthType
	}
	// The proxy strips base64 padding before parsing and pins HS256, because a token is
	// allowed to name its own algorithm and "none" is one of the names.
	token := strings.ReplaceAll(strings.TrimPrefix(header.Get(AuthorizationHeader), "Bearer "), "=", "")
	claims := jwt.MapClaims{}
	if _, err := jwt.ParseWithClaims(token, claims,
		func(*jwt.Token) (any, error) { return []byte(secret), nil },
		jwt.WithValidMethods([]string{jwt.SigningMethodHS256.Alg()}),
	); err != nil {
		return "", err
	}
	user, _ := claims["user_id"].(string)
	return user, nil
}

var (
	errNoAPIKey = &admissionError{"api_key is required"}
	errAuthType = &admissionError{"stream-auth-type missing or invalid"}
)

type admissionError struct{ message string }

func (e *admissionError) Error() string { return e.message }

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
	t.Setenv(APIKeyEnv, "")
	t.Setenv(APISecretEnv, "")

	if _, err := (Backend{}).Resolve(); err == nil {
		t.Fatal("a request with no customer would be work nobody pays for")
	}
}

func TestACredentialSaysWhoIsCallingWithoutACustomerID(t *testing.T) {
	t.Setenv(URLEnv, "https://acceleration.example.com")
	t.Setenv(CustomerEnv, "")
	t.Setenv(AuthenticateEnv, "true")
	t.Setenv(APIKeyEnv, "ntv9")
	t.Setenv(APISecretEnv, "shh")

	// The proxy works the app out from the credential and strips the customer header, so
	// there is nothing left for a customer id to answer.
	resolved, err := Backend{}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if resolved.APIKey != "ntv9" || resolved.APISecret != "shh" {
		t.Errorf("the credential resolved to %q / %q", resolved.APIKey, resolved.APISecret)
	}
}

func TestACredentialLyingAroundDoesNotTurnAuthenticationOn(t *testing.T) {
	// A Stream key and secret are in the environment for video and chat whether or not this
	// router is reached through the proxy. Sending them anyway is not harmless: a router
	// with nothing in front of it reads stream-auth-type: jwt as an end user's device and
	// refuses it the paths that configure an agent.
	t.Setenv(URLEnv, "http://127.0.0.1:8098")
	t.Setenv(CustomerEnv, "support-local")
	t.Setenv(AuthenticateEnv, "")
	t.Setenv(APIKeyEnv, "ntv9")
	t.Setenv(APISecretEnv, "shh")

	resolved, err := Backend{}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	credentials, err := resolved.Credentials()
	if err != nil {
		t.Fatal(err)
	}
	if got := credentials.Get(CustomerHeader); got != "support-local" {
		t.Errorf("the customer header is %q", got)
	}
	for _, name := range []string{APIKeyHeader, AuthTypeHeader, AuthorizationHeader} {
		if got := credentials.Get(name); got != "" {
			t.Errorf("%s is set to %q against a router that was never asked to authenticate", name, got)
		}
	}
}

func TestAskingToAuthenticateWithNothingToAuthenticateWithIsRefused(t *testing.T) {
	t.Setenv(URLEnv, "https://acceleration.example.com")
	t.Setenv(CustomerEnv, "acme")
	t.Setenv(APIKeyEnv, "")
	t.Setenv(APISecretEnv, "")
	t.Setenv(AuthenticateEnv, "true")

	// Falling back to the customer header would send a request the proxy refuses, and
	// report it as whatever the proxy says rather than as the missing credential it is.
	if _, err := (Backend{}).Resolve(); err != ErrNoCredential {
		t.Errorf("resolved with %v, want a refusal naming the missing credential", err)
	}
}

func TestARequestForNoOneInParticularSpeaksForTheApp(t *testing.T) {
	backend := Backend{Authenticate: true, CustomerID: "acme", APIKey: "ntv9", APISecret: "shh"}
	credentials, err := backend.Credentials()
	if err != nil {
		t.Fatal(err)
	}

	user, err := admit(t, credentials, "shh")
	if err != nil {
		t.Fatalf("the proxy would refuse this: %v", err)
	}
	if user != "" {
		t.Errorf("the token names %q; a backend speaks for the app, which is what leaves the per-user limits out of it", user)
	}

	// The proxy reads the absence of a user; a router verifying the token itself reads
	// Stream's own marking. Both have to agree that this is a backend, or the paths that
	// configure an agent are refused.
	claims := jwt.MapClaims{}
	if _, err = jwt.ParseWithClaims(credentials.Get(AuthorizationHeader), claims,
		func(*jwt.Token) (any, error) { return []byte("shh"), nil }); err != nil {
		t.Fatal(err)
	}
	if server, _ := claims["server"].(bool); !server {
		t.Errorf("the token is not marked server-side: %v", claims)
	}
}

func TestARequestMadeForSomeoneNamesThem(t *testing.T) {
	credentials, err := Backend{Authenticate: true, APIKey: "ntv9", APISecret: "shh", UserID: "alice"}.Credentials()
	if err != nil {
		t.Fatal(err)
	}

	user, err := admit(t, credentials, "shh")
	if err != nil {
		t.Fatalf("the proxy would refuse this: %v", err)
	}
	if user != "alice" {
		t.Errorf("the token names %q, want alice", user)
	}
}

func TestATokenSignedWithTheWrongSecretIsWorthless(t *testing.T) {
	credentials, err := Backend{Authenticate: true, APIKey: "ntv9", APISecret: "shh"}.Credentials()
	if err != nil {
		t.Fatal(err)
	}

	if _, err := admit(t, credentials, "not-the-secret"); err == nil {
		t.Fatal("a token verified against the wrong secret, so it is not signed with the app's")
	}
}

func TestATokenExpiresAndCarriesNoPaddingForTheProxyToStrip(t *testing.T) {
	credentials, err := Backend{Authenticate: true, APIKey: "ntv9", APISecret: "shh"}.Credentials()
	if err != nil {
		t.Fatal(err)
	}
	token := credentials.Get(AuthorizationHeader)
	if strings.Contains(token, "=") {
		t.Errorf("the token carries padding, which the proxy strips before parsing: %q", token)
	}

	claims := jwt.MapClaims{}
	if _, err := jwt.ParseWithClaims(token, claims, func(*jwt.Token) (any, error) { return []byte("shh"), nil }); err != nil {
		t.Fatal(err)
	}
	expiry, err := claims.GetExpirationTime()
	if err != nil || expiry == nil {
		t.Fatalf("the token never expires (%v), so a stolen one works forever", err)
	}
	if left := time.Until(expiry.Time); left <= 0 || left > tokenValidity {
		t.Errorf("the token has %s left, want at most %s", left, tokenValidity)
	}
}

func TestARouterThatTrustsTheCustomerHeaderIsToldNothingExtra(t *testing.T) {
	// Nothing to authenticate with is the local case, and it has to keep working exactly
	// as it did: the customer header alone.
	credentials, err := Backend{CustomerID: "acme"}.Credentials()
	if err != nil {
		t.Fatal(err)
	}
	if got := credentials.Get(CustomerHeader); got != "acme" {
		t.Errorf("the customer header is %q", got)
	}
	for _, name := range []string{APIKeyHeader, AuthTypeHeader, AuthorizationHeader} {
		if got := credentials.Get(name); got != "" {
			t.Errorf("%s is set to %q with nothing to authenticate with", name, got)
		}
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

func TestEveryRequestCarriesTheCredentialTheProxyAsksFor(t *testing.T) {
	t.Setenv(CustomerEnv, "")
	t.Setenv(APIKeyEnv, "")
	t.Setenv(APISecretEnv, "")

	seen := make(chan http.Header, 1)
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[]`))
	}))
	defer proxy.Close()

	client, err := Backend{URL: proxy.URL, Authenticate: true, APIKey: "ntv9", APISecret: "shh"}.Client()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.ListSkillsWithResponse(t.Context(), nil); err != nil {
		t.Fatal(err)
	}

	if _, err := admit(t, <-seen, "shh"); err != nil {
		t.Fatalf("the proxy would refuse what the client sent: %v", err)
	}
}

func TestASocketIsAuthenticatedOnTheHandshake(t *testing.T) {
	// The router reads the identity once, when the upgrade happens, and then trusts the
	// connection for its whole life, so the credential has to be on the handshake.
	seen := make(chan http.Header, 1)
	upgrader := websocket.Upgrader{}
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen <- r.Header.Clone()
		connection, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		_, _, _ = connection.ReadMessage()
	}))
	defer proxy.Close()

	backend := Backend{URL: proxy.URL, Authenticate: true, APIKey: "ntv9", APISecret: "shh", UserID: "alice"}
	credentials, err := backend.Credentials()
	if err != nil {
		t.Fatal(err)
	}
	socket := NewSocket(backend.SocketURL("/v1/dispatch"), credentials, nil, slog.New(slog.DiscardHandler))
	if err := socket.Open(t.Context()); err != nil {
		t.Fatal(err)
	}
	defer socket.Close()

	user, err := admit(t, <-seen, "shh")
	if err != nil {
		t.Fatalf("the proxy would refuse the upgrade: %v", err)
	}
	if user != "alice" {
		t.Errorf("the socket was opened for %q, want alice", user)
	}
}
