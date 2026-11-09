package githubapp_test

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// The ids the fake GitHub knows. Synthetic: no app or installation has them.
const (
	appID          = "4242"
	installationID = "1337"
)

// tokenLifetime is an installation token's: «The installation access token will expire after
// 1 hour» (docs.github.com, «Generating an installation access token for a GitHub App»).
const tokenLifetime = time.Hour

// github is a local GitHub that answers the requests the scheme sends as GitHub's REST API
// documents them: it mints installation tokens for an app JWT it verifies with the app's
// public key, takes them as bearer tokens until they expire or are revoked, and refuses the
// rest with GitHub's status and a message body.
type github struct {
	*httptest.Server
	clock *clock

	mu sync.Mutex
	// key is the public key the app JWT must verify with; rotate replaces it.
	key    *rsa.PublicKey
	tokens map[string]time.Time
	mints  int
	// minted is the body of the last mint, and claims the last app JWT's claims.
	minted []byte
	claims map[string]any
	// limited makes every mint a 403 primary rate limit, secondary one a 403 secondary rate
	// limit with neither header, and suspended a 403 for a suspended installation.
	limited, secondary, suspended bool
}

func newGitHub(t *testing.T, key *rsa.PrivateKey, clock *clock) *github {
	g := &github{clock: clock, key: &key.PublicKey, tokens: map[string]time.Time{}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /app/installations/{id}/access_tokens", g.mint)
	mux.HandleFunc("GET /installation/repositories", g.withToken(func(w http.ResponseWriter, _ string) {
		answer(w, http.StatusOK, map[string]any{"total_count": 1, "repositories": []any{map[string]any{"name": "hello"}}})
	}))
	mux.HandleFunc("DELETE /installation/token", g.withToken(func(w http.ResponseWriter, token string) {
		g.mu.Lock()
		delete(g.tokens, token)
		g.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	}))
	mux.HandleFunc("GET /repos/octo/hello", g.withToken(func(w http.ResponseWriter, _ string) {
		answer(w, http.StatusOK, map[string]any{"full_name": "octo/hello"})
	}))
	g.Server = httptest.NewTLSServer(mux)
	t.Cleanup(g.Close)
	return g
}

func (g *github) mint(w http.ResponseWriter, r *http.Request) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.limited {
		w.Header().Set("X-RateLimit-Remaining", "0")
		answer(w, http.StatusForbidden, map[string]any{"message": "API rate limit exceeded"})
		return
	}
	if g.secondary {
		answer(w, http.StatusForbidden, map[string]any{"message": "You have exceeded a secondary rate limit. Please wait a few minutes before you try again."})
		return
	}
	if g.suspended {
		answer(w, http.StatusForbidden, map[string]any{"message": "This installation has been suspended"})
		return
	}
	jwt, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	claims, ok := g.verify(jwt)
	if !ok {
		// GitHub echoes nothing of the JWT; this one does, so a test can see it redacted.
		answer(w, http.StatusUnauthorized, map[string]any{"message": "A JSON web token could not be decoded: " + jwt})
		return
	}
	g.claims = claims
	if r.PathValue("id") != installationID {
		answer(w, http.StatusNotFound, map[string]any{"message": "Not Found"})
		return
	}
	g.minted, _ = io.ReadAll(r.Body)
	var narrowing struct {
		Repositories []string `json:"repositories"`
	}
	_ = json.Unmarshal(g.minted, &narrowing)
	for _, name := range narrowing.Repositories {
		if name != "hello" {
			answer(w, http.StatusUnprocessableEntity, map[string]any{"message": "There is at least one repository that does not exist or is not accessible to the parent installation."})
			return
		}
	}
	g.mints++
	random := make([]byte, 16)
	_, _ = rand.Read(random)
	token := "ghs_" + hex.EncodeToString(random)
	expires := g.clock.Now().Add(tokenLifetime).Truncate(time.Second)
	g.tokens[token] = expires
	answer(w, http.StatusCreated, map[string]any{"token": token, "expires_at": expires.UTC().Format(time.RFC3339)})
}

// verify checks an app JWT as GitHub documents it: RS256, signed with the app's key, iss the
// app, iat not in the future and exp at most 10 minutes out.
func (g *github) verify(jwt string) (map[string]any, bool) {
	parts := strings.Split(jwt, ".")
	if len(parts) != 3 {
		return nil, false
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, false
	}
	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	if rsa.VerifyPKCS1v15(g.key, crypto.SHA256, digest[:], signature) != nil {
		return nil, false
	}
	var header map[string]any
	var claims map[string]any
	if decode(parts[0], &header) != nil || decode(parts[1], &claims) != nil || header["alg"] != "RS256" {
		return nil, false
	}
	now := g.clock.Now().Unix()
	iat, _ := claims["iat"].(float64)
	exp, _ := claims["exp"].(float64)
	if claims["iss"] != appID || int64(iat) > now || int64(exp) <= now || int64(exp) > now+600 {
		return nil, false
	}
	return claims, true
}

// withToken answers 401 Bad credentials, GitHub's answer to a token it does not take, unless
// the request carries a token this GitHub minted that has not expired or been revoked.
func (g *github) withToken(next func(w http.ResponseWriter, token string)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		g.mu.Lock()
		expires, found := g.tokens[token]
		g.mu.Unlock()
		if !found || !g.clock.Now().Before(expires) {
			answer(w, http.StatusUnauthorized, map[string]any{"message": "Bad credentials"})
			return
		}
		next(w, token)
	}
}

func (g *github) rotate(key *rsa.PrivateKey) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.key = &key.PublicKey
}

func (g *github) limit() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.limited = true
}

func (g *github) limitSecondary() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.secondary = true
}

func (g *github) suspend() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.suspended = true
}

func (g *github) mintCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.mints
}

func (g *github) lastMint() ([]byte, map[string]any) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.minted, g.claims
}

func answer(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func decode(segment string, into any) error {
	raw, err := base64.RawURLEncoding.DecodeString(segment)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, into)
}

// newKey is a throwaway app key, PEM-encoded as GitHub hands one out (PKCS#1). 2048 bits:
// RFC 7518 section 3.3 requires «2048 bits or larger» for RS256.
func newKey(t *testing.T) (*rsa.PrivateKey, string) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	return key, string(pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)}))
}

type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// x509PKCS8 is key as a PKCS#8 PEM, the form openssl converts GitHub's key to.
func x509PKCS8(key *rsa.PrivateKey) (string, error) {
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return "", err
	}
	return string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})), nil
}
