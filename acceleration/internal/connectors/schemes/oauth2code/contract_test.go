package oauth2code_test

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core/contracttest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// TestOAuth2CodeSchemeContract runs the scheme contract against the fake provider: the core
// Slack fixture pinned at the fake's authorize, token and revoke endpoints, the fake's
// preregistered client as the operator's, and a clock the contract moves with the fake's.
func TestOAuth2CodeSchemeContract(t *testing.T) {
	suite.Run(t, &contracttest.SchemeContract{New: func(t *testing.T, logger *slog.Logger) contracttest.Subject {
		srv := fakeprovider.New(t)
		clock := &clock{now: time.Now()}
		scheme, err := oauth2code.New(oauth2code.Config{
			HTTP: srv.Client(),
			Clients: func(_ context.Context, _ core.ConnectionRef, _ core.ResolvedManifest, source core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
				return oauth2code.Client{ID: srv.ClientID, Secret: srv.ClientSecret}, source == core.ClientOperator, nil
			},
			Now:            clock.Now,
			PublicEndpoint: loopbackOrPublic,
			Logger:         logger,
		})
		if err != nil {
			t.Fatal(err)
		}
		return contracttest.Subject{
			Scheme:      scheme,
			Manifest:    contractManifest(t, srv),
			RedirectURI: fakeprovider.RedirectURI,
			Consent: func(authorizeURL string) (url.Values, error) {
				callback, err := srv.Consent(authorizeURL)
				if err != nil {
					return nil, err
				}
				return callback.Query(), nil
			},
			Transport: srv.Client().Transport,
			Call: func() *http.Request {
				request, _ := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathMCP,
					strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`))
				return request
			},
			Secrets: func(stored core.StoredCredentials) []string {
				var payload struct {
					AccessToken  string `json:"access_token"`
					RefreshToken string `json:"refresh_token"`
				}
				_ = json.Unmarshal(stored.Payload, &payload)
				return []string{payload.AccessToken, payload.RefreshToken, srv.ClientSecret}
			},
			Expire: func() {
				clock.Advance(fakeprovider.AccessTTL)
				srv.Advance(fakeprovider.AccessTTL)
			},
			Renewals: srv.Refreshes,
		}
	}})
}

// contractManifest is the core Slack fixture resolved for oauth2_code, with its endpoints
// pointed at the fake, revoke among them, and no capture or identity rule the fake's plain
// token response would not satisfy.
func contractManifest(t *testing.T, srv *fakeprovider.Server) core.ResolvedManifest {
	raw, err := os.ReadFile("../../core/testdata/manifests/slack.yaml")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := core.ParseManifest(raw)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := manifest.Resolve(oauth2code.Name, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	resolved.Endpoints = map[string]string{
		"authorize": srv.URL + fakeprovider.PathAuthorize,
		"token":     srv.URL + fakeprovider.PathToken,
		"revoke":    srv.URL + fakeprovider.PathRevoke,
	}
	resolved.Capture, resolved.Identity = nil, nil
	resolved.Client.AuthMethod = ""
	resolved.Scopes = core.ScopePolicy{List: []string{"files:read", "files:write"}}
	resolved.Refresh = core.RefreshPolicy{}
	return resolved
}

// clock is a time the contract moves, safe to read from the concurrent resolves.
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
