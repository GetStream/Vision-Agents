//go:build integration

package resolver_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2cc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// manifest is a customer's connector at the fake provider: its authorize and token endpoints
// pinned, the fake's preregistered client as the operator's, no refresh policy of its own, so
// oauth2code renews a minute before expiry and has no grace retry.
const manifest = `
id: custom_acme
revision: 1
name: Acme
category: Testing
endpoints:
  authorize: %[1]s/authorize
  token: %[1]s/token
  mcp: %[1]s/mcp
schemes: [oauth2_code]
client:
  registration: [operator]
scopes:
  list: [read]
sources:
  - kind: mcp
    endpoint: mcp
`

// customer is the tenant every connection is made for.
const customer = "acme-app"

// fixture is one database, one fake provider and one clock, which several routers share.
type fixture struct {
	tb     testing.TB
	ctx    context.Context
	dsn    string
	db     *store.Store
	sealer *auth.Sealer
	srv    *fakeprovider.Server
	clock  *clock
	// revision is the definition revision connections are made from.
	revision int
	// clientRemoved makes the operator's client lookup find nothing, as after its record
	// or variables are gone.
	clientRemoved atomic.Bool
}

// database is the resolver's own test database, emptied and migrated, as the store suite
// does with its own.
func database(tb testing.TB) (string, *store.Store) {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		tb.Skip("ROUTER_POSTGRES_DSN not set")
	}
	ctx := context.Background()
	dsn = testenv.Database(dsn, "resolver")
	db, err := store.Open(dsn)
	require.NoError(tb, err)
	require.NoError(tb, db.Ping(ctx))
	var name string
	require.NoError(tb, db.DB().QueryRowContext(ctx, "SELECT current_database()").Scan(&name))
	require.True(tb, strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	_, err = db.DB().ExecContext(ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	require.NoError(tb, err)
	require.NoError(tb, db.Migrate(ctx))
	return dsn, db
}

// newFixture empties the connector tables of db and points a definition at a new fake provider.
func newFixture(tb testing.TB, dsn string, db *store.Store) *fixture {
	ctx := context.Background()
	_, err := db.DB().ExecContext(ctx, "TRUNCATE connector_connections, connector_authorization_attempts, connector_definitions CASCADE")
	require.NoError(tb, err)
	sealer, err := auth.NewSealerWithKeyring(1, map[int]string{1: "resolver test key"})
	require.NoError(tb, err)
	srv := fakeprovider.New(tb)
	parsed, err := core.ParseManifest([]byte(fmt.Sprintf(manifest, srv.URL)))
	require.NoError(tb, err)
	definition, err := db.CreateConnectorDefinition(ctx, customer, parsed)
	require.NoError(tb, err)
	return &fixture{tb: tb, ctx: ctx, dsn: dsn, db: db, sealer: sealer, srv: srv,
		clock: &clock{now: time.Now()}, revision: definition.Revision}
}

// router is another router: a pool of its own, connected before it is used, and a resolver
// whose scheme reaches the fake through client.
func (f *fixture) router(client *http.Client) *resolver.Resolver {
	db, err := store.Open(f.dsn)
	require.NoError(f.tb, err)
	f.tb.Cleanup(func() { _ = db.Close() })
	require.NoError(f.tb, db.Ping(f.ctx))
	credentials, err := pgsealed.New(db, f.sealer)
	require.NoError(f.tb, err)
	r, err := resolver.New(resolver.Config{Store: db, Credentials: credentials,
		Schemes: map[string]core.Scheme{oauth2code.Name: f.scheme(client), oauth2cc.Name: f.clientCredentials(client)}, Now: f.clock.Now})
	require.NoError(f.tb, err)
	return r
}

// scheme is oauth2_code on the fixture's clock, with the fake's client as the operator's.
func (f *fixture) scheme(client *http.Client) *oauth2code.Scheme {
	scheme, err := oauth2code.New(oauth2code.Config{
		HTTP:           client,
		PublicEndpoint: loopbackOrPublic,
		Now:            f.clock.Now,
		Clients: func(_ context.Context, _ core.ConnectionRef, _ core.ResolvedManifest, method core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
			if method != core.ClientOperator || f.clientRemoved.Load() {
				return oauth2code.Client{}, false, nil
			}
			return oauth2code.Client{ID: f.srv.ClientID, Secret: f.srv.ClientSecret}, true, nil
		},
	})
	require.NoError(f.tb, err)
	return scheme
}

// clientCredentials is oauth2_client_credentials on the fixture's clock.
func (f *fixture) clientCredentials(client *http.Client) *oauth2cc.Scheme {
	scheme, err := oauth2cc.New(oauth2cc.Config{HTTP: client, PublicEndpoint: loopbackOrPublic, Now: f.clock.Now})
	require.NoError(f.tb, err)
	return scheme
}

// pending is a new connection no consent has finished for.
func (f *fixture) pending() core.ConnectionRef {
	connection := &store.ConnectorConnection{
		CustomerID: customer, ConnectorID: "custom_acme", DefinitionRevision: f.revision,
		OwnerType: store.OwnerApp, AuthScheme: oauth2code.Name,
	}
	registry := core.Registry{Schemes: map[string]core.Scheme{oauth2code.Name: f.scheme(f.srv.Client())}}
	require.NoError(f.tb, f.db.CreateConnectorConnection(f.ctx, registry, connection))
	return core.ConnectionRef{CustomerID: customer, ConnectionID: connection.ID}
}

// connected is a new connection with one consent at the fake behind it.
func (f *fixture) connected() core.ConnectionRef {
	ref := f.pending()
	f.consent(ref)
	return ref
}

// consent runs a consent for ref through the scheme and stores what it got, as the
// authorization callback does (internal/api/authorizations.go completeConsent): connected,
// expiry unknown.
func (f *fixture) consent(ref core.ConnectionRef) {
	scheme := f.scheme(f.srv.Client())
	connection := f.stored(ref)
	definition, err := f.db.ConnectorDefinition(f.ctx, ref.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	require.NoError(f.tb, err)
	resolved, err := definition.Manifest.Resolve(oauth2code.Name, connection.Inputs, connection.Metadata)
	require.NoError(f.tb, err)
	out, err := scheme.Begin(f.ctx, core.BeginInput{Ref: ref, Manifest: resolved, RedirectURI: fakeprovider.RedirectURI})
	require.NoError(f.tb, err)
	callback, err := f.srv.Consent(out.AuthorizeURL)
	require.NoError(f.tb, err)
	credentials, account, err := scheme.Complete(f.ctx, core.CompleteInput{Ref: ref, Manifest: resolved, State: out.State, Query: callback.Query()})
	require.NoError(f.tb, err)
	f.update(ref, func(state *core.CredentialState) {
		state.Credentials = credentials
		state.Status = store.ConnectionConnected
		state.LastError = ""
		state.AccountID, state.Metadata, state.Scopes = account.AccountID, account.Metadata, account.Scopes
		state.ExpiresAt = time.Time{}
		state.ConnectedAt = time.Now().UTC()
	})
}

// update changes ref's credential state under the lock, through a credential store of the
// fixture's own.
func (f *fixture) update(ref core.ConnectionRef, change func(state *core.CredentialState)) {
	credentials, err := pgsealed.New(f.db, f.sealer)
	require.NoError(f.tb, err)
	require.NoError(f.tb, credentials.Update(f.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		change(state)
		return true, nil
	}))
}

// hold takes ref's lock on a router of its own and keeps it until the returned release is
// called, or the test ends.
func (f *fixture) hold(ref core.ConnectionRef) (release func()) {
	db, err := store.Open(f.dsn)
	require.NoError(f.tb, err)
	f.tb.Cleanup(func() { _ = db.Close() })
	credentials, err := pgsealed.New(db, f.sealer)
	require.NoError(f.tb, err)
	locked, released, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		done <- credentials.Update(f.ctx, ref, func(*core.CredentialState, func() error) (bool, error) {
			close(locked)
			<-released
			return false, nil
		})
	}()
	select {
	case <-locked:
	case err := <-done:
		require.NoError(f.tb, err)
	}
	var once sync.Once
	release = func() {
		once.Do(func() {
			close(released)
			require.NoError(f.tb, <-done)
		})
	}
	f.tb.Cleanup(release)
	return release
}

func (f *fixture) stored(ref core.ConnectionRef) store.ConnectorConnection {
	connection, err := f.db.ConnectorConnection(f.ctx, ref.CustomerID, ref.ConnectionID)
	require.NoError(f.tb, err)
	return connection
}

// due moves the schemes' clock to just past the access token's expiry, so the next
// Retrieve refreshes and no still valid token comes back beside a failure.
func (f *fixture) due() {
	f.clock.Add(fakeprovider.AccessTTL + time.Second)
}

// works is whether the fake's MCP endpoint takes a tool call carrying credential, wrapped by
// the scheme that issued it.
func (f *fixture) works(credential core.AccessCredential) bool {
	var scheme core.Scheme = f.scheme(f.srv.Client())
	if credential.Scheme == oauth2cc.Name {
		scheme = f.clientCredentials(f.srv.Client())
	}
	client := &http.Client{Transport: scheme.Wrap(f.srv.Client().Transport, credential)}
	request, err := http.NewRequest(http.MethodPost, f.srv.URL+fakeprovider.PathMCP,
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`))
	require.NoError(f.tb, err)
	response, err := client.Do(request)
	require.NoError(f.tb, err)
	_, _ = io.Copy(io.Discard, response.Body)
	require.NoError(f.tb, response.Body.Close())
	return response.StatusCode == http.StatusOK
}

// token is the access token a credential carries, so a test can tell one issued token from
// another. Compare two with ==, not Equal, so a failure prints no token.
func (f *fixture) token(credential core.AccessCredential) string {
	var secret struct {
		AccessToken string `json:"access_token"`
	}
	require.NoError(f.tb, json.Unmarshal(credential.Secret(), &secret))
	return secret.AccessToken
}

// interposed is the fake's client with a hook that runs when a refresh grant reaches the
// transport, before it is sent.
func (f *fixture) interposed(onRefresh func()) *http.Client {
	base := f.srv.Client()
	return &http.Client{Transport: &interposer{base: base.Transport, onRefresh: onRefresh}}
}

type interposer struct {
	base      http.RoundTripper
	onRefresh func()
}

func (i *interposer) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Path == fakeprovider.PathToken && r.Body != nil {
		body, err := io.ReadAll(r.Body)
		_ = r.Body.Close()
		if err != nil {
			return nil, err
		}
		if form, err := url.ParseQuery(string(body)); err == nil && form.Get("grant_type") == "refresh_token" {
			i.onRefresh()
		}
		r.Body = io.NopCloser(strings.NewReader(string(body)))
	}
	return i.base.RoundTrip(r)
}

// clock is the time the schemes and the resolvers read, moved by the test.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// loopbackOrPublic is egress's endpoint check with one hole, a loopback IP literal, where the
// fake listens (as in the oauth2code suite).
func loopbackOrPublic(ctx context.Context, raw string) error {
	if u, err := url.Parse(raw); err == nil {
		if ip, err := netip.ParseAddr(u.Hostname()); err == nil && ip.IsLoopback() && u.User == nil {
			return nil
		}
	}
	return egress.ValidatePublicHTTPSURL(ctx, raw)
}
