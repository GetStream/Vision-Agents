//go:build integration

package pgsealed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/fstest"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// manifest is a connector whose one scheme is the test's own, so a connection can be made.
const manifest = `
id: acme
revision: 1
name: Acme
category: Testing
endpoints:
  mcp: https://mcp.acme.example/mcp
schemes: [test_rotating]
sources:
  - kind: mcp
    endpoint: mcp
`

// registered is a scheme that is only a name: the store checks a connection's scheme is
// registered, and the refresh these tests drive is written in the test, not in a scheme.
type registered string

func (n registered) Name() string { return string(n) }

func (registered) Begin(context.Context, core.BeginInput) (core.BeginOutput, error) {
	return core.BeginOutput{Done: true}, nil
}

func (registered) Complete(context.Context, core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	return core.StoredCredentials{}, core.AccountInfo{}, nil
}

func (registered) Retrieve(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest, _ core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	return core.AccessCredential{}, stored, nil
}

func (registered) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return base
}

func (registered) Classify(*http.Response, []byte, error) core.Outcome { return core.Outcome{} }

func (registered) Revoke(context.Context, core.StoredCredentials, core.ResolvedManifest) error {
	return nil
}

var schemes = core.Registry{Schemes: map[string]core.Scheme{"test_rotating": registered("test_rotating")}}

// tokens is the test scheme's StoredCredentials payload.
type tokens struct {
	Access  string `json:"access_token"`
	Refresh string `json:"refresh_token"`
}

var (
	// errReauthorize is the test resolver finding a connection it may not use.
	errReauthorize = errors.New("the connection needs to be reauthorized")
	// errUncertain is a refresh whose outcome the caller never learned.
	errUncertain = errors.New("the refresh did not finish")
)

// lostRefresh is the LastError the test resolver checkpoints before it spends a refresh
// token, which stays if it never learns what became of it.
const lostRefresh = "A refresh did not finish durably; reconnect the account"

type PGSealedSuite struct {
	suite.Suite
	ctx context.Context
	dsn string
	db  *store.Store
	// v1 seals under key version 1, the keyring before a rotation.
	v1 *auth.Sealer
}

func TestPGSealedSuite(t *testing.T) {
	suite.Run(t, new(PGSealedSuite))
}

func (s *PGSealedSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	// A database of this suite's own, as the store suite has, since it drops the schema.
	s.dsn = testenv.Database(dsn, "pgsealed")
	db, err := store.Open(s.dsn)
	s.Require().NoError(err)
	s.db = db
	s.Require().NoError(db.Ping(s.ctx))

	var database string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.Require().NoError(db.SeedConnectorDefinitions(s.ctx, fstest.MapFS{"acme.yaml": {Data: []byte(manifest)}}))

	s.v1, err = auth.NewSealerWithKeyring(1, map[int]string{1: "test key one"})
	s.Require().NoError(err)
}

func (s *PGSealedSuite) TearDownSuite() {
	if s.db != nil {
		s.Require().NoError(s.db.Close())
	}
}

func (s *PGSealedSuite) SetupTest() {
	_, err := s.db.DB().ExecContext(s.ctx, "TRUNCATE connector_connections, connector_authorization_attempts CASCADE")
	s.Require().NoError(err)
}

// credentialStore is a CredentialStore over the suite's database, sealing with sealer.
func (s *PGSealedSuite) credentialStore(sealer *auth.Sealer) *CredentialStore {
	credentialStore, err := New(s.db, sealer)
	s.Require().NoError(err)
	return credentialStore
}

// router is another router's credential store: a pool of its own, connected before the test starts.
func (s *PGSealedSuite) router(sealer *auth.Sealer) *CredentialStore {
	db, err := store.Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { db.Close() })
	s.Require().NoError(db.Ping(s.ctx))
	credentialStore, err := New(db, sealer)
	s.Require().NoError(err)
	return credentialStore
}

// connected is a new connection that a consent left holding held, expiring at expires.
func (s *PGSealedSuite) connected(customerID string, held tokens, expires time.Time) core.ConnectionRef {
	connection := &store.ConnectorConnection{
		CustomerID: customerID, ConnectorID: "acme", DefinitionRevision: 1,
		OwnerType: store.OwnerApp, AuthScheme: "test_rotating",
	}
	s.Require().NoError(s.db.CreateConnectorConnection(s.ctx, schemes, connection))
	ref := core.ConnectionRef{CustomerID: customerID, ConnectionID: connection.ID}
	s.Require().NoError(s.credentialStore(s.v1).Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = credentials(held)
		state.Status = store.ConnectionConnected
		state.ExpiresAt = expires
		return true, nil
	}))
	return ref
}

func (s *PGSealedSuite) stored(ref core.ConnectionRef) store.ConnectorConnection {
	connection, err := s.db.ConnectorConnection(s.ctx, ref.CustomerID, ref.ConnectionID)
	s.Require().NoError(err)
	return connection
}

// held is the credential state the credential store opens for ref, read without changing it.
func (s *PGSealedSuite) held(credentialStore *CredentialStore, ref core.ConnectionRef) core.CredentialState {
	var seen core.CredentialState
	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		seen = *state
		return false, nil
	}))
	return seen
}

func credentials(held tokens) core.StoredCredentials {
	payload, err := json.Marshal(held)
	if err != nil {
		panic(err)
	}
	return core.StoredCredentials{Scheme: "test_rotating", Version: 1, Payload: payload}
}

func opened(c core.StoredCredentials) tokens {
	var held tokens
	if err := json.Unmarshal(c.Payload, &held); err != nil {
		panic(err)
	}
	return held
}

// rotation is a token endpoint that rotates the refresh token on every use and revokes the
// grant when a spent one comes back, which is what RFC 9700 §4.14.2 has an authorization
// server do with a replayed rotating refresh token.
type rotation struct {
	server *httptest.Server
	// refreshes counts the refresh requests that reached it.
	refreshes atomic.Int32
	// revoked is set by a replayed refresh token.
	revoked atomic.Bool
	mu      sync.Mutex
	current string
	issued  int
}

// rotating starts an endpoint whose live refresh token is refresh. deliver writes the
// rotated tokens, or fails to; before it runs, before reports what the endpoint sees.
func (s *PGSealedSuite) rotating(refresh string, before func(), deliver func(http.ResponseWriter, *http.Request, tokens)) *rotation {
	r := &rotation{current: refresh}
	r.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		r.refreshes.Add(1)
		if before != nil {
			before()
		}
		if err := req.ParseForm(); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		r.mu.Lock()
		if r.revoked.Load() || req.PostForm.Get("refresh_token") != r.current {
			r.revoked.Store(true)
			r.mu.Unlock()
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"invalid_grant"}`))
			return
		}
		r.issued++
		next := tokens{Access: fmt.Sprintf("access-%d", r.issued), Refresh: fmt.Sprintf("refresh-%d", r.issued)}
		r.current = next.Refresh
		r.mu.Unlock()
		// Long enough for every concurrent caller to be queued on the lock meanwhile.
		time.Sleep(20 * time.Millisecond)
		deliver(w, req, next)
	}))
	s.T().Cleanup(r.server.Close)
	return r
}

func answer(w http.ResponseWriter, _ *http.Request, next tokens) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(next)
}

// resolve is what a resolver does with the credential store, in the shape the prototype's
// ResolveCredentials did (internal/connectors/runtime.go:26-153 at cf62af0d): use the access token
// while it is fresh, and otherwise checkpoint needs_reauthorization before spending the
// refresh token, so an outcome it never learns leaves the connection to be reconnected
// rather than the spent token to be sent again.
func resolve(ctx context.Context, credentialStore *CredentialStore, ref core.ConnectionRef, endpoint *rotation) (tokens, error) {
	var resolved tokens
	err := credentialStore.Update(ctx, ref, func(state *core.CredentialState, checkpoint func() error) (bool, error) {
		if state.Status != store.ConnectionConnected {
			return false, errReauthorize
		}
		held := opened(state.Credentials)
		if time.Until(state.ExpiresAt) > time.Minute {
			resolved = held
			return false, nil
		}
		state.Status = store.ConnectionNeedsReauthorization
		state.LastError = lostRefresh
		if err := checkpoint(); err != nil {
			return false, err
		}
		next, err := refresh(ctx, endpoint, held.Refresh)
		if err != nil {
			return false, err
		}
		state.Credentials = credentials(next)
		state.Status = store.ConnectionConnected
		state.LastError = ""
		state.ExpiresAt = time.Now().Add(time.Hour)
		resolved = next
		return true, nil
	})
	return resolved, err
}

func refresh(ctx context.Context, endpoint *rotation, refreshToken string) (tokens, error) {
	form := url.Values{"grant_type": {"refresh_token"}, "refresh_token": {refreshToken}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.server.URL, strings.NewReader(form.Encode()))
	if err != nil {
		return tokens{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := endpoint.server.Client().Do(req)
	if err != nil {
		return tokens{}, fmt.Errorf("%w: %v", errUncertain, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return tokens{}, fmt.Errorf("refresh: status %d", resp.StatusCode)
	}
	var next tokens
	if err := json.NewDecoder(resp.Body).Decode(&next); err != nil {
		return tokens{}, fmt.Errorf("%w: %v", errUncertain, err)
	}
	return next, nil
}

func (s *PGSealedSuite) TestConcurrentCredentialResolutionCommitsOneRotatedRefreshToken() {
	endpoint := s.rotating("refresh-0", nil, answer)
	ref := s.connected("acme-app", tokens{Access: "access-0", Refresh: "refresh-0"}, time.Now().Add(-time.Minute))
	before := s.stored(ref).Revision

	const callers = 8
	routers := make([]*CredentialStore, callers)
	for i := range routers {
		routers[i] = s.router(s.v1)
	}
	resolved := make([]tokens, callers)
	errs := make([]error, callers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, router := range routers {
		wg.Go(func() {
			<-start
			resolved[i], errs[i] = resolve(s.ctx, router, ref, endpoint)
		})
	}
	close(start)
	wg.Wait()

	for i := range callers {
		s.Require().NoError(errs[i])
		s.Equal(tokens{Access: "access-1", Refresh: "refresh-1"}, resolved[i], "every caller gets the one rotation")
	}
	s.Equal(int32(1), endpoint.refreshes.Load(), "a rotating refresh token is spent once")
	s.False(endpoint.revoked.Load(), "no spent refresh token was sent again")

	stored := s.stored(ref)
	s.Equal(before+1, stored.Revision, "one rotation is one revision")
	s.Equal(store.ConnectionConnected, stored.Status)
	s.NotContains(string(stored.CredentialsSealed), "refresh-1", "the credentials are stored sealed")
	s.Equal(tokens{Access: "access-1", Refresh: "refresh-1"}, opened(s.held(s.credentialStore(s.v1), ref).Credentials))
}

func (s *PGSealedSuite) TestRefreshOutcomeSurvivesLostResponsesAndCanceledWorkers() {
	failures := []struct {
		name    string
		deliver func(cancel context.CancelFunc) func(http.ResponseWriter, *http.Request, tokens)
	}{
		// The provider rotated, and the body that says so never arrives whole.
		{"lost response", func(context.CancelFunc) func(http.ResponseWriter, *http.Request, tokens) {
			return func(w http.ResponseWriter, _ *http.Request, _ tokens) {
				_, _ = w.Write([]byte(`{"access_token":`))
			}
		}},
		// The worker is canceled while the provider is answering.
		{"canceled worker", func(cancel context.CancelFunc) func(http.ResponseWriter, *http.Request, tokens) {
			return func(_ http.ResponseWriter, req *http.Request, _ tokens) {
				cancel()
				<-req.Context().Done()
			}
		}},
	}
	for _, failure := range failures {
		name := failure.name
		ctx, cancel := context.WithCancel(s.ctx)
		var ref core.ConnectionRef
		var checkpointed atomic.Bool
		endpoint := s.rotating("refresh-0", func() {
			checkpointed.Store(s.stored(ref).Status == store.ConnectionNeedsReauthorization)
		}, failure.deliver(cancel))
		ref = s.connected("acme-app", tokens{Access: "access-0", Refresh: "refresh-0"}, time.Now().Add(30*time.Second))
		before := s.stored(ref)

		_, err := resolve(ctx, s.router(s.v1), ref, endpoint)
		cancel()
		s.ErrorIs(err, errUncertain, name)
		s.True(checkpointed.Load(), "%s: needs_reauthorization is committed before the token leaves", name)

		stored := s.stored(ref)
		s.Equal(store.ConnectionNeedsReauthorization, stored.Status, name)
		s.Equal(lostRefresh, stored.LastError, name)
		s.Equal(before.Revision, stored.Revision, "%s: no credentials were saved", name)
		s.True(bytes.Equal(before.CredentialsSealed, stored.CredentialsSealed), "%s: the blob is the one before the refresh", name)

		_, err = resolve(s.ctx, s.router(s.v1), ref, endpoint)
		s.ErrorIs(err, errReauthorize, "%s: the next caller does not refresh", name)
		s.Equal(int32(1), endpoint.refreshes.Load(), "%s: the spent refresh token is never sent again", name)
		s.False(endpoint.revoked.Load(), name)
	}
}

func (s *PGSealedSuite) TestNewCredentialsAreSealedForTheNextRevision() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	s.Equal(2, s.stored(ref).Revision, "the consent's credentials are revision 2 of a connection created at 1")

	s.Require().NoError(s.credentialStore(s.v1).Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		s.Equal(2, state.Revision)
		state.Credentials = credentials(tokens{Access: "b", Refresh: "r"})
		return true, nil
	}))

	stored := s.stored(ref)
	s.Equal(3, stored.Revision)
	s.Equal(tokens{Access: "b", Refresh: "r"}, opened(s.held(s.credentialStore(s.v1), ref).Credentials))
}

func (s *PGSealedSuite) TestAStatusChangeKeepsTheRevisionAndTheBlob() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	before := s.stored(ref)

	s.Require().NoError(s.credentialStore(s.v1).Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Status = store.ConnectionNeedsReauthorization
		state.LastError = "Reconnect the account"
		return true, nil
	}))

	stored := s.stored(ref)
	s.Equal(before.Revision, stored.Revision)
	s.True(bytes.Equal(before.CredentialsSealed, stored.CredentialsSealed), "the blob is unchanged")
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal("Reconnect the account", stored.LastError)
}

func (s *PGSealedSuite) TestTheAccountAConsentFoundIsWrittenBesideTheCredentialsAndARefreshKeepsIt() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	credentialStore := s.credentialStore(s.v1)
	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = credentials(tokens{Access: "a2", Refresh: "r2"})
		state.AccountID = "T1:U1"
		state.Metadata = map[string]string{"team_id": "T1"}
		state.Scopes = []string{"chat:write"}
		return true, nil
	}))

	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		s.Equal("T1:U1", state.AccountID, "what the consent wrote is what the next use sees")
		state.Credentials = credentials(tokens{Access: "a3", Refresh: "r3"})
		return true, nil
	}))

	stored := s.stored(ref)
	s.Equal(4, stored.Revision)
	s.Equal("T1:U1", stored.AccountID)
	s.Equal(map[string]string{"team_id": "T1"}, stored.Metadata)
	s.Equal([]string{"chat:write"}, stored.GrantedScopes)
}

func (s *PGSealedSuite) TestWhenAConsentConnectedIsKeptAndARefreshKeepsIt() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	credentialStore := s.credentialStore(s.v1)
	connected := time.Date(2026, 10, 6, 12, 0, 0, 0, time.UTC)
	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.ConnectedAt = connected
		return true, nil
	}))

	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		s.True(connected.Equal(state.ConnectedAt), "what the consent wrote is what the next use sees")
		state.Credentials = credentials(tokens{Access: "a2", Refresh: "r2"})
		return true, nil
	}))

	stored := s.stored(ref)
	s.Require().NotNil(stored.ConnectedAt)
	s.True(connected.Equal(*stored.ConnectedAt))
}

// A credentials write that connects a connection names no time of its own (T18's write,
// AI-849); the grant begins when it is committed.
func (s *PGSealedSuite) TestAConnectionCommittedConnectedFromAnotherStatusIsStampedConnectedNow() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	credentialStore := s.credentialStore(s.v1)
	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Status = store.ConnectionNeedsReauthorization
		return true, nil
	}))
	before := time.Now().UTC()

	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = credentials(tokens{Access: "a2", Refresh: "r2"})
		state.Status = store.ConnectionConnected
		return true, nil
	}))

	stored := s.stored(ref)
	s.Require().NotNil(stored.ConnectedAt)
	s.False(stored.ConnectedAt.Before(before.Truncate(time.Microsecond)))
}

// A refresh checkpoints needs_reauthorization and commits connected again; it is the same
// grant, so its time stays.
func (s *PGSealedSuite) TestARefreshPastItsCheckpointKeepsWhenTheGrantBegan() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	credentialStore := s.credentialStore(s.v1)
	connected := time.Date(2026, 10, 6, 12, 0, 0, 0, time.UTC)
	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.ConnectedAt = connected
		return true, nil
	}))

	s.Require().NoError(credentialStore.Update(s.ctx, ref, func(state *core.CredentialState, checkpoint func() error) (bool, error) {
		state.Status = store.ConnectionNeedsReauthorization
		if err := checkpoint(); err != nil {
			return false, err
		}
		state.Credentials = credentials(tokens{Access: "a2", Refresh: "r2"})
		state.Status = store.ConnectionConnected
		return true, nil
	}))

	stored := s.stored(ref)
	s.Require().NotNil(stored.ConnectedAt)
	s.True(connected.Equal(*stored.ConnectedAt))
}

func (s *PGSealedSuite) TestTheRevisionIsNotTheCallbacksToMove() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})

	err := s.credentialStore(s.v1).Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Revision = 7
		return true, nil
	})
	s.ErrorIs(err, errRevisionIsTheStores)
	s.Equal(2, s.stored(ref).Revision)
}

func (s *PGSealedSuite) TestCredentialsSealedUnderAnOlderKeyAreRewrappedOnNextUse() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	before := s.stored(ref)
	s.Equal(1, before.CredentialsKEKVersion)

	rotated, err := auth.NewSealerWithKeyring(2, map[int]string{1: "test key one", 2: "test key two"})
	s.Require().NoError(err)
	used := s.held(s.credentialStore(rotated), ref)
	s.Equal(tokens{Access: "a", Refresh: "r"}, opened(used.Credentials), "the old key still opens it")

	stored := s.stored(ref)
	s.Equal(2, stored.CredentialsKEKVersion, "a use that changed nothing still rewraps")
	s.Equal(before.Revision, stored.Revision, "a rewrap is the same credentials at the same revision")
	s.False(bytes.Equal(before.CredentialsSealed, stored.CredentialsSealed), "the blob is sealed again")

	retired, err := auth.NewSealerWithKeyring(2, map[int]string{2: "test key two"})
	s.Require().NoError(err)
	s.Equal(tokens{Access: "a", Refresh: "r"}, opened(s.held(s.credentialStore(retired), ref).Credentials),
		"once rewrapped, the old key can be retired")
}

func (s *PGSealedSuite) TestAFailedUseDoesNotRewrap() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	rotated, err := auth.NewSealerWithKeyring(2, map[int]string{1: "test key one", 2: "test key two"})
	s.Require().NoError(err)

	failed := errors.New("the use failed")
	err = s.credentialStore(rotated).Update(s.ctx, ref, func(*core.CredentialState, func() error) (bool, error) { return false, failed })
	s.ErrorIs(err, failed)
	s.Equal(1, s.stored(ref).CredentialsKEKVersion)
}

func (s *PGSealedSuite) TestABlobFromAnEarlierRevisionDoesNotOpen() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "spent"}, time.Time{})
	earlier := s.stored(ref).CredentialsSealed
	s.Require().NoError(s.credentialStore(s.v1).Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = credentials(tokens{Access: "b", Refresh: "live"})
		return true, nil
	}))
	// Somebody with write access to the table puts the earlier credentials back.
	_, err := s.db.DB().ExecContext(s.ctx, "UPDATE connector_connections SET credentials_sealed = ? WHERE id = ?", earlier, ref.ConnectionID)
	s.Require().NoError(err)

	s.assertUnreadable(ref)
}

func (s *PGSealedSuite) TestABlobFromAnotherConnectionDoesNotOpen() {
	theirs := s.connected("acme-app", tokens{Access: "theirs", Refresh: "theirs"}, time.Time{})
	mine := s.connected("acme-app", tokens{Access: "mine", Refresh: "mine"}, time.Time{})
	s.Equal(s.stored(theirs).Revision, s.stored(mine).Revision, "both at the same revision, so only the id tells them apart")
	_, err := s.db.DB().ExecContext(s.ctx, "UPDATE connector_connections SET credentials_sealed = ? WHERE id = ?",
		s.stored(theirs).CredentialsSealed, mine.ConnectionID)
	s.Require().NoError(err)

	s.assertUnreadable(mine)
}

func (s *PGSealedSuite) TestABlobFromAnotherCustomerDoesNotOpen() {
	theirs := s.connected("other-app", tokens{Access: "theirs", Refresh: "theirs"}, time.Time{})
	// The row moves to another customer with its id, revision and blob, as an import that
	// kept the blob would leave it.
	_, err := s.db.DB().ExecContext(s.ctx, "UPDATE connector_connections SET customer_id = 'acme-app' WHERE id = ?", theirs.ConnectionID)
	s.Require().NoError(err)

	s.assertUnreadable(core.ConnectionRef{CustomerID: "acme-app", ConnectionID: theirs.ConnectionID})
}

func (s *PGSealedSuite) TestAKeyVersionMissingFromTheKeyringWritesNothingAndItsReturnRestoresTheConnection() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	before := s.stored(ref)
	// A deploy that dropped key version 1 before every row was rewrapped.
	misconfigured, err := auth.NewSealerWithKeyring(2, map[int]string{2: "test key two"})
	s.Require().NoError(err)

	err = s.credentialStore(misconfigured).Update(s.ctx, ref, func(*core.CredentialState, func() error) (bool, error) {
		s.Fail("fn does not run on credentials no key here can open")
		return false, nil
	})
	s.ErrorIs(err, auth.ErrKeyVersionUnavailable)
	stored := s.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status, "a keyring fault is not the connection's")
	s.Empty(stored.LastError)
	s.Equal(before.Revision, stored.Revision)

	restored, err := auth.NewSealerWithKeyring(2, map[int]string{1: "test key one", 2: "test key two"})
	s.Require().NoError(err)
	state := s.held(s.credentialStore(restored), ref)
	s.Equal(store.ConnectionConnected, state.Status, "putting the key back restores the connection")
	s.Equal(tokens{Access: "a", Refresh: "r"}, opened(state.Credentials))
}

func (s *PGSealedSuite) TestARewrapKeepsWhatAnUnchangedUseEditedButDidNotCommit() {
	ref := s.connected("acme-app", tokens{Access: "a", Refresh: "r"}, time.Time{})
	before := s.stored(ref)
	rotated, err := auth.NewSealerWithKeyring(2, map[int]string{1: "test key one", 2: "test key two"})
	s.Require().NoError(err)

	err = s.credentialStore(rotated).Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Status = store.ConnectionDisconnected
		state.LastError = "an edit fn did not ask to keep"
		state.Credentials = credentials(tokens{Access: "edited", Refresh: "edited"})
		return false, nil
	})
	s.Require().NoError(err)

	stored := s.stored(ref)
	s.Equal(2, stored.CredentialsKEKVersion, "the rewrap still happens")
	s.Equal(before.Revision, stored.Revision, "fn's StoredCredentials were not committed")
	s.Equal(store.ConnectionConnected, stored.Status)
	s.Empty(stored.LastError)
	s.Equal(tokens{Access: "a", Refresh: "r"}, opened(s.held(s.credentialStore(rotated), ref).Credentials))
}

// assertUnreadable checks that a connection whose blob does not open is handed over empty and
// needing reauthorization, durably, and that a reconnect then replaces the blob.
func (s *PGSealedSuite) assertUnreadable(ref core.ConnectionRef) {
	before := s.stored(ref)
	state := s.held(s.credentialStore(s.v1), ref)
	s.Equal(store.ConnectionNeedsReauthorization, state.Status)
	s.Equal(core.StoredCredentials{}, state.Credentials)

	stored := s.stored(ref)
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status, "committed before the callback ran")
	s.Equal(unreadableError, stored.LastError)
	s.True(bytes.Equal(before.CredentialsSealed, stored.CredentialsSealed), "the blob is left as it was")

	s.Require().NoError(s.credentialStore(s.v1).Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = credentials(tokens{Access: "reconnected", Refresh: "reconnected"})
		state.Status = store.ConnectionConnected
		state.LastError = ""
		return true, nil
	}))
	reconnected := s.held(s.credentialStore(s.v1), ref)
	s.Equal(store.ConnectionConnected, reconnected.Status)
	s.Equal(before.Revision+1, reconnected.Revision)
	s.Equal(tokens{Access: "reconnected", Refresh: "reconnected"}, opened(reconnected.Credentials))
}
