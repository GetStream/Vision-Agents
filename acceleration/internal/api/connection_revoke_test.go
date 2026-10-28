//go:build integration

package api

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// revokeFailed is the line revokeAtProvider logs when the provider did not revoke (AI-1544).
const revokeFailed = "could not revoke a grant at the provider"

// optInRevoke is the manifest YAML that opts a connector into a revoke on delete.
const optInRevoke = "revoke_on_delete: true\n"

// TestDeletingAConnectionWhoseManifestOptsInEndsItsGrantAtTheProvider (AI-1544): the delete
// revokes at the manifest's revoke endpoint, and the refresh token the router held is then
// refused, so the grant is gone at the provider and not only in the router.
func (s *AuthorizationsSuite) TestDeletingAConnectionWhoseManifestOptsInEndsItsGrantAtTheProvider() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	s.connect(id)
	stored, manifest := s.held(id)
	before := s.provider.Hits(fakeprovider.PathRevoke)
	revoked := len(s.provider.Revoked())

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil))

	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
	// The refresh token, which a provider must revoke (RFC 7009 section 2); an access token
	// past its expiry may revoke nothing, as at Nessy.
	s.Equal([]string{"refresh_token"}, s.provider.Revoked()[revoked:])
	var refused *core.OutcomeError
	s.Require().ErrorAs(s.renew(stored, manifest), &refused)
	s.Equal(core.OutcomeInvalidGrant, refused.Outcome.Kind)
	s.Empty(s.logged.lines(revokeFailed, id))
}

// TestDeletingAConnectionWhoseManifestDoesNotOptInLeavesItsGrantAtTheProvider is the control:
// the same connector without revoke_on_delete sends nothing, and the held refresh token still
// renews, as on the base (a103c19ab), which had no revoke on delete.
func (s *AuthorizationsSuite) TestDeletingAConnectionWhoseManifestDoesNotOptInLeavesItsGrantAtTheProvider() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, "")
	s.connect(id)
	stored, manifest := s.held(id)
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil))

	s.Equal(before, s.provider.Hits(fakeprovider.PathRevoke))
	s.NoError(s.renew(stored, manifest))
}

// A revoke the provider refuses is logged, and the connection is deleted all the same.
func (s *AuthorizationsSuite) TestARevokeTheProviderRefusesStillDeletesTheConnection() {
	// The fake serves no such path, so it answers 404: nothing was revoked.
	id := s.revokingConnection(s.provider.URL+"/no-revoke-here", optInRevoke)
	s.connect(id)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil))

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id, nil, nil))
	s.Contains(s.logged.line(s.T(), revokeFailed, id), "HTTP 404")
}

// A delete the store refuses, since an agent config binds the connection, revokes nothing:
// the connection stays, and so must its grant.
func (s *AuthorizationsSuite) TestARefusedDeleteRevokesNothing() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	s.connect(id)
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "binder-" + s.utils.uuid(), "connectors": []map[string]any{{"name": "crm", "connector_id": s.get(id).ConnectorID,
			"connection": map[string]any{"type": "fixed", "connection_id": id}, "tools": []map[string]any{}}},
	}, nil))
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusConflict, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil))

	s.Equal(before, s.provider.Hits(fakeprovider.PathRevoke))
	s.NoError(s.refresh(id))
}

// A forced delete of a connection an agent config binds deletes it, and revokes its grant as
// an unforced delete does.
func (s *AuthorizationsSuite) TestAForcedDeleteOfABoundConnectionEndsItsGrantAtTheProvider() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	s.connect(id)
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "binder-" + s.utils.uuid(), "connectors": []map[string]any{{"name": "crm", "connector_id": s.get(id).ConnectorID,
			"connection": map[string]any{"type": "fixed", "connection_id": id}, "tools": []map[string]any{}}},
	}, nil))
	stored, manifest := s.held(id)
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+id+"?force=true", nil, nil))

	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
	var refused *core.OutcomeError
	s.Require().ErrorAs(s.renew(stored, manifest), &refused)
	s.Equal(core.OutcomeInvalidGrant, refused.Outcome.Kind)
}

// A caller that hangs up once the row is deleted does not cancel the revoke: the router can
// never revoke the grant later, since the row that held its tokens is gone.
func (s *AuthorizationsSuite) TestACallerThatHangsUpStillEndsTheGrantAtTheProvider() {
	reached := make(chan struct{})
	gone := make(chan struct{})
	provider, err := url.Parse(s.provider.URL)
	s.Require().NoError(err)
	proxy := httputil.NewSingleHostReverseProxy(provider)
	proxy.Transport = s.provider.Client().Transport
	// The revoke endpoint says it was reached, waits for the caller to hang up, then a second
	// more, in which a revoke cancelled with the caller closes its connection and is dropped
	// here; a revoke still alive is passed on to the fake.
	slow := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Read up front: net/http notices a closed connection, and cancels r's context, only
		// once the handler has read the whole body.
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		close(reached)
		select {
		case <-gone:
		case <-time.After(5 * time.Second):
		}
		select {
		case <-r.Context().Done():
			return
		case <-time.After(time.Second):
		}
		r.URL.Path = fakeprovider.PathRevoke
		proxy.ServeHTTP(w, r)
	}))
	defer slow.Close()
	id := s.revokingConnection(slow.URL+"/revoke", optInRevoke)
	s.connect(id)
	stored, manifest := s.held(id)
	before := s.provider.Hits(fakeprovider.PathRevoke)

	ctx, hangUp := context.WithCancel(context.Background())
	defer hangUp()
	request, err := http.NewRequestWithContext(ctx, http.MethodDelete, s.server.URL+"/v1/agents/connections/"+id, nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()
	answered := make(chan error, 1)
	go func() {
		_, err := s.server.Client().Do(request) //nolint:bodyclose // no response: the caller hung up
		answered <- err
	}()
	// The caller hangs up once the revoke reached the provider, so after the delete committed:
	// a hang-up before it cancels the delete, which leaves the connection and its grant live.
	select {
	case <-reached:
	case err := <-answered:
		s.Require().FailNow("the delete was answered before its revoke reached the provider", "error: %v", err)
	case <-time.After(5 * time.Second):
		s.Require().FailNow("the delete's revoke never reached the provider")
	}
	hangUp()
	s.Require().ErrorIs(<-answered, context.Canceled)
	close(gone)

	s.Require().Eventually(func() bool { return s.provider.Hits(fakeprovider.PathRevoke) == before+1 }, 5*time.Second, 20*time.Millisecond)
	var refused *core.OutcomeError
	s.Require().Eventually(func() bool { return errors.As(s.renew(stored, manifest), &refused) }, 5*time.Second, 20*time.Millisecond)
	s.Equal(core.OutcomeInvalidGrant, refused.Outcome.Kind)
	s.Empty(s.logged.lines(revokeFailed, id))
}

// A caller that hangs up once the delete committed still leaves the delete's "grant revoked"
// audit row (AI-2163): nothing could write it later. The test holds one of the connection's
// MCP event subscriptions locked, so the delete waits in mcpEvents.Stop, after its commit and
// before its audit row, and hangs up there, as net/http does when the caller goes.
func (s *AuthorizationsSuite) TestACallerThatHangsUpAfterTheDeleteStillLeavesItsAuditRow() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	s.connect(id)
	ctx := context.Background()
	_, err := s.store.AddConnectionEventSubscription(ctx, &store.ConnectionEventSubscription{
		CustomerID: s.customerID(), ConnectionID: id, ConfigID: "cfg_" + s.utils.uuid(), Binding: "crm",
		Event: "issue_created", Key: "issue_created", Token: s.utils.uuid(), SecretSealed: []byte("synthetic"),
		KEKVersion: 1, Status: store.ConnectionEventActive,
	})
	s.Require().NoError(err)
	hold, err := s.store.DB().BeginTx(ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = hold.Rollback() }()
	var holder int
	s.Require().NoError(hold.QueryRowContext(ctx, "SELECT pg_backend_pid()").Scan(&holder))
	_, err = hold.ExecContext(ctx, "SELECT 1 FROM connection_event_subscriptions WHERE connection_id = ? FOR UPDATE", id)
	s.Require().NoError(err)

	request, hangUp := context.WithCancel(context.WithValue(context.WithValue(ctx,
		customerContextKey{}, s.customerID()), serverSideContextKey{}, true))
	defer hangUp()
	answered := make(chan error, 1)
	go func() {
		_, err := s.router.deleteConnection(request, &deleteConnectionRequest{ID: id})
		answered <- err
	}()
	s.Require().Eventually(func() bool {
		var waiting int
		err := s.store.DB().QueryRowContext(ctx,
			"SELECT count(*) FROM pg_stat_activity WHERE ? = ANY(pg_blocking_pids(pid))", holder).Scan(&waiting)
		return err == nil && waiting > 0
	}, 5*time.Second, 10*time.Millisecond, "the delete never reached its MCP event subscriptions")
	hangUp()
	s.Require().NoError(hold.Rollback())
	s.Require().NoError(<-answered)

	events, err := s.store.ConnectorAuditEvents(ctx, s.customerID(), store.AuditFilter{ConnectionID: id})
	s.Require().NoError(err)
	s.Require().NotEmpty(events)
	s.Equal(store.AuditGrantRevoked, events[0].Action)
	s.Equal(store.AuditReasonDeleted, events[0].Reason)
}

// The client a consent's credentials name is kept although its caller hung up once the
// consent committed (AI-2163): only the next consent would write it again.
func (s *AuthorizationsSuite) TestTheClientOfAConsentWhoseCallerHungUpIsKept() {
	consented := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	s.connect(consented)
	stored, _ := s.held(consented)
	ctx := context.Background()
	recorded, err := s.store.ConnectorConnectionClients(ctx, []string{consented})
	s.Require().NoError(err)
	s.Require().Contains(recorded, consented, "the consent records its client")
	pending := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)

	gone, hangUp := context.WithCancel(ctx)
	hangUp()
	s.router.recordClient(gone, pending, stored)

	clients, err := s.store.ConnectorConnectionClients(ctx, []string{pending})
	s.Require().NoError(err)
	s.Require().Contains(clients, pending)
	s.Equal(recorded[consented].ClientID, clients[pending].ClientID)
}

// A connection that never finished a consent holds no grant, so its delete asks nothing of the
// provider and logs no failure.
func (s *AuthorizationsSuite) TestDeletingAPendingConnectionRevokesNothing() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil))

	s.Equal(before, s.provider.Hits(fakeprovider.PathRevoke))
	s.Empty(s.logged.lines(revokeFailed, id))
}

// The opt-in is read from the revision the connection was connected on, as every other part
// of its manifest is: a later revision that opts in changes nothing until a reconnect.
func (s *AuthorizationsSuite) TestTheOptInIsReadFromTheConnectionsRevision() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, "")
	s.connect(id)
	s.revise(s.get(id).ConnectorID, func(m *core.Manifest) { m.RevokeOnDelete = true })
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil))

	s.Equal(before, s.provider.Hits(fakeprovider.PathRevoke))
}

// revokingConnection is a pending app-owned connection to connector's manifest with its
// revoke endpoint at revoke; extra is more manifest YAML.
func (s *AuthorizationsSuite) revokingConnection(revoke, extra string) string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		appOwned(s.revokingConnector("operator", revoke, extra)), &created))
	return created.ID
}

// revokingConnector is a connector of the test's app at the fake whose client is of
// registration, with its revoke endpoint at revoke; extra is more manifest YAML.
func (s *AuthorizationsSuite) revokingConnector(registration, revoke, extra string) string {
	id := "custom_fake" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(s.fakeManifest(id, 1, registration, extra)))
	s.Require().NoError(err)
	manifest.Endpoints["revoke"] = revoke
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

// held is the stored credentials of connection id and its resolved manifest, read as the
// resolver reads them, so a test can present them to the provider after the delete.
func (s *AuthorizationsSuite) held(id string) (core.StoredCredentials, core.ResolvedManifest) {
	ctx := context.Background()
	connection, err := s.store.ConnectorConnection(ctx, s.customerID(), id)
	s.Require().NoError(err)
	definition, err := s.store.ConnectorDefinition(ctx, s.customerID(), connection.ConnectorID, connection.DefinitionRevision)
	s.Require().NoError(err)
	manifest, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
	s.Require().NoError(err)
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	var stored core.StoredCredentials
	s.Require().NoError(credentials.Update(ctx, core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: id},
		func(state *core.CredentialState, _ func() error) (bool, error) {
			stored = state.Credentials
			return false, nil
		}))
	s.Require().NotEmpty(stored.Payload)
	return stored, manifest
}

// renew presents stored's refresh token to the provider, whatever its expiry says: what the
// provider answers.
func (s *AuthorizationsSuite) renew(stored core.StoredCredentials, manifest core.ResolvedManifest) error {
	_, _, err := s.scheme.Retrieve(context.Background(), stored, manifest, core.RetrieveOptions{Refused: true})
	return err
}
