//go:build integration

package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TestDeletingAUsersConnectionsEndsTheirGrantsAtTheProvider: the user delete revokes as a
// connection's own delete does, with the refresh token, and the provider then refuses it.
func (s *AuthorizationsSuite) TestDeletingAUsersConnectionsEndsTheirGrantsAtTheProvider() {
	alice := s.data.createUser()
	id := s.usersConnection(alice, s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	stored, manifest := s.held(id)
	before, revoked := s.provider.Hits(fakeprovider.PathRevoke), len(s.provider.Revoked())

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/users/"+alice.userID+"/connections", nil, nil))

	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
	s.Equal([]string{"refresh_token"}, s.provider.Revoked()[revoked:])
	var refused *core.OutcomeError
	s.Require().ErrorAs(s.renew(stored, manifest), &refused)
	s.Equal(core.OutcomeInvalidGrant, refused.Outcome.Kind)
	s.Empty(s.logged.lines(revokeFailed, id))
}

// TestDeletingAUsersConnectionsWithoutTheOptInLeavesTheirGrantsAtTheProvider is the control:
// without revoke_on_delete nothing is sent and the held refresh token still renews, as on the
// base (2c083b25b), probed there.
func (s *AuthorizationsSuite) TestDeletingAUsersConnectionsWithoutTheOptInLeavesTheirGrantsAtTheProvider() {
	alice := s.data.createUser()
	id := s.usersConnection(alice, s.provider.URL+fakeprovider.PathRevoke, "")
	stored, manifest := s.held(id)
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/users/"+alice.userID+"/connections", nil, nil))

	s.Equal(before, s.provider.Hits(fakeprovider.PathRevoke))
	s.NoError(s.renew(stored, manifest))
}

// Each of the user's grants is revoked, not only the first.
func (s *AuthorizationsSuite) TestDeletingAUsersConnectionsEndsEachOfTheirGrants() {
	alice := s.data.createUser()
	first := s.usersConnection(alice, s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	second := s.usersConnection(alice, s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	firstStored, firstManifest := s.held(first)
	secondStored, secondManifest := s.held(second)
	before, revoked := s.provider.Hits(fakeprovider.PathRevoke), len(s.provider.Revoked())

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/users/"+alice.userID+"/connections", nil, nil))

	s.Equal(before+2, s.provider.Hits(fakeprovider.PathRevoke))
	s.Equal([]string{"refresh_token", "refresh_token"}, s.provider.Revoked()[revoked:])
	s.assertRefusedAtTheProvider(firstStored, firstManifest)
	s.assertRefusedAtTheProvider(secondStored, secondManifest)
}

// A revoke the provider refuses is logged, and the user's connections are deleted all the same.
func (s *AuthorizationsSuite) TestARevokeTheProviderRefusesStillDeletesTheUsersConnections() {
	alice := s.data.createUser()
	// The fake serves no such path, so it answers 404: nothing was revoked.
	id := s.usersConnection(alice, s.provider.URL+"/no-revoke-here", optInRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/users/"+alice.userID+"/connections", nil, nil))

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id, nil, nil))
	s.Contains(s.logged.line(s.T(), revokeFailed, id), "HTTP 404")
}

// TestARefusedFirstConsentEndsTheTokensItGotAtTheProvider is AI-1643's Done when: Bob finishes
// Alice's consent, nothing is stored, and the tokens the provider issued him are revoked with
// the refresh token.
func (s *AuthorizationsSuite) TestARefusedFirstConsentEndsTheTokensItGotAtTheProvider() {
	alice := s.data.createUser()
	id := s.ownersConnection(s.ownerChecked("owner_email: true\n"+optInRevoke), alice, "alice@fake.example")
	before, revoked := s.provider.Hits(fakeprovider.PathRevoke), len(s.provider.Revoked())

	s.Equal(s.landing(id, consentAccountMismatch), s.finishAs(alice, id))

	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
	s.Equal([]string{"refresh_token"}, s.provider.Revoked()[revoked:])
	s.assertNothingStored(id)
	s.Empty(s.logged.lines(revokeFailed, id))
}

// TestARefusedFirstConsentWithoutTheOptInRevokesNothing is the control: as on the base
// (2c083b25b, probed there), the refused tokens are dropped and left to expire.
func (s *AuthorizationsSuite) TestARefusedFirstConsentWithoutTheOptInRevokesNothing() {
	alice := s.data.createUser()
	id := s.ownersConnection(s.ownerChecked("owner_email: true\n"), alice, "alice@fake.example")
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Equal(s.landing(id, consentAccountMismatch), s.finishAs(alice, id))

	s.Equal(before, s.provider.Hits(fakeprovider.PathRevoke))
	s.assertNothingStored(id)
}

// The opt-in is the attempt's revision's, the one the refused consent ran on: a connection made
// on a revision without it, whose consent starts on a later one with it, revokes.
func (s *AuthorizationsSuite) TestARefusedConsentReadsTheOptInFromTheRevisionItRanOn() {
	alice := s.data.createUser()
	connector := s.ownerChecked("owner_email: true\n")
	id := s.ownersConnection(connector, alice, "alice@fake.example")
	s.revise(connector, func(m *core.Manifest) { m.RevokeOnDelete = true })
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Equal(s.landing(id, consentAccountMismatch), s.finishAs(alice, id))

	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
}

// A reconnect for another account drops what it got as a refused first consent does, so it
// revokes that too. The grant the connection keeps is another one, and still renews.
func (s *AuthorizationsSuite) TestAReconnectForAnotherAccountEndsOnlyTheTokensItGot() {
	id := s.revokingConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	s.connect(id)
	stored, manifest := s.held(id)
	s.provider.SwitchAccount()
	before, revoked := s.provider.Hits(fakeprovider.PathRevoke), len(s.provider.Revoked())

	alice := s.browser()
	finished := alice.finish(s.consent(alice.handOff(s.start(id))))

	s.Equal(s.landing(id, consentAccountMismatch), finished.Header.Get("Location"))
	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
	s.Equal([]string{"refresh_token"}, s.provider.Revoked()[revoked:])
	s.NoError(s.renew(stored, manifest))
}

// TestRemovingTheAppsClientEndsTheGrantsIssuedToItFirst (AI-1648): the revoke needs the
// client's secret, so it runs before the client goes. The connection stays, and needs a
// reconnect: it says so at once, and its export answers that rather than the revoked access
// token. Its grant is refused at the provider even once the same client is put back.
func (s *AuthorizationsSuite) TestRemovingTheAppsClientEndsTheGrantsIssuedToItFirst() {
	connector, id := s.clientConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	stored, manifest := s.held(id)
	before, revoked := s.provider.Hits(fakeprovider.PathRevoke), len(s.provider.Revoked())

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))

	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
	s.Equal([]string{"refresh_token"}, s.provider.Revoked()[revoked:])
	s.Empty(s.logged.lines(revokeFailed, id))
	s.Equal(ConnectionStatus(store.ConnectionNeedsReauthorization), s.get(id).Status, "the connection stays, needing a reconnect")
	s.putClient(connector, s.provider.ClientSecret)
	status, exported := s.serverClient.call(http.MethodPost, tokenPath(id), nil)
	s.Equal(http.StatusConflict, status, string(exported))
	s.Contains(string(exported), "reconnect")
	err := s.renew(stored, manifest)
	var refused *core.OutcomeError
	s.Require().ErrorAs(err, &refused)
	s.Equal(core.OutcomeInvalidGrant, refused.Outcome.Kind)
	var removed *oauth2code.ClientRemovedError
	s.False(errors.As(err, &removed), "the provider refused it, not the router: %v", err)
}

// Each grant the client issued is revoked, not only the first.
func (s *AuthorizationsSuite) TestRemovingTheAppsClientEndsEachGrantIssuedToIt() {
	connector, first := s.clientConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	second := s.clientUsersConnection(connector)
	firstStored, firstManifest := s.held(first)
	secondStored, secondManifest := s.held(second)
	before, revoked := s.provider.Hits(fakeprovider.PathRevoke), len(s.provider.Revoked())

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))

	s.Equal(before+2, s.provider.Hits(fakeprovider.PathRevoke))
	s.Equal([]string{"refresh_token", "refresh_token"}, s.provider.Revoked()[revoked:])
	s.putClient(connector, s.provider.ClientSecret)
	s.assertRefusedAtTheProvider(firstStored, firstManifest)
	s.assertRefusedAtTheProvider(secondStored, secondManifest)
}

// The client's connections of one revision read it once, not once each (R1.3): a client delete
// reads every connection the client issued, opted in or not.
func (s *AuthorizationsSuite) TestRemovingTheAppsClientReadsEachRevisionOnce() {
	connector, _ := s.clientConnection(s.provider.URL+fakeprovider.PathRevoke, "")
	s.clientUsersConnection(connector)
	reads := &definitionReads{connector: connector}
	s.store.DB().AddQueryHook(reads)

	reads.on.Store(true)
	status := s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil)
	reads.on.Store(false)

	s.Require().Equal(http.StatusNoContent, status)
	s.Equal(int64(1), reads.reads.Load())
}

// The opt-in is each connection's revision's: of a client's connection made before the
// connector opted in and one made after, only the second is revoked.
func (s *AuthorizationsSuite) TestRemovingTheAppsClientReadsTheOptInOfEachConnectionsRevision() {
	connector, before := s.clientConnection(s.provider.URL+fakeprovider.PathRevoke, "")
	s.revise(connector, func(m *core.Manifest) { m.RevokeOnDelete = true })
	after := s.clientUsersConnection(connector)
	beforeStored, beforeManifest := s.held(before)
	afterStored, afterManifest := s.held(after)
	hits := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))

	s.Equal(hits+1, s.provider.Hits(fakeprovider.PathRevoke))
	s.putClient(connector, s.provider.ClientSecret)
	s.NoError(s.renew(beforeStored, beforeManifest))
	s.assertRefusedAtTheProvider(afterStored, afterManifest)
}

// TestRemovingTheAppsClientWithoutTheOptInRevokesNothing is the control: as on the base
// (2c083b25b, probed there), nothing is sent, and with the client put back the grant renews.
func (s *AuthorizationsSuite) TestRemovingTheAppsClientWithoutTheOptInRevokesNothing() {
	connector, id := s.clientConnection(s.provider.URL+fakeprovider.PathRevoke, "")
	stored, manifest := s.held(id)
	before := s.provider.Hits(fakeprovider.PathRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))

	s.Equal(before, s.provider.Hits(fakeprovider.PathRevoke))
	s.putClient(connector, s.provider.ClientSecret)
	s.NoError(s.renew(stored, manifest))
}

// Without the opt-in the connection's credentials are not even read, so a row whose blob does
// not open stays connected, as on the base (2c083b25b, probed there); a read would have moved it
// to needs_reauthorization.
func (s *AuthorizationsSuite) TestRemovingTheAppsClientWithoutTheOptInLeavesItsConnectionsAsTheyWere() {
	connector, id := s.clientConnection(s.provider.URL+fakeprovider.PathRevoke, "")
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE connector_connections SET credentials_sealed = 'not a sealed blob' WHERE id = ?", id)
	s.Require().NoError(err)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))

	s.Equal(ConnectionStatus(store.ConnectionConnected), s.get(id).Status)
}

// A provider app's record holds no client, so removing it revokes nothing and still removes it.
func (s *AuthorizationsSuite) TestRemovingAProviderAppWithoutAClientRemovesIt() {
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, "/v1/agents/connectors/linq/oauth-client",
		ConnectorOAuthClientRequest{ProviderAppID: "line-" + s.utils.uuid(), SigningSecret: "whsec_" + s.utils.uuid()}, nil))

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/linq/oauth-client", nil, nil))
}

// A revoke the provider refuses is logged, and the client is removed all the same.
func (s *AuthorizationsSuite) TestARevokeTheProviderRefusesStillRemovesTheClient() {
	connector, id := s.clientConnection(s.provider.URL+"/no-revoke-here", optInRevoke)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))
	s.Contains(s.logged.line(s.T(), revokeFailed, id), "HTTP 404")
	s.Equal(ConnectionStatus(store.ConnectionConnected), s.get(id).Status, "nothing was revoked")
}

// usersConnection is owner's connected connection to a connector at the fake with its revoke
// endpoint at revoke; extra is more manifest YAML.
func (s *AuthorizationsSuite) usersConnection(owner *testClient, revoke, extra string) string {
	id := s.ownersConnection(s.revokingConnector("operator", revoke, extra), owner, "")
	s.Require().Equal(s.landing(id, consentConnected), s.finishAs(owner, id))
	return id
}

// clientUsersConnection is a new user's connection to connector, connected with the app's
// client clientConnection put.
func (s *AuthorizationsSuite) clientUsersConnection(connector string) string {
	bob := s.data.createUser()
	id := s.ownersConnection(connector, bob, "")
	s.Require().Equal(s.landing(id, consentConnected), s.finishAs(bob, id))
	return id
}

// assertRefusedAtTheProvider is stored's refresh token refused by the provider: its grant was
// revoked.
func (s *AuthorizationsSuite) assertRefusedAtTheProvider(stored core.StoredCredentials, manifest core.ResolvedManifest) {
	var refused *core.OutcomeError
	s.Require().ErrorAs(s.renew(stored, manifest), &refused)
	s.Equal(core.OutcomeInvalidGrant, refused.Outcome.Kind)
}

// definitionReads counts the reads of connector's definitions a pool sends while on.
type definitionReads struct {
	connector string
	on        atomic.Bool
	reads     atomic.Int64
}

func (*definitionReads) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context {
	return ctx
}

func (r *definitionReads) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	if r.on.Load() && strings.Contains(event.Query, "connector_definitions") && strings.Contains(event.Query, r.connector) {
		r.reads.Add(1)
	}
}

// clientConnection is a connector whose client is the app's own, put as the fake's, with its
// revoke endpoint at revoke, and an app-owned connection to it connected with that client.
func (s *AuthorizationsSuite) clientConnection(revoke, extra string) (connector, id string) {
	connector = s.revokingConnector("customer", revoke, extra)
	s.putClient(connector, s.provider.ClientSecret)
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(connector), &created))
	s.connect(created.ID)
	return connector, created.ID
}
