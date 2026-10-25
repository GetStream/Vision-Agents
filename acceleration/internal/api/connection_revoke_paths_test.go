//go:build integration

package api

import (
	"context"
	"errors"
	"net/http"

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
// reconnect; its grant is refused at the provider even once the same client is put back.
func (s *AuthorizationsSuite) TestRemovingTheAppsClientEndsTheGrantsIssuedToItFirst() {
	connector, id := s.clientConnection(s.provider.URL+fakeprovider.PathRevoke, optInRevoke)
	stored, manifest := s.held(id)
	before, revoked := s.provider.Hits(fakeprovider.PathRevoke), len(s.provider.Revoked())

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+connector+"/oauth-client", nil, nil))

	s.Equal(before+1, s.provider.Hits(fakeprovider.PathRevoke))
	s.Equal([]string{"refresh_token"}, s.provider.Revoked()[revoked:])
	s.Empty(s.logged.lines(revokeFailed, id))
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id, nil, nil), "the connection stays")
	s.putClient(connector, s.provider.ClientSecret)
	err := s.renew(stored, manifest)
	var refused *core.OutcomeError
	s.Require().ErrorAs(err, &refused)
	s.Equal(core.OutcomeInvalidGrant, refused.Outcome.Kind)
	var removed *oauth2code.ClientRemovedError
	s.False(errors.As(err, &removed), "the provider refused it, not the router: %v", err)
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
}

// usersConnection is owner's connected connection to a connector at the fake with its revoke
// endpoint at revoke; extra is more manifest YAML.
func (s *AuthorizationsSuite) usersConnection(owner *testClient, revoke, extra string) string {
	id := s.ownersConnection(s.revokingConnector("operator", revoke, extra), owner, "")
	s.Require().Equal(s.landing(id, consentConnected), s.finishAs(owner, id))
	return id
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
