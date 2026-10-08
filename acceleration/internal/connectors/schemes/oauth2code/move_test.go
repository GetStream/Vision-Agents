package oauth2code_test

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// TestAMovedGrantIsRefreshedWithThePreregisteredClientItWasIssuedTo: a plugin login's grant,
// moved with its client id and the token endpoint the plugin renewed it at, is renewed by
// this scheme at the manifest's endpoint with the operator's client, its secret looked up.
func (s *OAuth2CodeSuite) TestAMovedGrantIsRefreshedWithThePreregisteredClientItWasIssuedTo() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/github.yaml", nil))
	resolved.Capture, resolved.Identity = nil, nil
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	issued, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	access, refresh := s.tokens(issued)

	stored, account, err := scheme.MoveGrant(s.ctx, s.ref, resolved, oauth2code.MovedGrant{
		AccessToken: access, RefreshToken: refresh, ExpiresAt: time.Now().Add(-time.Minute),
		ClientID: srv.ClientID, TokenEndpoint: srv.URL + fakeprovider.PathToken, Scopes: []string{"repo"},
	})
	s.Require().NoError(err)
	s.Equal([]string{"repo"}, account.Scopes)
	s.Equal(core.ClientOperator, s.movedClient(stored).Owner)

	_, renewed, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})

	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(renewed)))
}

// TestAMovedGrantOfARegisteredClientKeepsThatClient: the plugin registered a public client of
// its own; the moved grant keeps it, registers none, and still renews.
func (s *OAuth2CodeSuite) TestAMovedGrantOfARegisteredClientKeepsThatClient() {
	srv := fakeprovider.New(s.T())
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})
	issued, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	access, refresh := s.tokens(issued)
	registered := s.movedClient(issued).ID

	stored, _, err := scheme.MoveGrant(s.ctx, s.ref, resolved, oauth2code.MovedGrant{
		AccessToken: access, RefreshToken: refresh, ExpiresAt: time.Now().Add(-time.Minute),
		ClientID: registered, TokenEndpoint: srv.URL + fakeprovider.PathToken, MaybeRegistered: true,
	})
	s.Require().NoError(err)
	moved := s.movedClient(stored)
	s.Equal(core.ClientDCR, moved.Owner)
	s.Equal(registered, moved.ID)
	s.Equal(core.AuthNone, moved.AuthMethod)

	_, renewed, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})

	s.Require().NoError(err)
	s.Equal(1, srv.Hits(fakeprovider.PathRegister), "the consent registered one; the move none")
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(renewed)))
}

// TestAMovedTokenThatDoesNotExpireIsUsedAsItIs: the plugin stored no expiry, and the moved
// grant has none, so it is used without a refresh.
func (s *OAuth2CodeSuite) TestAMovedTokenThatDoesNotExpireIsUsedAsItIs() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/github.yaml", nil))
	resolved.Capture, resolved.Identity = nil, nil
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	issued, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	access, _ := s.tokens(issued)

	stored, _, err := scheme.MoveGrant(s.ctx, s.ref, resolved, oauth2code.MovedGrant{
		AccessToken: access, ClientID: srv.ClientID, TokenEndpoint: srv.URL + fakeprovider.PathToken,
	})
	s.Require().NoError(err)
	_, kept, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})

	s.Require().NoError(err)
	s.Equal(0, srv.Refreshes())
	s.Equal(access, s.accessToken(kept))
}

// TestAGrantRenewedAtAnotherEndpointIsNotMoved: the token endpoint comes from the manifest,
// and a grant the plugin renewed elsewhere would fail at its first refresh here.
func (s *OAuth2CodeSuite) TestAGrantRenewedAtAnotherEndpointIsNotMoved() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/github.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, _, err := scheme.MoveGrant(s.ctx, s.ref, resolved, oauth2code.MovedGrant{
		AccessToken: "plugin-access", ClientID: srv.ClientID, TokenEndpoint: "https://elsewhere.example/token",
	})

	s.ErrorIs(err, oauth2code.ErrGrantElsewhere)
	s.NotContains(err.Error(), "plugin-access")
}

// TestAGrantOfAnotherClientIsNotMovedWhereNoneIsRegistered: the app's client for the connector
// is not the one the grant was issued to, and the manifest registers none, so no client could
// renew it.
func (s *OAuth2CodeSuite) TestAGrantOfAnotherClientIsNotMovedWhereNoneIsRegistered() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/github.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, _, err := scheme.MoveGrant(s.ctx, s.ref, resolved, oauth2code.MovedGrant{
		AccessToken: "plugin-access", ClientID: "an-older-client", TokenEndpoint: srv.URL + fakeprovider.PathToken,
	})

	s.ErrorIs(err, oauth2code.ErrGrantElsewhere)
	s.NotContains(err.Error(), "plugin-access")
}

// TestAClientThePluginCannotHaveRegisteredIsNotTakenAsRegistered: the manifest allows dcr, but
// the grant was issued to a client set in advance that the app no longer has, so no client
// could renew it.
func (s *OAuth2CodeSuite) TestAClientThePluginCannotHaveRegisteredIsNotTakenAsRegistered() {
	srv := fakeprovider.New(s.T())
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	_, _, err := scheme.MoveGrant(s.ctx, s.ref, resolved, oauth2code.MovedGrant{
		AccessToken: "plugin-access", ClientID: "an-older-client", TokenEndpoint: srv.URL + fakeprovider.PathToken,
	})

	s.ErrorIs(err, oauth2code.ErrGrantElsewhere)
	s.Equal(0, srv.Hits(fakeprovider.PathRegister))
}

// movedClient reads the client out of stored credentials.
func (s *OAuth2CodeSuite) movedClient(stored core.StoredCredentials) struct {
	Owner      core.ClientRegistrationMethod `json:"owner"`
	ID         string                        `json:"id"`
	AuthMethod core.ClientAuthMethod         `json:"auth_method"`
} {
	var payload struct {
		Client struct {
			Owner      core.ClientRegistrationMethod `json:"owner"`
			ID         string                        `json:"id"`
			AuthMethod core.ClientAuthMethod         `json:"auth_method"`
		} `json:"client"`
	}
	s.Require().NoError(json.Unmarshal(stored.Payload, &payload))
	return payload.Client
}
