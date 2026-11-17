package oauth2code_test

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// TestAnImportedGrantIsRefreshedAtTheManifestsEndpointWithThePreregisteredClient: a grant the
// fake issued is imported with no endpoint and no client in it, and its refresh still reaches
// the fake's token endpoint with the operator's client, so both came from the manifest and the
// lookup.
func (s *OAuth2CodeSuite) TestAnImportedGrantIsRefreshedAtTheManifestsEndpointWithThePreregisteredClient() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	resolved.Capture, resolved.Identity = nil, nil
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	issued, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	access, refresh := s.tokens(issued)

	stored, account, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, Supplied: map[string]string{
		oauth2code.SuppliedAccessToken:  access,
		oauth2code.SuppliedRefreshToken: refresh,
		oauth2code.SuppliedExpiresAt:    time.Now().Add(-time.Minute).Format(time.RFC3339),
		oauth2code.SuppliedScope:        "chat:write,channels:history",
	}})
	s.Require().NoError(err)
	s.Equal([]string{"chat:write", "channels:history"}, account.Scopes)
	s.Empty(account.AccountID, "nothing is captured from an import")

	_, renewed, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})

	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(renewed)))
}

func (s *OAuth2CodeSuite) TestAnImportNamingAnEndpointIsRefusedWithoutQuotingIt() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, Supplied: map[string]string{
		oauth2code.SuppliedAccessToken: "imported-access",
		oauth2code.SuppliedExpiresAt:   time.Now().Add(time.Hour).Format(time.RFC3339),
		oauth2code.SuppliedScope:       "chat:write",
		"token_endpoint":               "https://attacker.example/token",
	}})

	s.ErrorContains(err, `"token_endpoint" is not a value an imported grant takes`)
	s.NotContains(err.Error(), "attacker.example")
	s.NotContains(err.Error(), "imported-access")
}

func (s *OAuth2CodeSuite) TestAnImportedScopeTheConnectorNeverAsksForIsRefused() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, Supplied: map[string]string{
		oauth2code.SuppliedAccessToken: "imported-access",
		oauth2code.SuppliedExpiresAt:   time.Now().Add(time.Hour).Format(time.RFC3339),
		oauth2code.SuppliedScope:       "admin",
	}})

	s.ErrorContains(err, `scope "admin" is not one this connector asks for`)
}

// TestAnImportWithNoPreregisteredClientIsRefused: the manifest would let a consent register a
// client, but a client registered now is not the one the grant was issued to.
func (s *OAuth2CodeSuite) TestAnImportWithNoPreregisteredClientIsRefused() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	resolved.Client.Registration = []core.ClientRegistrationMethod{core.ClientDCR}
	// A manifest that registers clients discovers the server through its MCP endpoint.
	resolved.Endpoints["mcp"] = srv.URL + fakeprovider.PathMCP
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, Supplied: map[string]string{
		oauth2code.SuppliedAccessToken: "imported-access",
		oauth2code.SuppliedExpiresAt:   time.Now().Add(time.Hour).Format(time.RFC3339),
		oauth2code.SuppliedScope:       "chat:write",
	}})

	s.ErrorIs(err, oauth2code.ErrNoClient)
}

// tokens reads the access and refresh tokens out of stored credentials, as the provider issued
// them.
func (s *OAuth2CodeSuite) tokens(stored core.StoredCredentials) (access, refresh string) {
	var payload struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
	}
	s.Require().NoError(json.Unmarshal(stored.Payload, &payload))
	s.Require().NotEmpty(payload.RefreshToken)
	return payload.AccessToken, payload.RefreshToken
}
