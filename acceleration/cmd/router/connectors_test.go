package main

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// ConnectorRegistrySuite covers which connector adapters the router starts with.
type ConnectorRegistrySuite struct {
	suite.Suite
	settings config.Config
}

func TestConnectorRegistrySuite(t *testing.T) {
	suite.Run(t, new(ConnectorRegistrySuite))
}

func (s *ConnectorRegistrySuite) SetupTest() {
	s.settings = config.Defaults()
	s.settings.Auth.Mode = string(auth.Proxy)
}

func (s *ConnectorRegistrySuite) TestWithConnectorsOffNoSchemeIsRegistered() {
	registry, err := newConnectorRegistry(s.settings, nil)

	s.Require().NoError(err)
	s.Empty(registry.Schemes, "no connection or custom connector can name one")
}

func (s *ConnectorRegistrySuite) TestWithConnectorsOnTheFourSchemesAreRegisteredByName() {
	s.settings.Connectors.Enabled = true

	registry, err := newConnectorRegistry(s.settings, nil)

	s.Require().NoError(err)
	for _, name := range []string{"oauth2_code", "api_key", "bearer", "none"} {
		s.Require().Contains(registry.Schemes, name)
		s.Equal(name, registry.Schemes[name].Name())
	}
	s.Len(registry.Schemes, 4)
}

// TestAnHTTPSPublicURLIsTheClientIDUnderCIMD pins the client_id the router hands an
// authorization server that supports CIMD: the URL the API serves the client metadata
// document at under that public URL (api.ConnectorClientMetadataPath).
func (s *ConnectorRegistrySuite) TestAnHTTPSPublicURLIsTheClientIDUnderCIMD() {
	s.settings.PublicURL = "https://router.example"

	out, err := s.beginWithCIMDOnly()

	s.Require().NoError(err)
	authorize, err := url.Parse(out.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal("https://router.example/.well-known/oauth-client-metadata", authorize.Query().Get("client_id"))
}

// TestTheOperatorsClientComesFromTheEnvironment pins the lookup the router starts oauth2_code
// with: an operator client is <client.env>_MCP_CLIENT_ID and _SECRET, read when a consent
// begins, with no database needed for it.
func (s *ConnectorRegistrySuite) TestTheOperatorsClientComesFromTheEnvironment() {
	srv := fakeprovider.New(s.T())
	environment := map[string]string{"FAKE_MCP_CLIENT_ID": srv.ClientID, "FAKE_MCP_CLIENT_SECRET": srv.ClientSecret}
	cfg := connectorSchemeConfig(s.settings, api.ConnectorClients(nil, nil, func(name string) string { return environment[name] }))

	out, err := s.begin(srv, cfg, core.ClientPolicy{
		Registration: []core.ClientRegistrationMethod{core.ClientOperator},
		Env:          "FAKE",
	})

	s.Require().NoError(err)
	authorize, err := url.Parse(out.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal(srv.ClientID, authorize.Query().Get("client_id"))
}

func (s *ConnectorRegistrySuite) TestAPlainHTTPPublicURLStartsWithCIMDOff() {
	s.settings.PublicURL = "http://localhost:8080"

	_, err := s.beginWithCIMDOnly()

	s.ErrorIs(err, oauth2code.ErrNoClient, "an http client_id URL is not CIMD's (section 3), so none is passed")
}

// beginWithCIMDOnly starts a consent with the scheme the router builds from s.settings,
// against a fake authorization server that supports CIMD, for a manifest whose only
// client.registration is cimd.
func (s *ConnectorRegistrySuite) beginWithCIMDOnly() (core.BeginOutput, error) {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientMetadataDocuments)
	return s.begin(srv, connectorSchemeConfig(s.settings, nil),
		core.ClientPolicy{Registration: []core.ClientRegistrationMethod{core.ClientCIMD}})
}

// begin starts a consent at srv with cfg for a manifest whose client is client. HTTP and
// PublicEndpoint point at the fake, since egress refuses its loopback address.
func (s *ConnectorRegistrySuite) begin(srv *fakeprovider.Server, cfg oauth2code.Config, client core.ClientPolicy) (core.BeginOutput, error) {
	cfg.HTTP = srv.Client()
	cfg.PublicEndpoint = func(_ context.Context, raw string) error {
		if !strings.HasPrefix(raw, srv.URL+"/") {
			return fmt.Errorf("%s is not the fake provider", raw)
		}
		return nil
	}
	scheme, err := oauth2code.New(cfg)
	s.Require().NoError(err)
	return scheme.Begin(context.Background(), core.BeginInput{
		Ref: core.ConnectionRef{CustomerID: "acme", ConnectionID: "conn-1"},
		Manifest: core.ResolvedManifest{
			Scheme:    oauth2code.Name,
			Endpoints: map[string]string{"mcp": srv.URL + fakeprovider.PathMCP},
			Client:    client,
		},
		RedirectURI: fakeprovider.RedirectURI,
	})
}
