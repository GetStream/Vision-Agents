package main

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
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
	registry, err := newConnectorRegistry(s.settings)

	s.Require().NoError(err)
	s.Empty(registry.Schemes, "no connection or custom connector can name one")
}

func (s *ConnectorRegistrySuite) TestWithConnectorsOnTheFourSchemesAreRegisteredByName() {
	s.settings.Connectors.Enabled = true

	registry, err := newConnectorRegistry(s.settings)

	s.Require().NoError(err)
	for _, name := range []string{"oauth2_code", "api_key", "bearer", "none"} {
		s.Require().Contains(registry.Schemes, name)
		s.Equal(name, registry.Schemes[name].Name())
	}
	s.Len(registry.Schemes, 4)
}

func (s *ConnectorRegistrySuite) TestAnHTTPSPublicURLIsTheClientMetadataDocumentsHost() {
	s.settings.Connectors.Enabled = true
	s.settings.PublicURL = "https://router.example"

	registry, err := newConnectorRegistry(s.settings)

	s.Require().NoError(err, "oauth2code takes the client_id URL the API serves the document at")
	s.Contains(registry.Schemes, "oauth2_code")
}

func (s *ConnectorRegistrySuite) TestAPlainHTTPPublicURLStartsWithCIMDOff() {
	s.settings.Connectors.Enabled = true
	s.settings.PublicURL = "http://localhost:8080"

	registry, err := newConnectorRegistry(s.settings)

	s.Require().NoError(err, "an http client_id URL is not CIMD's (section 3), so none is passed")
	s.Contains(registry.Schemes, "oauth2_code")
}
