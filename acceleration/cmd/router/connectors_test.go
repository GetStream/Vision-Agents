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

func (s *ConnectorRegistrySuite) TestWithConnectorsOnOAuth2CodeIsRegistered() {
	s.settings.Connectors.Enabled = true

	registry, err := newConnectorRegistry(s.settings)

	s.Require().NoError(err)
	s.Require().Contains(registry.Schemes, "oauth2_code")
	s.Equal("oauth2_code", registry.Schemes["oauth2_code"].Name())
	s.Len(registry.Schemes, 1)
}
