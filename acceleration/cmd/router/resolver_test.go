package main

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorResolverSuite covers when the router builds the credential resolver.
type ConnectorResolverSuite struct {
	suite.Suite
	settings config.Config
}

func TestConnectorResolverSuite(t *testing.T) {
	suite.Run(t, new(ConnectorResolverSuite))
}

func (s *ConnectorResolverSuite) SetupTest() {
	s.settings = config.Defaults()
	s.settings.Auth.Mode = string(auth.Proxy)
}

func (s *ConnectorResolverSuite) TestWithConnectorsOffThereIsNoResolver() {
	sealer, err := newSecretSealer(s.settings)
	s.Require().NoError(err)
	registry, err := newConnectorRegistry(s.settings, nil)
	s.Require().NoError(err)

	built, err := newConnectorResolver(registry, s.store(), sealer)
	s.Require().NoError(err)
	s.Nil(built)
}

func (s *ConnectorResolverSuite) TestWithoutADatabaseThereIsNoResolver() {
	built, err := newConnectorResolver(s.connectorsOn(), nil, s.sealer())
	s.Require().NoError(err)
	s.Nil(built)
}

func (s *ConnectorResolverSuite) TestWithConnectorsOnAndADatabaseTheResolverIsBuilt() {
	built, err := newConnectorResolver(s.connectorsOn(), s.store(), s.sealer())
	s.Require().NoError(err)
	s.NotNil(built)
}

// TestWithConnectorsOffThereIsNoLimiterAndNothingIsLogged: a deployment without connectors
// starts as it did before the limiter, with no new warning.
func (s *ConnectorResolverSuite) TestWithConnectorsOffThereIsNoLimiterAndNothingIsLogged() {
	var logged bytes.Buffer

	limiter := newConnectorLimiter(nil, nil, slog.New(slog.NewTextHandler(&logged, nil)))

	s.Nil(limiter)
	s.Empty(logged.String())
}

// TestWithConnectorsOnAndNoRedisThereIsNoLimiterAndOneWarning: as for the daily limits,
// nothing is limited, and the operator is told once (Kanat, 2026-10-07, D7).
func (s *ConnectorResolverSuite) TestWithConnectorsOnAndNoRedisThereIsNoLimiterAndOneWarning() {
	resolver, err := newConnectorResolver(s.connectorsOn(), s.store(), s.sealer())
	s.Require().NoError(err)
	transports, err := newConnectorTransports(resolver)
	s.Require().NoError(err)
	var logged bytes.Buffer

	limiter := newConnectorLimiter(transports, nil, slog.New(slog.NewTextHandler(&logged, nil)))

	s.Nil(limiter)
	s.Equal(1, strings.Count(logged.String(), "level=WARN"))
	s.Contains(logged.String(), "connector calls will not wait out a provider's rate limit")
}

func (s *ConnectorResolverSuite) connectorsOn() core.Registry {
	s.settings.Connectors.Enabled = true
	registry, err := newConnectorRegistry(s.settings, nil)
	s.Require().NoError(err)
	return registry
}

func (s *ConnectorResolverSuite) sealer() *auth.Sealer {
	sealer, err := auth.NewSealerWithKeyring(1, map[int]string{1: "resolver wiring test key"})
	s.Require().NoError(err)
	return sealer
}

// store is a store that never connects: opening a pool dials nothing until a query runs.
func (s *ConnectorResolverSuite) store() *store.Store {
	db, err := store.Open("postgres://postgres@127.0.0.1:1/unused_test?sslmode=disable")
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = db.Close() })
	return db
}
