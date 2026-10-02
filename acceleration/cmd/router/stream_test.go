package main

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// StreamClientsSuite covers which Stream app the router starts out acting in.
type StreamClientsSuite struct {
	suite.Suite
}

func TestStreamClientsSuite(t *testing.T) {
	suite.Run(t, new(StreamClientsSuite))
}

func (s *StreamClientsSuite) TestRouterStartsWithTenancyDeployment() {
	// A router whose settings name none of the new tenancy settings acts in its own app
	// for every customer, as it always has.
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret = "deploy-key", "deploy-secret"
	s.Empty(settings.Stream.Tenancy)

	clients := newStreamClients(settings)

	for _, customer := range []string{"acme", "globex"} {
		bound, err := clients.For(context.Background(), customer)
		s.Require().NoError(err)
		s.Equal("deploy-key", bound.Identity.APIKey)
		s.Zero(bound.Identity.StreamApp)
	}
	s.IsType(&streamapp.Deployment{}, clients.Source())
}

func (s *StreamClientsSuite) TestTheDeploymentAppCarriesTheUserToken() {
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret = "deploy-key", "deploy-secret"
	settings.Stream.UserToken = "fixed-token"

	bound, err := newStreamClients(settings).For(context.Background(), "acme")

	s.Require().NoError(err)
	s.Equal("fixed-token", bound.Identity.UserToken)
}
