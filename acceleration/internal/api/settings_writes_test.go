package api

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// WritesIntoSuite covers how the calling app is told which Stream app its work goes into,
// which must never tell it the router's own app is its own unless it is.
type WritesIntoSuite struct {
	suite.Suite
}

func TestWritesIntoSuite(t *testing.T) {
	suite.Run(t, new(WritesIntoSuite))
}

func (s *WritesIntoSuite) server(options streamapp.DeploymentOptions) *Server {
	return &Server{stream: streamapp.NewClients(streamapp.NewDeployment(options), streamapp.ClientsOptions{})}
}

func (s *WritesIntoSuite) TestTheRoutersOwnAppIsSharedToEveryOtherApp() {
	server := s.server(streamapp.DeploymentOptions{APIKey: "key", Secret: "secret", App: 77})

	s.Equal(WritesIntoDeploymentApp, server.writesInto("acme", streamapp.Identity{}))
	s.Equal(WritesIntoDeploymentApp, server.writesInto("acme", streamapp.Identity{StreamApp: 77}))
}

func (s *WritesIntoSuite) TestTheRoutersOwnAppIsItsOwnAppsOwn() {
	server := s.server(streamapp.DeploymentOptions{APIKey: "key", Secret: "secret", App: 77})

	s.Equal(WritesIntoThisApp, server.writesInto("77", streamapp.Identity{}))
	s.Equal(WritesIntoDeploymentApp, server.writesInto("077", streamapp.Identity{}),
		"only the id exactly as Stream writes it is that app")
}

func (s *WritesIntoSuite) TestAnAppOfItsOwnIsItsOwn() {
	server := s.server(streamapp.DeploymentOptions{APIKey: "key", Secret: "secret", App: 77})

	s.Equal(WritesIntoThisApp, server.writesInto("acme", streamapp.Identity{StreamApp: 4242}))
}

func (s *WritesIntoSuite) TestADeploymentWithNoStreamAppWritesNowhere() {
	server := s.server(streamapp.DeploymentOptions{})
	ctx := context.WithValue(context.Background(), customerContextKey{}, "acme")

	read, err := server.getAppSettings(ctx, nil)

	s.Require().NoError(err)
	s.Equal(WritesIntoNowhere, read.Body.Stream.WritesInto)
	s.Equal(StreamTypeState(streamapp.TypeUnknown), read.Body.Stream.ChannelType)
}
