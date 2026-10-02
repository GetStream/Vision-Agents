package main

import (
	"bytes"
	"context"
	"log/slog"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
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

func (s *StreamClientsSuite) TestDeploymentModeStartsWhenStreamIsUnreachable() {
	// Learning which app is the deployment's own happens beside startup, and a Stream that
	// cannot be reached changes nothing about what new work is written with.
	closed := httptest.NewServer(nil)
	closed.Close()
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.BaseURL = "deploy-key", "deploy-secret", closed.URL
	clients := newStreamClients(settings)
	retry := learnRetry
	learnRetry = time.Millisecond
	s.T().Cleanup(func() { learnRetry = retry })
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	go func() {
		learnDeploymentApp(ctx, clients, slog.New(slog.DiscardHandler))
		close(done)
	}()

	bound, err := clients.For(ctx, "acme")
	s.Require().NoError(err)
	s.Equal("deploy-key", bound.Identity.APIKey)
	s.Zero(bound.Identity.StreamApp)
	s.Zero(clients.DeploymentApp())
	cancel()
	s.Eventually(func() bool {
		select {
		case <-done:
			return true
		default:
			return false
		}
	}, 5*time.Second, 10*time.Millisecond, "asking again stops with the router")
}

func (s *StreamClientsSuite) TestTheDeploymentLearnsItsOwnAppAndSaysSoOnce() {
	stream := chattest.NewServer(s.T())
	stream.SetApp(chattest.App{ID: 1234})
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.BaseURL = "deploy-key", "deploy-secret", stream.URL
	clients := newStreamClients(settings)
	var logs bytes.Buffer

	learnDeploymentApp(context.Background(), clients, slog.New(slog.NewTextHandler(&logs, nil)))

	s.Equal(int64(1234), clients.DeploymentApp())
	s.Equal(1, strings.Count(logs.String(), "stream_app=1234"))
	s.NotContains(logs.String(), "deploy-secret")
	bound, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)
	s.Zero(bound.Identity.StreamApp, "deployment mode still writes no pin")
}
