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

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// StreamClientsSuite covers which Stream app the router starts out acting in.
type StreamClientsSuite struct {
	suite.Suite
}

func TestStreamClientsSuite(t *testing.T) {
	suite.Run(t, new(StreamClientsSuite))
}

// clients builds a deployment's Stream clients with no store or keyring, which is all
// deployment mode needs.
func (s *StreamClientsSuite) clients(settings config.Config) *streamapp.Clients {
	clients, err := newStreamClients(settings, nil, nil, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	return clients
}

func (s *StreamClientsSuite) TestRouterStartsWithTenancyDeployment() {
	// A router whose settings name none of the new tenancy settings acts in its own app
	// for every customer, as it always has.
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret = "deploy-key", "deploy-secret"
	s.Empty(settings.Stream.Tenancy)

	clients := s.clients(settings)

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

	bound, err := s.clients(settings).For(context.Background(), "acme")

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
	clients := s.clients(settings)
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
	clients := s.clients(settings)
	var logs bytes.Buffer

	learnDeploymentApp(context.Background(), clients, slog.New(slog.NewTextHandler(&logs, nil)))

	s.Equal(int64(1234), clients.DeploymentApp())
	s.Equal(1, strings.Count(logs.String(), "stream_app=1234"))
	s.NotContains(logs.String(), "deploy-secret")
	bound, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)
	s.Zero(bound.Identity.StreamApp, "deployment mode still writes no pin")
}

func (s *StreamClientsSuite) TestAppModeRefusesToStartWithoutAKeyring() {
	settings := config.Defaults()
	settings.Stream.Tenancy = config.TenancyApp

	_, err := newStreamClients(settings, &store.Store{}, nil, slog.New(slog.DiscardHandler))

	s.ErrorContains(err, "keyring")
}

func (s *StreamClientsSuite) TestAppModeRefusesToStartWithoutPostgres() {
	settings := config.Defaults()
	settings.Stream.Tenancy = config.TenancyApp

	_, err := newStreamClients(settings, nil, nil, slog.New(slog.DiscardHandler))

	s.ErrorContains(err, "postgres.dsn")
}

func (s *StreamClientsSuite) TestAMismatchedDeploymentAppRefusesToStart() {
	// Every pin app mode writes would name an app the deployment's key is not.
	stream := chattest.NewServer(s.T())
	stream.SetApp(chattest.App{ID: 1234})
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.BaseURL = "deploy-key", "deploy-secret", stream.URL
	settings.Stream.Tenancy, settings.Stream.AppID = config.TenancyApp, 99
	sealer, err := auth.NewSealer("first-key")
	s.Require().NoError(err)
	clients, err := newStreamClients(settings, &store.Store{}, sealer, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	err = checkDeploymentApp(context.Background(), settings, clients)

	s.ErrorIs(err, streamapp.ErrDeploymentAppMismatch)
}

func (s *StreamClientsSuite) TestAppModeStartsWhenStreamCannotBeReached() {
	// The id is checked beside the router once Stream answers; until then the deployment's
	// work waits rather than the router refusing to start.
	closed := httptest.NewServer(nil)
	closed.Close()
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.BaseURL = "deploy-key", "deploy-secret", closed.URL
	settings.Stream.Tenancy, settings.Stream.AppID = config.TenancyApp, 99
	sealer, err := auth.NewSealer("first-key")
	s.Require().NoError(err)
	clients, err := newStreamClients(settings, &store.Store{}, sealer, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	s.NoError(checkDeploymentApp(context.Background(), settings, clients))
	s.True(clients.PerApp())
}

func (s *StreamClientsSuite) TestDeploymentModeServesPinsEqualToAConfiguredAppID() {
	settings := config.Defaults()
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.AppID = "deploy-key", "deploy-secret", 1234
	clients := s.clients(settings)

	bound, err := clients.ForApp(context.Background(), "acme", 1234)

	s.Require().NoError(err)
	s.Equal("deploy-key", bound.Identity.APIKey)
	_, err = clients.ForApp(context.Background(), "acme", 4242)
	s.ErrorIs(err, streamapp.ErrStreamAppMoved, "a pin naming another app is parked")
	s.False(clients.PerApp())
}
