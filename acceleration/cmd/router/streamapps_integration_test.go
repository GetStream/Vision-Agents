//go:build integration

package main

import (
	"bytes"
	"context"
	"log/slog"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// StreamAppsCLISuite runs the stream-apps commands against Postgres and a Stream in memory,
// where the deployment's key answers as app 1.
type StreamAppsCLISuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	stream   *chattest.Server
	store    *store.Store
	customer string
}

func TestStreamAppsCLISuite(t *testing.T) {
	suite.Run(t, new(StreamAppsCLISuite))
}

func (s *StreamAppsCLISuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "router_cli")
	opened, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *StreamAppsCLISuite) SetupTest() {
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.stream = chattest.NewServer(s.T())
	s.stream.SetAppFor("deploy-key", chattest.App{ID: 1})
	s.customer = strconv.FormatInt(1_000_000_000+rand.Int64N(8_000_000_000), 10)
	s.settings.Stream = config.Stream{
		APIKey: "deploy-key", APISecret: "deploy-secret", BaseURL: s.stream.URL,
		Tenancy: config.TenancyApp, AppID: 1, MintURL: "http://gateway-internal:8082", MintToken: "the-routers-bearer",
	}
}

func (s *StreamAppsCLISuite) TestLegacyCountsOtherCustomersWorkInTheDeploymentApp() {
	other := "someone-" + s.customer
	s.Require().NoError(s.store.SaveSession(s.ctx, &store.AgentSession{
		ID: uuid.NewString(), CustomerID: other, AgentID: "agent", UserID: "user",
	}))
	s.Require().NoError(s.store.SaveSession(s.ctx, &store.AgentSession{
		ID: uuid.NewString(), CustomerID: "1", AgentID: "agent", UserID: "user",
	}))
	var out bytes.Buffer
	s.Require().NoError(runLegacy(s.ctx, []string{"--by-customer"}, s.settings, slog.New(slog.DiscardHandler), &out))

	s.Contains(out.String(), other)
	for _, line := range strings.Split(out.String(), "\n") {
		fields := strings.Fields(line)
		// A row of the table is a kind, a customer and a count.
		s.False(len(fields) == 3 && fields[1] == "1", "the deployment's own customer's work is its own: %q", line)
	}
}
