//go:build integration

package main

import (
	"bytes"
	"context"
	"log/slog"
	"math/rand/v2"
	"os"
	"path/filepath"
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
// where the deployment's key answers as app 1 and each test's customer's key as its own.
type StreamAppsCLISuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	stream   *chattest.Server
	store    *store.Store
	customer string
	key      string
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
	app := 1_000_000_000 + rand.Int64N(8_000_000_000)
	s.customer, s.key = strconv.FormatInt(app, 10), "own-key-"+strconv.FormatInt(app, 10)
	s.stream.SetAppFor(s.key, chattest.App{ID: app})
	s.settings.Stream = config.Stream{
		APIKey: "deploy-key", APISecret: "deploy-secret", BaseURL: s.stream.URL,
		Tenancy: config.TenancyApp, AppID: 1,
	}
}

func (s *StreamAppsCLISuite) secretFile(secret string) string {
	path := filepath.Join(s.T().TempDir(), "secret")
	s.Require().NoError(os.WriteFile(path, []byte(secret+"\n"), 0o600))
	return path
}

func (s *StreamAppsCLISuite) register(args ...string) (string, error) {
	var out bytes.Buffer
	err := runRegister(s.ctx, args, s.settings, slog.New(slog.DiscardHandler), &bytes.Buffer{}, &out)
	return out.String(), err
}

func (s *StreamAppsCLISuite) TestRegisterFromTheCommandLineVerifiesWithStream() {
	out, err := s.register("--customer", s.customer, s.key+"="+s.secretFile("a-long-stream-secret"))

	s.Require().NoError(err)
	s.Contains(out, s.customer)
	app, err := s.store.StreamApp(s.ctx, s.customer)
	s.Require().NoError(err)
	s.Equal(s.customer, strconv.FormatInt(app.StreamAppPK, 10), "the app id is the one Stream gave")
	s.Equal(s.key, app.PrimaryKey)
}

func (s *StreamAppsCLISuite) TestAKeyOfAnotherAppIsRefusedWithoutTheOperatorNamingIt() {
	_, err := s.register("--customer", "someone-"+s.customer, s.key+"="+s.secretFile("a-long-stream-secret"))

	s.ErrorContains(err, "belongs to another Stream app")
}

func (s *StreamAppsCLISuite) TestTheOperatorMayNameTheStreamAppOfAnAppKeyedCustomer() {
	// An api_key-mode customer's id is not a Stream app id at all.
	id := "app-" + s.customer
	_, err := s.register("--customer", id, "--stream-app", s.customer, s.key+"="+s.secretFile("a-long-stream-secret"))

	s.Require().NoError(err)
	app, err := s.store.StreamApp(s.ctx, id)
	s.Require().NoError(err)
	s.Equal(s.customer, strconv.FormatInt(app.StreamAppPK, 10))
}

func (s *StreamAppsCLISuite) TestTheDeploymentAppCannotBeBoundToAnotherCustomer() {
	s.stream.SetAppFor("second-deploy-key", chattest.App{ID: 1})

	_, err := s.register("--customer", s.customer, "--stream-app", "1", "second-deploy-key="+s.secretFile("a-long-stream-secret"))

	s.ErrorContains(err, "router's own Stream app")
}

func (s *StreamAppsCLISuite) TestForgetRemovesTheTombstone() {
	_, err := s.register("--customer", s.customer, s.key+"="+s.secretFile("a-long-stream-secret"))
	s.Require().NoError(err)
	_, err = s.store.DisconnectStreamApp(s.ctx, s.customer, nil, "test")
	s.Require().NoError(err)

	var out bytes.Buffer
	s.Require().NoError(runForget(s.ctx, []string{"--customer", s.customer}, s.settings, slog.New(slog.DiscardHandler), &out))

	_, err = s.store.StreamApp(s.ctx, s.customer)
	s.ErrorIs(err, store.ErrNoStreamApp)
}

func (s *StreamAppsCLISuite) TestListShowsNoSecret() {
	_, err := s.register("--customer", s.customer, s.key+"="+s.secretFile("a-long-stream-secret"))
	s.Require().NoError(err)
	pgStore, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	defer pgStore.Close()

	var out bytes.Buffer
	s.Require().NoError(listStreamApps(s.ctx, pgStore, &out))

	s.Contains(out.String(), s.key)
	s.NotContains(out.String(), "a-long-stream-secret")
}

func (s *StreamAppsCLISuite) TestLegacyCountsOtherCustomersWorkInTheDeploymentApp() {
	other := "someone-" + s.customer
	s.Require().NoError(s.store.SaveSession(s.ctx, &store.AgentSession{
		ID: uuid.NewString(), CustomerID: other, AgentID: "agent", UserID: "user",
	}))
	s.Require().NoError(s.store.SaveSession(s.ctx, &store.AgentSession{
		ID: uuid.NewString(), CustomerID: "1", AgentID: "agent", UserID: "user",
	}))
	outbox := s.T().TempDir()
	s.Require().NoError(os.MkdirAll(filepath.Join(outbox, "support-"+uuid.NewString()), 0o700))
	record := filepath.Join(outbox, "support-"+uuid.NewString())
	s.Require().NoError(os.MkdirAll(record, 0o700))
	s.Require().NoError(os.WriteFile(filepath.Join(record, "state.json"),
		[]byte(`{"outbox_version": 1, "customer": "`+other+`"}`), 0o600))

	var out bytes.Buffer
	s.Require().NoError(runLegacy(s.ctx, []string{"--by-customer"}, s.settings, slog.New(slog.DiscardHandler), outbox, &out))

	s.Contains(out.String(), other)
	for _, line := range strings.Split(out.String(), "\n") {
		fields := strings.Fields(line)
		// A row of the table is a kind, a customer and a count.
		s.False(len(fields) == 3 && fields[1] == "1", "the deployment's own customer's work is its own: %q", line)
	}
}
