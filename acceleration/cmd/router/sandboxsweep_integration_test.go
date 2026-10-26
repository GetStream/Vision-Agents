//go:build integration

package main

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// SecretSweepSuite is where a router sweeps the Daytona secrets a crashed router left behind:
// under the daytona broker alone, with Daytona's API served by the suite.
type SecretSweepSuite struct {
	suite.Suite
	redis     *live.Client
	directory *node.Directory
	daytona   *httptest.Server
	logger    *slog.Logger

	mu sync.Mutex
	// asked are the requests Daytona got, each its path and the name it lists.
	asked []string
}

func TestSecretSweepSuite(t *testing.T) {
	suite.Run(t, new(SecretSweepSuite))
}

func (s *SecretSweepSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.redis, s.logger = redis, slog.New(slog.DiscardHandler)
	s.directory, err = node.NewDirectory(node.DirectoryOptions{Redis: redis.Redis(), Address: "10.0.0.1:1", Prefix: "w15sweep:router",
		Logger: s.logger})
	s.Require().NoError(err)
}

func (s *SecretSweepSuite) TearDownSuite() {
	if s.redis != nil {
		s.directory.Close()
		s.redis.Close()
	}
}

func (s *SecretSweepSuite) SetupTest() {
	s.asked = nil
	s.daytona = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.asked = append(s.asked, r.URL.Path+" "+r.URL.Query().Get("name"))
		s.mu.Unlock()
		// The SDK's client also opens Daytona's event socket, which the sweep never uses.
		if r.URL.Path != "/api/secret/paginated" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"items":[],"total":0,"nextCursor":null}`))
	}))
	s.T().Cleanup(s.daytona.Close)
	s.T().Setenv("DAYTONA_API_KEY", "key-1")
	s.T().Setenv("DAYTONA_API_URL", s.daytona.URL+"/api")
}

func (s *SecretSweepSuite) TearDownTest() {
	ctx := context.Background()
	s.Require().NoError(s.redis.Redis().Do(ctx, s.redis.Redis().B().Del().Key("sandbox:secret-sweep:w15sweep").Build()).Error())
}

// swept is what Daytona was asked for by the sweep secretSweeper gives settings, if any.
func (s *SecretSweepSuite) swept(settings config.CodeSandbox, redis *live.Client, directory *node.Directory) []string {
	sweeper, err := secretSweeper(settings, redis, directory, s.logger)
	s.Require().NoError(err)
	if sweeper != nil {
		s.Require().NoError(sweeper.Sweep(context.Background()))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.asked
}

// TestADefaultRouterSweepsNoSecrets is the control: on base e5d6d8057 nothing in the router
// lists Daytona's secrets (`git grep -n 'Secret.List' e5d6d8057 -- acceleration` finds none),
// so Daytona is asked nothing.
func (s *SecretSweepSuite) TestADefaultRouterSweepsNoSecrets() {
	s.Empty(s.swept(config.Defaults().CodeSandbox, s.redis, s.directory))
}

func (s *SecretSweepSuite) TestTheDaytonaBrokerSweepsItsOwnPrefix() {
	settings := config.Defaults().CodeSandbox
	settings.SecretPrefix = "w15sweep"

	s.Contains(s.swept(settings, s.redis, s.directory), "/api/secret/paginated va_w15sweep_")
}

func (s *SecretSweepSuite) TestTheProxyBrokerSweepsNoSecrets() {
	settings := config.Defaults().CodeSandbox
	settings.Broker, settings.SecretPrefix = config.CodeSandboxBrokerProxy, "w15sweep"

	s.Empty(s.swept(settings, s.redis, s.directory))
}

func (s *SecretSweepSuite) TestARouterWithoutDaytonaSweepsNoSecrets() {
	s.T().Setenv("DAYTONA_API_KEY", "")
	settings := config.Defaults().CodeSandbox
	settings.SecretPrefix = "w15sweep"

	s.Empty(s.swept(settings, s.redis, s.directory))
}

// TestARouterWithoutRedisSweepsNoSecrets: only the node directory says whether a session runs
// on another router.
func (s *SecretSweepSuite) TestARouterWithoutRedisSweepsNoSecrets() {
	settings := config.Defaults().CodeSandbox
	settings.SecretPrefix = "w15sweep"

	s.Empty(s.swept(settings, nil, nil))
}
