//go:build integration

package daytona

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"os"
	"testing"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ProxySuite is the sandbox Daytona is asked for when the egress proxy is the broker.
type ProxySuite struct {
	suite.Suite
	ctx      context.Context
	platform *platform
	redis    rueidis.Client
}

func TestProxySuite(t *testing.T) {
	suite.Run(t, new(ProxySuite))
}

func (s *ProxySuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	var err error
	s.redis, err = rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(s.redis.Close)
}

func (s *ProxySuite) SetupTest() {
	s.ctx = context.Background()
	s.platform = newPlatform()
	s.T().Cleanup(s.platform.server.Close)
}

// proxied is a sandbox of config whose github grant the egress proxy broker holds.
func (s *ProxySuite) proxied(config sandbox.Config) *Sandbox {
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: egress.NewStore(s.redis), URL: "http://proxy.stg.example:8091",
		CACert: []byte("-----BEGIN CERTIFICATE-----\nca\n-----END CERTIFICATE-----\n"), Session: "session-1",
		Tokens: func(context.Context, sandbox.Grant) (sandbox.Token, error) {
			return sandbox.Token{Value: "gho_first", Revision: 1}, nil
		},
		Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	box, err := New(Options{APIKey: "key-1", APIURL: s.platform.server.URL + "/api", Config: config, Grants: []sandbox.Grant{github},
		Broker: broker})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

// TestTheSandboxReachesTheProxyAlone: whatever the config opens, the sandbox's one host is
// the proxy, its HTTP(S)_PROXY, and it trusts the proxy's CA before any code runs.
func (s *ProxySuite) TestTheSandboxReachesTheProxyAlone() {
	open := s.proxied(sandbox.Config{Network: true, AllowedDomains: []string{"pypi.org"}})
	ranges := s.proxied(sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8"}})

	for _, box := range []*Sandbox{open, ranges} {
		_, err := box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
		_, err = box.Run(s.ctx, "print(2)", nil)
		s.Require().NoError(err)
	}

	for i, created := range s.platform.requests() {
		env, _ := created["env"].(map[string]any)
		s.Regexp(`^va_ph_[A-Z2-7]{26}$`, env["GITHUB_TOKEN"])
		s.Regexp(`^http://sbx:[A-Z2-7]{26}@proxy\.stg\.example:8091$`, created["outboundProxyUrl"], i)
		s.Equal(created["outboundProxyUrl"], env["HTTPS_PROXY"])
		s.Equal("proxy.stg.example", created["domainAllowList"], i)
		s.Equal(false, created["networkBlockAll"])
		s.NotContains(created, "networkAllowList")
		s.NotContains(created, "secrets")
	}
	s.Equal([]string{"/tmp/va-egress-ca.pem after 0 runs", "/tmp/va-egress-ca.pem after 2 runs"}, s.platform.uploads,
		"once per sandbox, before its first run")
	s.Equal("-----BEGIN CERTIFICATE-----\nca\n-----END CERTIFICATE-----\n", string(s.platform.files["/tmp/va-egress-ca.pem"]))
}

// TestACAFileThatCouldNotBeWrittenIsWrittenBeforeTheNextRun: no code runs without it.
func (s *ProxySuite) TestACAFileThatCouldNotBeWrittenIsWrittenBeforeTheNextRun() {
	box := s.proxied(sandbox.Config{})
	s.platform.mu.Lock()
	s.platform.refuseUpload = true
	s.platform.mu.Unlock()

	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().Error(err)
	s.platform.mu.Lock()
	s.platform.refuseUpload = false
	s.platform.mu.Unlock()
	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal([]string{"/tmp/va-egress-ca.pem after 0 runs"}, s.platform.uploads)
	s.Len(s.platform.ran, 1)
}

// TestClosingTheSandboxEndsItsProxyCredential: its record is gone, which the proxy answers
// with a 407.
func (s *ProxySuite) TestClosingTheSandboxEndsItsProxyCredential() {
	box := s.proxied(sandbox.Config{})
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	credential, _ := box.mount.Proxy.User.Password()
	sum := sha256.Sum256([]byte(credential))
	key := "sandbox:egress:" + hex.EncodeToString(sum[:])
	s.Require().EqualValues(1, s.exists(key))

	s.Require().NoError(box.Close())

	s.Zero(s.exists(key))
}

func (s *ProxySuite) exists(key string) int64 {
	exists, err := s.redis.Do(s.ctx, s.redis.B().Exists().Key(key).Build()).AsInt64()
	s.Require().NoError(err)
	return exists
}
