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

// proxied is a sandbox of config whose github grant the egress proxy broker at proxy holds.
func (s *ProxySuite) proxied(config sandbox.Config, proxy string) *Sandbox {
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: egress.NewStore(s.redis), URL: proxy,
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

// TestTheSandboxReachesTheProxyAndTheConfigsOwnHosts: the proxy joins the config's allow list,
// HTTP(S)_PROXY names it, the config's own hosts go around it (NO_PROXY) and a grant's host
// never does, and the sandbox trusts the proxy's CA before any code runs.
func (s *ProxySuite) TestTheSandboxReachesTheProxyAndTheConfigsOwnHosts() {
	const proxy = "http://proxy.stg.example:8091"
	cases := []struct {
		config                     sandbox.Config
		proxy                      string
		domains, ranges, noProxied string
	}{
		{sandbox.Config{}, proxy, "proxy.stg.example", "", ""},
		{sandbox.Config{}, "http://203.0.113.7:8091", "203.0.113.7", "", ""},
		{sandbox.Config{Network: true, AllowedDomains: []string{"pypi.org", "api.github.com", "github.com", "*.pythonhosted.org"}}, proxy,
			"proxy.stg.example,pypi.org,api.github.com,github.com,*.pythonhosted.org", "", "pypi.org,.pythonhosted.org"},
		{sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8"}}, "http://203.0.113.7:8091", "", "10.0.0.0/8,203.0.113.7/32",
			"10.0.0.0/8"},
		{sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8"}}, "http://[2001:db8::7]:8091", "", "10.0.0.0/8,2001:db8::7/128",
			"10.0.0.0/8"},
	}
	for _, c := range cases {
		box := s.proxied(c.config, c.proxy)
		_, err := box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
		_, err = box.Run(s.ctx, "print(2)", nil)
		s.Require().NoError(err)
	}

	for i, created := range s.platform.requests() {
		c := cases[i]
		env, _ := created["env"].(map[string]any)
		s.Regexp(`^va_ph_[A-Z2-7]{26}$`, env["GITHUB_TOKEN"])
		s.Regexp(`^http://sbx:[A-Z2-7]{26}@`, created["outboundProxyUrl"], i)
		s.Equal(created["outboundProxyUrl"], env["HTTPS_PROXY"])
		s.Equal(c.domains, valueOr(created, "domainAllowList"), i)
		s.Equal(c.ranges, valueOr(created, "networkAllowList"), i)
		s.Equal(c.noProxied, valueOr(env, "NO_PROXY"), i)
		s.Equal(c.noProxied, valueOr(env, "no_proxy"), i)
		s.Equal(false, created["networkBlockAll"])
		s.NotContains(created, "secrets")
	}
	s.Len(s.platform.requests(), len(cases))
	s.Equal([]string{"/tmp/va-egress-ca.pem after 0 runs", "/tmp/va-egress-ca.pem after 2 runs", "/tmp/va-egress-ca.pem after 4 runs",
		"/tmp/va-egress-ca.pem after 6 runs", "/tmp/va-egress-ca.pem after 8 runs"}, s.platform.uploads, "once per sandbox, before its first run")
	s.Equal("-----BEGIN CERTIFICATE-----\nca\n-----END CERTIFICATE-----\n", string(s.platform.files["/tmp/va-egress-ca.pem"]))
}

// valueOr is key's value in m, or "" without it.
func valueOr(m map[string]any, key string) any {
	if value, ok := m[key]; ok {
		return value
	}
	return ""
}

// TestACAFileThatCouldNotBeWrittenIsWrittenBeforeTheNextRun: no code runs without it.
func (s *ProxySuite) TestACAFileThatCouldNotBeWrittenIsWrittenBeforeTheNextRun() {
	box := s.proxied(sandbox.Config{}, "http://proxy.stg.example:8091")
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
	box := s.proxied(sandbox.Config{}, "http://proxy.stg.example:8091")
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
