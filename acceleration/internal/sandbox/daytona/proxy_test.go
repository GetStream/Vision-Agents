//go:build integration

package daytona

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"maps"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
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

// disconnected is a grant whose connection gives the sandbox no credential.
var disconnected = sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-2"},
	Binding: "linear", Env: "LINEAR_TOKEN", Hosts: []string{"api.linear.app"}}

// caCert is the proxy CA's certificate as the broker is given it.
const caCert = "-----BEGIN CERTIFICATE-----\nY2E=\n-----END CERTIFICATE-----\n"

// proxied is a sandbox of config whose github grant the egress proxy broker at proxy holds.
func (s *ProxySuite) proxied(config sandbox.Config, proxy string, more ...sandbox.Grant) *Sandbox {
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: egress.NewStore(s.redis), URL: proxy,
		CACert: []byte(caCert), Session: "session-1",
		Tokens: func(_ context.Context, grant sandbox.Grant) (sandbox.Token, error) {
			if grant.Binding == disconnected.Binding {
				return sandbox.Token{}, sandbox.ErrNoCredential
			}
			return sandbox.Token{Value: "gho_first", Revision: 1}, nil
		},
		Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	box, err := New(Options{APIKey: "key-1", APIURL: s.platform.server.URL + "/api", Config: config, Grants: append([]sandbox.Grant{github}, more...),
		Broker: broker})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

// TestTheSandboxGoesThroughTheProxyToItsGrantsAndAllowedDomains: Daytona holds the proxy URL with
// the sandbox's credential, the allow list is the grants' hosts and the config's domains and
// never the proxy's host, the sandbox's own env holds the placeholder alone, and every command
// and run gets the CA env, which Daytona's netleash would replace in the sandbox's. The CA file
// and its bundle come before any code, once per sandbox.
func (s *ProxySuite) TestTheSandboxGoesThroughTheProxyToItsGrantsAndAllowedDomains() {
	const proxy = "http://proxy.stg.example:8091"
	cases := []struct {
		config  sandbox.Config
		proxy   string
		domains string
	}{
		{sandbox.Config{}, proxy, "api.github.com,uploads.github.com"},
		{sandbox.Config{}, "http://203.0.113.7:8091", "api.github.com,uploads.github.com"},
		{sandbox.Config{Network: true, AllowedDomains: []string{"pypi.org", "api.github.com", "github.com", "*.pythonhosted.org"}}, proxy,
			"pypi.org,api.github.com,github.com,*.pythonhosted.org,uploads.github.com"},
	}
	for _, c := range cases {
		box := s.proxied(c.config, c.proxy)
		s.Require().NoError(box.Write(s.ctx, "dir/f.txt", []byte("x")))
		_, err := box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
		_, err = box.Run(s.ctx, "print(2)", nil)
		s.Require().NoError(err)
	}

	for i, created := range s.platform.requests() {
		c := cases[i]
		s.Equal([]string{"GITHUB_TOKEN"}, slices.Collect(maps.Keys(created["env"].(map[string]any))), i)
		s.Regexp(`^va_ph_[A-Z2-7]{26}$`, created["env"].(map[string]any)["GITHUB_TOKEN"])
		s.Regexp(`^http://sbx:[A-Z2-7]{26}@`+regexp.QuoteMeta(strings.TrimPrefix(c.proxy, "http://"))+`$`, created["outboundProxyUrl"], i)
		s.Equal(c.domains, valueOr(created, "domainAllowList"), i)
		s.NotContains(created, "networkAllowList")
		s.Equal(false, created["networkBlockAll"])
		s.NotContains(created, "secrets")
	}
	s.Len(s.platform.requests(), len(cases))
	bundle := "/tmp/va-egress-ca-bundle.pem"
	runEnv := map[string]any{"SSL_CERT_FILE": bundle, "REQUESTS_CA_BUNDLE": bundle, "CURL_CA_BUNDLE": bundle, "GIT_SSL_CAINFO": bundle,
		"NODE_EXTRA_CA_CERTS": "/tmp/va-egress-ca.pem"}
	s.Len(s.platform.executed(), 2*len(cases))
	for _, ran := range s.platform.executed() {
		s.Equal(runEnv, ran["envs"], "every run")
	}
	s.Len(s.platform.rawCommands, 2*len(cases))
	for _, raw := range s.platform.rawCommands {
		var command struct{ Envs map[string]any }
		s.Require().NoError(json.Unmarshal([]byte(raw), &command))
		s.Equal(runEnv, command.Envs, "every command")
	}
	s.Equal([]string{"/tmp/va-egress-ca.pem after 0 runs", "dir/f.txt after 0 runs", "/tmp/va-egress-ca.pem after 2 runs",
		"dir/f.txt after 2 runs", "/tmp/va-egress-ca.pem after 4 runs", "dir/f.txt after 4 runs"}, s.platform.uploads,
		"once per sandbox, before its first file and run")
	s.Equal(caCert, string(s.platform.files["/tmp/va-egress-ca.pem"]))
	for i := range cases {
		s.Equal([]string{"upload", "command", "command", "upload"}, s.platform.steps[4*i:4*i+4],
			"the CA file is written before the setup uses it, and both before the sandbox's first file")
		s.Regexp(`^sh -c '.*> /tmp/va-egress-ca-bundle\.pem.*' after `+strconv.Itoa(2*i)+` runs$`, s.platform.commands[2*i],
			"once per sandbox, after the CA file")
	}
}

// TestAGrantTheBrokerLeavesOutIsNotInTheAllowList: its host is not reachable, and its env var is
// not set.
func (s *ProxySuite) TestAGrantTheBrokerLeavesOutIsNotInTheAllowList() {
	box := s.proxied(sandbox.Config{}, "http://proxy.stg.example:8091", disconnected)

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	created := s.platform.requests()[0]
	s.Equal("api.github.com,uploads.github.com", valueOr(created, "domainAllowList"))
	s.Equal([]string{"GITHUB_TOKEN"}, slices.Collect(maps.Keys(created["env"].(map[string]any))))
}

// TestASetupThatFailsRunsNoCode: a sandbox whose image has no roots to bundle the CA with runs
// nothing, and the setup runs again before the next run.
func (s *ProxySuite) TestASetupThatFailsRunsNoCode() {
	box := s.proxied(sandbox.Config{}, "http://proxy.stg.example:8091")
	s.platform.mu.Lock()
	s.platform.commandExit = 1
	s.platform.mu.Unlock()

	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().ErrorContains(err, "setting the sandbox up")
	s.Empty(s.platform.executed())
	s.platform.mu.Lock()
	s.platform.commandExit = 0
	s.platform.mu.Unlock()
	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Len(s.platform.commands, 2)
	s.Len(s.platform.executed(), 1)
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
