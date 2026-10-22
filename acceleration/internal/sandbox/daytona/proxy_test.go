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
	"slices"
	"strconv"
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

// TestTheSandboxReachesTheProxyAlone: Daytona's network allow list is the proxy's address
// alone, with no domain list and no OutboundProxyUrl, which would bring Daytona's netleash in,
// whatever domains the config allows. The sandbox's own env holds the placeholder alone, and
// every command and run gets the proxy with the sandbox's credential and the CA env, which
// netleash would replace in the sandbox's. The CA file and its bundle come before any code,
// once per sandbox.
func (s *ProxySuite) TestTheSandboxReachesTheProxyAlone() {
	const proxy = "http://203.0.113.7:8091"
	cases := []sandbox.Config{
		{},
		{NetworkAllowList: []string{"10.0.0.0/8"}},
		{Network: true, AllowedDomains: []string{"pypi.org", "api.github.com", "github.com", "*.pythonhosted.org"}},
	}
	for _, config := range cases {
		box := s.proxied(config, proxy)
		s.Require().NoError(box.Write(s.ctx, "dir/f.txt", []byte("x")))
		_, err := box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
		_, err = box.Run(s.ctx, "print(2)", nil)
		s.Require().NoError(err)
	}

	for i, created := range s.platform.requests() {
		s.Equal([]string{"GITHUB_TOKEN"}, slices.Collect(maps.Keys(created["env"].(map[string]any))), i)
		s.Regexp(`^va_ph_[A-Z2-7]{26}$`, created["env"].(map[string]any)["GITHUB_TOKEN"])
		s.Equal("203.0.113.7/32", created["networkAllowList"], i)
		s.NotContains(created, "domainAllowList", i)
		s.NotContains(created, "outboundProxyUrl", i)
		s.Equal(false, created["networkBlockAll"])
		s.NotContains(created, "secrets")
	}
	s.Len(s.platform.requests(), len(cases))
	bundle := "/tmp/va-egress-ca-bundle.pem"
	runEnv := map[string]any{"SSL_CERT_FILE": bundle, "REQUESTS_CA_BUNDLE": bundle, "CURL_CA_BUNDLE": bundle, "GIT_SSL_CAINFO": bundle,
		"NODE_EXTRA_CA_CERTS": "/tmp/va-egress-ca.pem", "NO_PROXY": "localhost,127.0.0.1,::1", "no_proxy": "localhost,127.0.0.1,::1"}
	withProxy := func(env map[string]any) map[string]any {
		proxied := maps.Clone(runEnv)
		for _, name := range []string{"HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"} {
			s.Regexp(`^http://sbx:[A-Z2-7]{26}@203\.0\.113\.7:8091$`, env[name], name)
			proxied[name] = env[name]
		}
		return proxied
	}
	s.Len(s.platform.executed(), 2*len(cases))
	for _, ran := range s.platform.executed() {
		envs, _ := ran["envs"].(map[string]any)
		s.Equal(withProxy(envs), envs, "every run")
	}
	s.Len(s.platform.rawCommands, 2*len(cases))
	for _, raw := range s.platform.rawCommands {
		var command struct{ Envs map[string]any }
		s.Require().NoError(json.Unmarshal([]byte(raw), &command))
		s.Equal(withProxy(command.Envs), command.Envs, "every command")
	}
	var uploads []string
	for i := range cases {
		uploads = append(uploads, "/tmp/va-egress-ca.pem after "+strconv.Itoa(2*i)+" runs", "dir/f.txt after "+strconv.Itoa(2*i)+" runs")
	}
	s.Equal(uploads, s.platform.uploads, "once per sandbox, before its first file and run")
	s.Equal(caCert, string(s.platform.files["/tmp/va-egress-ca.pem"]))
	for i := range cases {
		s.Equal([]string{"upload", "command", "command", "upload"}, s.platform.steps[4*i:4*i+4],
			"the CA file is written before the setup uses it, and both before the sandbox's first file")
		s.Regexp(`^sh -c '.*> /tmp/va-egress-ca-bundle\.pem.*' after `+strconv.Itoa(2*i)+` runs$`, s.platform.commands[2*i],
			"once per sandbox, after the CA file")
	}
}

// TestAnHTTPSProxyIsReachedThroughATunnelCheckedBeforeEveryRun: the tunnel starts with the
// setup, after its file is written, and is checked before each run, whose clients it serves;
// a check that fails runs no code. The sandbox still reaches the proxy's address alone.
func (s *ProxySuite) TestAnHTTPSProxyIsReachedThroughATunnelCheckedBeforeEveryRun() {
	box := s.proxied(sandbox.Config{}, "https://203.0.113.7:8091")

	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	_, err = box.Run(s.ctx, "print(2)", nil)
	s.Require().NoError(err)

	s.Equal("203.0.113.7/32", s.platform.requests()[0]["networkAllowList"])
	s.Contains(s.platform.uploads, "/tmp/va-egress-tunnel.py after 0 runs")
	start := "sh -c " + quoted(box.mount.EachRun[0])
	s.Equal([]string{"sh -c " + quoted(box.mount.Setup[0]) + " after 0 runs", start + " after 0 runs", start + " after 0 runs",
		start + " after 1 runs"}, s.platform.commands)
	for _, ran := range s.platform.executed() {
		s.Regexp(`^http://sbx:[A-Z2-7]{26}@127\.0\.0\.1:3128$`, ran["envs"].(map[string]any)["HTTPS_PROXY"])
	}
	s.platform.mu.Lock()
	s.platform.commandExit = 1
	s.platform.mu.Unlock()
	_, err = box.Run(s.ctx, "print(3)", nil)
	s.ErrorContains(err, "readying the sandbox for a run")
	s.Len(s.platform.executed(), 2)
}

// TestAGrantTheBrokerLeavesOutHasNoEnvVar: the sandbox still reaches the proxy alone.
func (s *ProxySuite) TestAGrantTheBrokerLeavesOutHasNoEnvVar() {
	box := s.proxied(sandbox.Config{}, "http://203.0.113.7:8091", disconnected)

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	created := s.platform.requests()[0]
	s.Equal("203.0.113.7/32", created["networkAllowList"])
	s.NotContains(created, "domainAllowList")
	s.Equal([]string{"GITHUB_TOKEN"}, slices.Collect(maps.Keys(created["env"].(map[string]any))))
}

// TestASetupThatFailsRunsNoCode: a sandbox whose image has no roots to bundle the CA with runs
// nothing, and the setup runs again before the next run.
func (s *ProxySuite) TestASetupThatFailsRunsNoCode() {
	box := s.proxied(sandbox.Config{}, "http://203.0.113.7:8091")
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

// TestACAFileThatCouldNotBeWrittenIsWrittenBeforeTheNextRun: no code runs without it.
func (s *ProxySuite) TestACAFileThatCouldNotBeWrittenIsWrittenBeforeTheNextRun() {
	box := s.proxied(sandbox.Config{}, "http://203.0.113.7:8091")
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
	box := s.proxied(sandbox.Config{}, "http://203.0.113.7:8091")
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
