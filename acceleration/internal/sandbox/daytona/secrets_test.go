package daytona

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/brokertest"
)

func TestSecretsBrokerContract(t *testing.T) {
	var stub *platform
	suite.Run(t, &brokertest.BrokerContract{New: func(t *testing.T, session string, tokens sandbox.Tokens, logger *slog.Logger) brokertest.Subject {
		// One provider per test, shared by its sessions.
		if stub == nil || stub.test != t.Name() {
			stub = newPlatform()
			stub.test = t.Name()
			t.Cleanup(stub.server.Close)
		}
		secrets, err := NewSecrets(SecretsOptions{APIKey: "key-1", APIURL: stub.server.URL + "/api", Prefix: "stg",
			Session: session, Tokens: tokens, Logger: logger})
		if err != nil {
			t.Fatal(err)
		}
		held := stub
		return brokertest.Subject{
			Broker: secrets,
			Held: func(name string) (string, []string, bool) {
				held.mu.Lock()
				defer held.mu.Unlock()
				for _, secret := range held.secrets {
					if secret.Name == name {
						return secret.Value, secret.Hosts, true
					}
				}
				return "", nil, false
			},
			Left: held.secretNames,
			Refuse: func() {
				held.mu.Lock()
				defer held.mu.Unlock()
				held.refuse = true
			},
		}
	}})
}

// SecretsSuite is what Daytona's broker sends Daytona, and the sandbox it is created for.
type SecretsSuite struct {
	suite.Suite
	ctx      context.Context
	platform *platform
	token    sandbox.Token
	tokenErr error
	box      *Sandbox
}

func TestSecretsSuite(t *testing.T) {
	suite.Run(t, new(SecretsSuite))
}

func (s *SecretsSuite) SetupTest() {
	s.ctx = context.Background()
	s.platform = newPlatform()
	s.T().Cleanup(s.platform.server.Close)
	s.token, s.tokenErr = sandbox.Token{Value: "gho_first", Revision: 1}, nil
	s.box = s.granted("stg", sandbox.Config{}, github)
}

var github = sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-1"},
	Binding: "github", Env: "GITHUB_TOKEN", Hosts: []string{"api.github.com", "uploads.github.com"}}

// granted is a sandbox of session-1 made from config, whose grants Daytona's broker holds,
// under prefix.
func (s *SecretsSuite) granted(prefix string, config sandbox.Config, grants ...sandbox.Grant) *Sandbox {
	key, url := "key-1", s.platform.server.URL+"/api"
	broker, err := NewSecrets(SecretsOptions{APIKey: key, APIURL: url, Prefix: prefix, Session: "session-1",
		Tokens: func(context.Context, sandbox.Grant) (sandbox.Token, error) { return s.token, s.tokenErr },
		Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	box, err := New(Options{APIKey: key, APIURL: url, Config: config, Grants: grants, Broker: broker})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

func (s *SecretsSuite) TestTheSandboxIsCreatedWithTheSecretMadeForItsGrant() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	names := s.platform.secretNames()
	s.Require().Len(names, 1)
	s.Regexp(`^va_stg_session-1_github_[0-9a-f]{8}$`, names[0])
	s.Equal([]any{map[string]any{"GITHUB_TOKEN": names[0]}}, s.platform.requests()[0]["secrets"])
	s.Equal([]string{"POST /api/secret", "POST /api/sandbox"}, s.platform.history(), "the secret exists before the sandbox")
}

// TestOnlyANewRevisionUpdatesTheSecretBeforeTheRun: once per revision, and not before.
func (s *SecretsSuite) TestOnlyANewRevisionUpdatesTheSecretBeforeTheRun() {
	for _, revision := range []int{1, 1, 2, 2} {
		s.token = sandbox.Token{Value: fmt.Sprintf("gho_%d", revision), Revision: revision}
		_, err := s.box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
	}

	s.Equal([]string{"POST /api/secret", "POST /api/sandbox", "PATCH /api/secret/secret-1"}, s.platform.history())
	s.Equal("gho_2", s.platform.secrets["secret-1"].Value)
}

// TestASecretDaytonaWouldNotDeleteIsDeletedOnTheNextClose: Close keeps what it could not
// delete for the next try.
func (s *SecretsSuite) TestASecretDaytonaWouldNotDeleteIsDeletedOnTheNextClose() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.platform.mu.Lock()
	s.platform.refuse = true
	s.platform.mu.Unlock()
	s.Require().Error(s.box.broker.Close(s.ctx))

	s.platform.mu.Lock()
	s.platform.refuse = false
	s.platform.mu.Unlock()
	s.Require().NoError(s.box.broker.Close(s.ctx))

	s.Empty(s.platform.secretNames())
}

// TestARequiredGrantWithoutACredentialCreatesNoSandbox: the run fails before Daytona is asked
// for a sandbox.
func (s *SecretsSuite) TestARequiredGrantWithoutACredentialCreatesNoSandbox() {
	required := github
	required.Required = true
	box := s.granted("stg", sandbox.Config{}, required)
	s.tokenErr = sandbox.ErrNoCredential

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.ErrorIs(err, sandbox.ErrNoCredential)
	s.Empty(s.platform.history())
}

func (s *SecretsSuite) TestClosingDeletesTheSecretAndTheSandbox() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.Require().NoError(s.box.Close())

	s.Empty(s.platform.secretNames())
	s.Equal([]string{"POST /api/secret", "POST /api/sandbox", "DELETE /api/secret/secret-1"}, s.platform.history())
	s.Equal([]string{"/api/sandbox/sandbox-2"}, s.platform.released())
}

func (s *SecretsSuite) TestASandboxDaytonaWouldNotCreateLeavesNoSecret() {
	s.platform.mu.Lock()
	s.platform.createStatus = 503
	s.platform.mu.Unlock()

	_, err := s.box.Run(s.ctx, "print(1)", nil)

	s.Require().Error(err)
	s.Empty(s.platform.secretNames())
}

func (s *SecretsSuite) TestASecretNameKeepsToDaytonasCharacters() {
	box := s.granted("dev.kanat/mac", sandbox.Config{}, github)

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Regexp(`^va_dev_kanat_mac_session-1_github_[0-9a-f]{8}$`, s.platform.secretNames()[0])
}

// TestAGrantOpensItsHostsInASandboxThatReachesNothingElse: the config's network blocks
// everything, and the grant's hosts are all the sandbox reaches.
func (s *SecretsSuite) TestAGrantOpensItsHostsInASandboxThatReachesNothingElse() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	created := s.platform.requests()[0]
	s.Equal(false, created["networkBlockAll"])
	s.Equal("api.github.com,uploads.github.com", created["domainAllowList"])
}

func (s *SecretsSuite) TestAGrantAddsItsHostsToTheConfigsDomains() {
	box := s.granted("stg", sandbox.Config{AllowedDomains: []string{"pypi.org", "api.github.com"}}, github)

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	created := s.platform.requests()[0]
	s.Equal(false, created["networkBlockAll"])
	s.Equal("pypi.org,api.github.com,uploads.github.com", created["domainAllowList"])
}

// TestAGrantLeavesAnOpenNetworkAndCIDRRangesAsTheConfigSetsThem: an open network already
// reaches the hosts, and Daytona takes a domain list or a CIDR list, never both.
func (s *SecretsSuite) TestAGrantLeavesAnOpenNetworkAndCIDRRangesAsTheConfigSetsThem() {
	open := s.granted("stg", sandbox.Config{Network: true}, github)
	ranges := s.granted("stg", sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8"}}, github)

	_, err := open.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	_, err = ranges.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.Equal(false, s.platform.requests()[0]["networkBlockAll"])
	s.NotContains(s.platform.requests()[0], "domainAllowList")
	s.Equal(false, s.platform.requests()[1]["networkBlockAll"])
	s.Equal("10.0.0.0/8", s.platform.requests()[1]["networkAllowList"])
	s.NotContains(s.platform.requests()[1], "domainAllowList")
}

// TestAGrantWithoutACredentialOpensNoHost: a disconnected connection's hosts stay closed.
func (s *SecretsSuite) TestAGrantWithoutACredentialOpensNoHost() {
	s.tokenErr = sandbox.ErrNoCredential

	_, err := s.box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal(true, s.platform.requests()[0]["networkBlockAll"])
	s.NotContains(s.platform.requests()[0], "domainAllowList")
}

func (s *SecretsSuite) TestRunCodeTellsTheModelEachEnvVarAndItsHosts() {
	description := sandbox.Tool(s.box).Description

	s.Contains(description, "Code may call api.github.com, uploads.github.com with $GITHUB_TOKEN: send it over HTTPS as "+
		"`Authorization: Bearer $GITHUB_TOKEN`; the value is a placeholder that works only for api.github.com, uploads.github.com")
}

// TestASandboxWithoutGrantsIsCreatedAsOnBase is the control: a sandbox no binding gives
// access asks Daytona for what it asked before the broker existed, and run_code says what it
// said. Both expected values are from a probe on origin/accelerate 87fc7922e (a temporary
// DaytonaSuite test printing requests()[0] and sandbox.Tool().Description).
func (s *SecretsSuite) TestASandboxWithoutGrantsIsCreatedAsOnBase() {
	box := s.granted("stg", sandbox.Config{})

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	raw, err := json.Marshal(s.platform.requests()[0])
	s.Require().NoError(err)
	s.JSONEq(`{"labels":{"code-toolbox-language":"python"},"networkBlockAll":true,"public":false,"spot":false}`, string(raw))
	s.Equal([]string{"POST /api/sandbox"}, s.platform.history(), "no secret is asked for")
	s.Equal("Run a short Python program and read back what it printed. Use it for anything you would otherwise have "+
		"to work out in your head, such as arithmetic, dates or parsing, and for making files such as images. Print what "+
		"you want to know. A skill's own scripts are under skills/<name>/; run one in another language through "+
		`subprocess, such as subprocess.run(["go", "run", "skills/pdf/scripts/merge.go"]).`, sandbox.Tool(box).Description)
}

func (p *platform) secretNames() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var names []string
	for secret := range maps.Values(p.secrets) {
		names = append(names, secret.Name)
	}
	slices.Sort(names)
	return names
}

func (p *platform) history() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.calls)
}
