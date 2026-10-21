package daytona

import (
	"context"
	"encoding/json"
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
	box      *Sandbox
}

func TestSecretsSuite(t *testing.T) {
	suite.Run(t, new(SecretsSuite))
}

func (s *SecretsSuite) SetupTest() {
	s.ctx = context.Background()
	s.platform = newPlatform()
	s.T().Cleanup(s.platform.server.Close)
	s.token = sandbox.Token{Value: "gho_first", Revision: 1}
	s.box = s.granted("stg", github)
}

var github = sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-1"},
	Binding: "github", Env: "GITHUB_TOKEN", Hosts: []string{"api.github.com", "uploads.github.com"}}

// granted is a sandbox of session-1 whose grants Daytona's broker holds, under prefix.
func (s *SecretsSuite) granted(prefix string, grants ...sandbox.Grant) *Sandbox {
	key, url := "key-1", s.platform.server.URL+"/api"
	broker, err := NewSecrets(SecretsOptions{APIKey: key, APIURL: url, Prefix: prefix, Session: "session-1",
		Tokens: func(context.Context, sandbox.Grant) (sandbox.Token, error) { return s.token, nil },
		Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	box, err := New(Options{APIKey: key, APIURL: url, Grants: grants, Broker: broker})
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

func (s *SecretsSuite) TestAnUnchangedRevisionSendsNothingMore() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	_, err = s.box.Run(s.ctx, "print(2)", nil)
	s.Require().NoError(err)

	s.Equal([]string{"POST /api/secret", "POST /api/sandbox"}, s.platform.history())
}

func (s *SecretsSuite) TestANewRevisionUpdatesTheSecretBeforeTheRun() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.token = sandbox.Token{Value: "gho_second", Revision: 2}
	_, err = s.box.Run(s.ctx, "print(2)", nil)
	s.Require().NoError(err)

	s.Equal([]string{"POST /api/secret", "POST /api/sandbox", "PATCH /api/secret/secret-1"}, s.platform.history())
	s.Equal("gho_second", s.platform.secrets["secret-1"].Value)
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
	box := s.granted("dev.kanat/mac", github)

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Regexp(`^va_dev_kanat_mac_session-1_github_[0-9a-f]{8}$`, s.platform.secretNames()[0])
}

func (s *SecretsSuite) TestRunCodeTellsTheModelEachEnvVarAndItsHosts() {
	description := sandbox.Tool(s.box).Description

	s.Contains(description, "Code may call api.github.com, uploads.github.com with $GITHUB_TOKEN: send it over HTTPS as "+
		"`Authorization: Bearer $GITHUB_TOKEN`; the value is a placeholder that works only for api.github.com, uploads.github.com")
}

// TestASandboxWithoutGrantsIsCreatedAsOnBase is the control: a sandbox no binding gives
// access asks Daytona for what it asked before the broker existed, and run_code says what it
// said. Both expected values are from 97bb54003 (author-1.md, «base probe»).
func (s *SecretsSuite) TestASandboxWithoutGrantsIsCreatedAsOnBase() {
	box := s.granted("stg")

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	raw, err := json.Marshal(s.platform.requests()[0])
	s.Require().NoError(err)
	s.JSONEq(`{"labels":{"code-toolbox-language":"python"},"networkBlockAll":true,"public":false,"spot":false}`, string(raw))
	s.Empty(s.platform.secretNames())
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
