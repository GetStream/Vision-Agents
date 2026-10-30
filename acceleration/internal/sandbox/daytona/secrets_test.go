package daytona

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/brokertest"
)

func TestSecretsBrokerContract(t *testing.T) {
	var stub *platform
	suite.Run(t, &brokertest.BrokerContract{New: func(t *testing.T, session string, tokens sandbox.Tokens, bind sandbox.Bind, logger *slog.Logger) brokertest.Subject {
		// One provider per test, shared by its sessions.
		if stub == nil || stub.test != t.Name() {
			stub = newPlatform()
			stub.test = t.Name()
			t.Cleanup(stub.server.Close)
		}
		secrets, err := NewSecrets(SecretsOptions{APIKey: "key-1", APIURL: stub.server.URL + "/api", Prefix: "stg",
			Session: session, Tokens: tokens, Bind: bind, Logger: logger})
		if err != nil {
			t.Fatal(err)
		}
		held := stub
		return brokertest.Subject{
			Broker:    secrets,
			Elsewhere: true,
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
			Recover: func() {
				held.mu.Lock()
				defer held.mu.Unlock()
				held.refuse = false
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

// TestDaytonasBrokerSendsDaytonaWhatItDidBefore is the control: no per-run env and no proxy.
// The expected bodies are a probe of base 98410fe9b; the network-only configs after the first,
// of base 4588e625d; each with the auto-stop interval every sandbox is now created with.
func (s *SecretsSuite) TestDaytonasBrokerSendsDaytonaWhatItDidBefore() {
	box := s.granted("stg", sandbox.Config{AllowedDomains: []string{"pypi.org"}}, github)
	s.Require().NoError(box.Write(s.ctx, "dir/f.txt", []byte("x")))
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	networks := []sandbox.Config{{}, {Network: true}, {NetworkAllowList: []string{"10.0.0.0/8"}}}
	for _, config := range networks {
		_, err := s.granted("stg", config, github).Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
	}
	created, err := json.Marshal(s.platform.requests()[1:])
	s.Require().NoError(err)
	s.Equal(`[{"autoStopInterval":15,"domainAllowList":"api.github.com,uploads.github.com","labels":{"code-toolbox-language":"python"},`+
		`"networkBlockAll":false,"public":false,"secrets":[{"GITHUB_TOKEN":"SECRET"}],"spot":false},`+
		`{"autoStopInterval":15,"labels":{"code-toolbox-language":"python"},"networkBlockAll":false,"public":false,"secrets":[{"GITHUB_TOKEN":"SECRET"}],"spot":false},`+
		`{"autoStopInterval":15,"labels":{"code-toolbox-language":"python"},"networkAllowList":"10.0.0.0/8","networkBlockAll":false,"public":false,`+
		`"secrets":[{"GITHUB_TOKEN":"SECRET"}],"spot":false}]`,
		regexp.MustCompile(`va_stg_session-1_github_[0-9a-f]{8}`).ReplaceAllString(string(created), "SECRET"))

	created, err = json.Marshal(s.platform.requests()[:1])
	s.Require().NoError(err)
	ran, err := json.Marshal(s.platform.executed()[:1])
	s.Require().NoError(err)
	s.Equal(`[{"autoStopInterval":15,"domainAllowList":"pypi.org,api.github.com,uploads.github.com","labels":{"code-toolbox-language":"python"},`+
		`"networkBlockAll":false,"public":false,"secrets":[{"GITHUB_TOKEN":"SECRET"}],"spot":false}]`,
		regexp.MustCompile(`va_stg_session-1_github_[0-9a-f]{8}`).ReplaceAllString(string(created), "SECRET"))
	s.Equal(`[{"authorization":"Bearer key-1","code":"print(1)","language":"python","path":"/toolbox/sandbox-2/process/code-run","timeout":30}]`,
		string(ran))
	s.Equal([]string{"{\"command\":\"mkdir -p 'dir'\"}\n"}, s.platform.rawCommands)
}

// lateBox is a sandbox of session-1 with github and a late grant of linear, whose Bind gives the
// connection bound holds once it is set, and whose tokens are s.token.
func (s *SecretsSuite) lateBox(bound *sandbox.Grant) *Sandbox {
	key, url := "key-1", s.platform.server.URL+"/api"
	broker, err := NewSecrets(SecretsOptions{APIKey: key, APIURL: url, Prefix: "stg", Session: "session-1",
		Tokens: func(context.Context, sandbox.Grant) (sandbox.Token, error) { return s.token, s.tokenErr },
		Bind: func(context.Context, sandbox.Grant) (sandbox.Grant, bool) {
			return *bound, bound.Connection.ConnectionID != ""
		},
		Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	linear := sandbox.Grant{Binding: "linear", Env: "LINEAR_TOKEN", Hosts: []string{"api.linear.app"}, Late: true}
	box, err := New(Options{APIKey: key, APIURL: url, Grants: []sandbox.Grant{github, linear}, Broker: broker})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

// TestALateGrantsSecretHoldsAStandInUntilABindPutsTheTokenIn: Daytona creates no sandbox that
// names a missing secret, so the late grant's is made with the stand-in, and the first run after
// its bind puts the token in, whatever revision the credential is at.
func (s *SecretsSuite) TestALateGrantsSecretHoldsAStandInUntilABindPutsTheTokenIn() {
	s.token = sandbox.Token{Value: "lin_zero", Revision: 0}
	bound := sandbox.Grant{}
	box := s.lateBox(&bound)
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.ElementsMatch([]any{map[string]any{"GITHUB_TOKEN": s.platform.secrets["secret-1"].Name},
		map[string]any{"LINEAR_TOKEN": s.platform.secrets["secret-2"].Name}}, s.platform.requests()[0]["secrets"])
	s.Equal("va_unbound", s.platform.secrets["secret-2"].Value)
	s.Equal([]string{"api.linear.app"}, s.platform.secrets["secret-2"].Hosts)

	bound = sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-linear"}, Binding: "linear", Scheme: "bearer"}
	for range 2 {
		_, err = box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
	}

	s.Equal("lin_zero", s.platform.secrets["secret-2"].Value)
	s.Equal([]string{"POST /api/secret", "POST /api/secret", "POST /api/sandbox", "PATCH /api/secret/secret-2"}, s.platform.history())
}

// TestALateGrantsTokenDaytonaRefusedIsPutInOnTheNextRun: the run fails, and the next one tries
// again rather than keep the stand-in.
func (s *SecretsSuite) TestALateGrantsTokenDaytonaRefusedIsPutInOnTheNextRun() {
	s.token = sandbox.Token{Value: "lin_zero", Revision: 0}
	bound := sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-linear"}, Binding: "linear", Scheme: "bearer"}
	box := s.lateBox(&bound)
	s.platform.refuseUpdate = "linear"
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().Error(err)
	s.Equal("va_unbound", s.platform.secrets["secret-2"].Value)

	s.platform.refuseUpdate = ""
	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.Equal("lin_zero", s.platform.secrets["secret-2"].Value)
}

// TestAConfigWithoutLateGrantsSendsDaytonaWhatItDidBefore is the control: with a Bind and no
// late grant, the calls to Daytona, what each secret holds and the tokens asked for through
// Prepare and two runs are a probe of base 8e132bd37, and Bind is never called.
func (s *SecretsSuite) TestAConfigWithoutLateGrantsSendsDaytonaWhatItDidBefore() {
	var asked []string
	revision := 1
	broker, err := NewSecrets(SecretsOptions{APIKey: "key-1", APIURL: s.platform.server.URL + "/api", Prefix: "stg", Session: "session-1",
		Tokens: func(_ context.Context, grant sandbox.Grant) (sandbox.Token, error) {
			asked = append(asked, grant.Binding+"/"+grant.Connection.ConnectionID)
			return sandbox.Token{Value: fmt.Sprintf("tok_%s_%d", grant.Binding, revision), Revision: revision}, nil
		},
		Bind: func(context.Context, sandbox.Grant) (sandbox.Grant, bool) {
			s.Fail("Bind is called with no late grant")
			return sandbox.Grant{}, false
		}, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	gitlab := sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-gitlab"},
		Binding: "gitlab", Env: "GITLAB_TOKEN", Hosts: []string{"gitlab.com"}}

	mount, err := broker.Prepare(s.ctx, []sandbox.Grant{github, gitlab})
	s.Require().NoError(err)
	s.Require().NoError(broker.BeforeRun(s.ctx))
	revision = 2
	s.Require().NoError(broker.BeforeRun(s.ctx))

	s.Len(mount.Secrets, 2)
	s.Equal([]string{"POST /api/secret", "POST /api/secret", "PATCH /api/secret/secret-1", "PATCH /api/secret/secret-2"}, s.platform.history())
	s.Equal("tok_github_2", s.platform.secrets["secret-1"].Value)
	s.Equal([]string{"api.github.com", "uploads.github.com"}, s.platform.secrets["secret-1"].Hosts)
	s.Equal("tok_gitlab_2", s.platform.secrets["secret-2"].Value)
	s.Equal([]string{"gitlab.com"}, s.platform.secrets["secret-2"].Hosts)
	s.Equal([]string{"github/connection-1", "gitlab/connection-gitlab", "github/connection-1", "gitlab/connection-gitlab",
		"github/connection-1", "gitlab/connection-gitlab"}, asked)
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

// TestADaytonaRefusalOnTheWayLosesNoSecret: Daytona refuses once the withdrawal or the update
// of one of three grants, first, middle or last. The run fails; the grants before it hold what
// the run put in, it and the ones after it what they held; the next run puts it in theirs, and
// Close leaves nothing at Daytona.
func (s *SecretsSuite) TestADaytonaRefusalOnTheWayLosesNoSecret() {
	bindings := []string{"alpha", "bravo", "charlie"}
	grants := make([]sandbox.Grant, len(bindings))
	for i, binding := range bindings {
		grants[i] = sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-" + binding},
			Binding: binding, Env: strings.ToUpper(binding) + "_TOKEN", Hosts: []string{binding + ".example.com"}}
	}
	for _, path := range []string{"withdraw", "update"} {
		for refused := range bindings {
			s.Run(fmt.Sprintf("%s the %s", path, bindings[refused]), func() {
				platform := newPlatform()
				s.T().Cleanup(platform.server.Close)
				revision, withdrawn := 1, false
				key, url := "key-1", platform.server.URL+"/api"
				broker, err := NewSecrets(SecretsOptions{APIKey: key, APIURL: url, Prefix: "stg", Session: "session-1",
					Tokens: func(_ context.Context, grant sandbox.Grant) (sandbox.Token, error) {
						// A withdrawal takes every grant's credential; an update only the ones before the refused.
						if withdrawn && (path == "withdraw" || slices.Index(bindings, grant.Binding) < refused) {
							return sandbox.Token{}, sandbox.ErrNoCredential
						}
						return sandbox.Token{Value: fmt.Sprintf("gho_%d", revision), Revision: revision}, nil
					},
					Logger: slog.New(slog.DiscardHandler)})
				s.Require().NoError(err)
				box, err := New(Options{APIKey: key, APIURL: url, Grants: grants, Broker: broker})
				s.Require().NoError(err)
				s.T().Cleanup(func() { _ = box.Close() })
				_, err = box.Run(s.ctx, "print(1)", nil)
				s.Require().NoError(err)
				s.Require().Len(platform.secretNames(), len(bindings))

				withdrawn, revision = true, 2
				platform.mu.Lock()
				platform.refuseUpdate = bindings[refused]
				platform.mu.Unlock()
				_, err = box.Run(s.ctx, "print(1)", nil)
				s.Require().Error(err)
				after := map[string]string{"withdraw": "va_withdrawn", "update": "gho_2"}[path]
				want := map[string]string{}
				for i, binding := range bindings {
					want[binding] = "gho_1"
					if i < refused {
						want[binding] = "va_withdrawn"
					}
				}
				s.Equal(want, platform.values(bindings), "the grants before the refused one hold the run's, it and the ones after it theirs")

				platform.mu.Lock()
				platform.refuseUpdate = ""
				platform.mu.Unlock()
				_, err = box.Run(s.ctx, "print(1)", nil)
				s.Require().NoError(err)
				for _, binding := range bindings[refused:] {
					want[binding] = after
				}
				s.Equal(want, platform.values(bindings), "the refused grant and the ones after it get theirs on the next run")
				s.Require().NoError(box.broker.Close(s.ctx))
				s.Empty(platform.secretNames())
			})
		}
	}
}

// TestAWithdrawnGrantsSecretHoldsAStandInUntilClose (AI-1876): a run after the connection is
// revoked puts withdrawn in the secret rather than deleting it, so the sandbox's placeholder
// reaches the host as a credential it refuses, not Daytona's 502. A reconnect puts nothing back,
// and Close deletes the secret.
func (s *SecretsSuite) TestAWithdrawnGrantsSecretHoldsAStandInUntilClose() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.tokenErr = sandbox.ErrNoCredential
	_, err = s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.token, s.tokenErr = sandbox.Token{Value: "gho_second", Revision: 2}, nil
	_, err = s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.Equal("va_withdrawn", s.platform.secrets["secret-1"].Value)
	s.Equal(github.Hosts, s.platform.secrets["secret-1"].Hosts)
	s.Equal([]string{"POST /api/secret", "POST /api/sandbox", "PATCH /api/secret/secret-1"}, s.platform.history())
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

// TestASandboxDaytonaWouldNotDeleteStillLosesItsSecrets: the secrets go first, so a sandbox
// that stays at Daytona holds no credential, and Close reports the sandbox.
func (s *SecretsSuite) TestASandboxDaytonaWouldNotDeleteStillLosesItsSecrets() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.platform.mu.Lock()
	s.platform.deleteStatus = http.StatusInternalServerError
	s.platform.mu.Unlock()

	s.Error(s.box.Close())

	s.Empty(s.platform.secretNames())
}

// TestACloseARunOutlastsStillDeletesTheSecret: Close waits for the run until its timeout, and
// the secret goes then, not at the sweep. The run that outlasted it brings none back.
func (s *SecretsSuite) TestACloseARunOutlastsStillDeletesTheSecret() {
	s.box.timeout = time.Second
	ran, release := s.platform.outlasting(s.T(), s.box)
	s.Require().Len(s.platform.secretNames(), 1)

	s.ErrorIs(s.box.Close(), context.DeadlineExceeded)

	s.Empty(s.platform.secretNames())
	release()
	s.NoError(<-ran)
	s.Empty(s.platform.secretNames())
}

// TestAGrantWithoutHostsIsRefused: Daytona would keep a secret that no host may receive.
func (s *SecretsSuite) TestAGrantWithoutHostsIsRefused() {
	hostless := github
	hostless.Hosts = nil
	box := s.granted("stg", sandbox.Config{}, hostless)

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.ErrorContains(err, "no host")
	s.Empty(s.platform.history())
}

// TestAGrantAddsItsHostsToAnOpenNetworkWithDomains: network with allowed_domains is the
// domain list, so the grant's hosts join it (ruling 3).
func (s *SecretsSuite) TestAGrantAddsItsHostsToAnOpenNetworkWithDomains() {
	box := s.granted("stg", sandbox.Config{Network: true, AllowedDomains: []string{"pypi.org"}}, github)

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal("pypi.org,api.github.com,uploads.github.com", s.platform.requests()[0]["domainAllowList"])
}

// TestARequiredGrantWhoseCredentialFailedJustNowCreatesNoSandbox: a required grant is never
// left out silently, whatever the reason.
func (s *SecretsSuite) TestARequiredGrantWhoseCredentialFailedJustNowCreatesNoSandbox() {
	required := github
	required.Required = true
	box := s.granted("stg", sandbox.Config{}, required)
	s.tokenErr = errors.New("the provider could not renew the credential just now")

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().Error(err)
	s.NotErrorIs(err, sandbox.ErrNoCredential)
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

// TestASecretDaytonaWouldNotCreateLeavesNoOtherSecret: the grants made before it are deleted.
func (s *SecretsSuite) TestASecretDaytonaWouldNotCreateLeavesNoOtherSecret() {
	gitlab := github
	gitlab.Binding, gitlab.Env = "gitlab", "GITLAB_TOKEN"
	box := s.granted("stg", sandbox.Config{}, github, gitlab)
	s.platform.mu.Lock()
	s.platform.refuseSecret = "_gitlab_"
	s.platform.mu.Unlock()

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().Error(err)
	s.Empty(s.platform.secretNames())
	s.NotContains(s.platform.history(), "POST /api/sandbox")
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
	s.NotContains(description, "git over HTTPS", "a grant with no git host gets no git sentence")
	s.NotContains(description, "x-access-token")
}

// TestRunCodeTellsTheModelWhatARevokedGrantLooksLike: a grant connected at the start gets the
// line for a connection revoked or removed. A late grant's 401 first means "not connected yet",
// so it gets the revoked sentence only for a 401 or 403 after the user connected and it ran again.
// Neither promises a retry in this session.
func (s *SecretsSuite) TestRunCodeTellsTheModelWhatARevokedGrantLooksLike() {
	late := sandbox.Grant{Binding: "linear", Env: "LINEAR_TOKEN", Hosts: []string{"api.linear.app"}, Late: true}

	description := sandbox.Tool(s.granted("stg", sandbox.Config{}, githubGit, late)).Description

	s.Contains(description, " A 401 from api.github.com, github.com, or a 403 from the proxy, means the "+
		"github connection was revoked or removed: tell the user to connect github again; running the code again in this session will not help.")
	s.Contains(description, " A 401 from api.linear.app means the user has not connected linear yet:")
	s.Contains(description, "wait 20 seconds and run it once more. If it still gets 401 or 403 after the user connected and you ran it once more, "+
		"the connection was revoked or removed: tell the user; running it again in this session will not help.")
	s.NotContains(description, "means the linear connection was revoked")
	s.NotContains(description, "api.linear.app, or a 403")
	s.NotContains(sandbox.Tool(s.granted("stg", sandbox.Config{}, githubGit)).Description, "not connected", "a connected grant has no late text")
	s.NotContains(sandbox.Tool(s.granted("stg", sandbox.Config{})).Description, "revoked", "no grant, no line")
}

// githubGit is a GitHub grant that reaches github.com too, as the connector's manifest does.
var githubGit = sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-1"},
	Binding: "github", Env: "GITHUB_TOKEN", Hosts: []string{"api.github.com", "github.com"}}

// TestRunCodeTellsTheModelGitTakesTheGrant: Daytona replaces the placeholder in git's Base64
// Basic header too (live staging E2E, 2026-10-22, session 01a16514-923c-7843-9214-40a94453f194: git ls-remote with
// x-access-token:$GITHUB_TOKEN, rc 0). Bearer is not offered for github.com, which git cannot
// send it to.
func (s *SecretsSuite) TestRunCodeTellsTheModelGitTakesTheGrant() {
	description := sandbox.Tool(s.granted("stg", sandbox.Config{}, githubGit)).Description

	s.Contains(description, "Code may call api.github.com with $GITHUB_TOKEN: send it over HTTPS as "+
		"`Authorization: Bearer $GITHUB_TOKEN`; the value is a placeholder that works only for api.github.com, github.com")
	s.Contains(description, " git over HTTPS takes it as the password in the remote's URL, such as "+
		"https://x-access-token:$GITHUB_TOKEN@github.com/<owner>/<repo>.git.")
	s.NotContains(description, "cannot use")
	s.NotContains(description, "code_sandbox.broker")
}

// TestRunCodeOffersNoBearerForAGrantWhoseOnlyHostIsGithub: Bearer is not offered for github.com,
// so the description gives the placeholder's host alone.
func (s *SecretsSuite) TestRunCodeOffersNoBearerForAGrantWhoseOnlyHostIsGithub() {
	grant := githubGit
	grant.Hosts = []string{"github.com"}

	description := sandbox.Tool(s.granted("stg", sandbox.Config{}, grant)).Description

	s.Contains(description, " $GITHUB_TOKEN is a placeholder that works only for github.com, only in a request header.")
	s.NotContains(description, "Code may call")
	s.NotContains(description, "Bearer")
}

// TestASandboxWithoutGrantsIsCreatedAsOnBase is the control: a sandbox no binding gives
// access asks Daytona for what it asked before the broker existed, and run_code says what it
// said. Both expected values are from a probe on origin/accelerate 87fc7922e (a temporary
// DaytonaSuite test printing requests()[0] and sandbox.Tool().Description), with the auto-stop
// interval every sandbox is now created with.
func (s *SecretsSuite) TestASandboxWithoutGrantsIsCreatedAsOnBase() {
	box := s.granted("stg", sandbox.Config{})

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	raw, err := json.Marshal(s.platform.requests()[0])
	s.Require().NoError(err)
	s.JSONEq(`{"autoStopInterval":15,"labels":{"code-toolbox-language":"python"},"networkBlockAll":true,"public":false,"spot":false}`, string(raw))
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

// values are what the secret of each of bindings holds, by binding.
func (p *platform) values(bindings []string) map[string]string {
	p.mu.Lock()
	defer p.mu.Unlock()
	values := map[string]string{}
	for secret := range maps.Values(p.secrets) {
		for _, binding := range bindings {
			if strings.Contains(secret.Name, "_"+binding+"_") {
				values[binding] = secret.Value
			}
		}
	}
	return values
}

func (p *platform) history() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.calls)
}

// TestAGrantedSandboxIsCreatedAsOnBase is the control for the proxy broker: with Daytona's
// broker, the default, a granted sandbox is asked for as before it, and nothing is written
// into it. The expected requests are from a probe on origin/accelerate a103c19ab (a temporary
// SecretsSuite test printing requests()[0] and history() for these three configs), with the
// auto-stop interval every sandbox is now created with.
func (s *SecretsSuite) TestAGrantedSandboxIsCreatedAsOnBase() {
	expected := []string{
		`{"autoStopInterval":15,"domainAllowList":"api.github.com,uploads.github.com","labels":{"code-toolbox-language":"python"},"networkBlockAll":false,"public":false,"secrets":[{"GITHUB_TOKEN":"SECRET"}],"spot":false}`,
		`{"autoStopInterval":15,"domainAllowList":"pypi.org,api.github.com,uploads.github.com","labels":{"code-toolbox-language":"python"},"networkBlockAll":false,"public":false,"secrets":[{"GITHUB_TOKEN":"SECRET"}],"spot":false}`,
		`{"autoStopInterval":15,"labels":{"code-toolbox-language":"python"},"networkAllowList":"10.0.0.0/8","networkBlockAll":false,"public":false,"secrets":[{"GITHUB_TOKEN":"SECRET"}],"spot":false}`,
	}
	for i, config := range []sandbox.Config{{}, {Network: true, AllowedDomains: []string{"pypi.org"}}, {NetworkAllowList: []string{"10.0.0.0/8"}}} {
		s.platform = newPlatform()
		s.T().Cleanup(s.platform.server.Close)
		box := s.granted("stg", config, github)

		_, err := box.Run(s.ctx, "print(1)", nil)

		s.Require().NoError(err)
		raw, err := json.Marshal(s.platform.requests()[0])
		s.Require().NoError(err)
		s.JSONEq(strings.Replace(expected[i], "SECRET", s.platform.secretNames()[0], 1), string(raw))
		s.Equal([]string{"POST /api/secret", "POST /api/sandbox"}, s.platform.history())
		s.Empty(s.platform.uploads)
	}
}
