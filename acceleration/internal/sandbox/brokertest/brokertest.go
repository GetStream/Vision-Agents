// Package brokertest is the suite every sandbox.CredentialBroker passes: the contract the
// session relies on, and the attacks on it a test can make without a sandbox's real egress
// (A1, A2, A8, A9, A10 and A13, as far as logs and errors go, of the wave's attack table, A15 where the
// egress runs here, and A16; the egress ones run live for each broker). It is test code: import it only from _test.go files.
package brokertest

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// Subject is one broker under the contract, and its provider as the test sees it.
type Subject struct {
	Broker sandbox.CredentialBroker
	// Held is the credential and hosts the provider keeps under ref, the value of a grant's
	// env var; ok is false once there is none.
	Held func(ref string) (value string, hosts []string, ok bool)
	// Left is everything the broker has left with the provider, of every session.
	Left func() []string
	// Refuse makes the provider refuse every request from now on, until Recover.
	Refuse func()
	// Recover has the provider answer again.
	Recover func()
	// Live is a broker whose egress asks for the credential on every request and keeps no
	// copy: Held is what a request gets now, so a credential that cannot be had just now
	// reaches no request, and a refresh the provider refused still sends the newest.
	Live bool
	// Elsewhere is a broker whose egress runs outside this process, in the provider's network
	// (Daytona), so no test here can push through it. Only such a subject may leave Push nil.
	Elsewhere bool
	// Push is git in the sandbox the broker's last Prepare made: it commits a file holding
	// content and pushes it to https://github.com/octo/hello.git with ref, the value of a
	// grant's env var, as Basic auth's password. It returns the Authorization of each request
	// github.com got, the file as github.com keeps it, and every other header value and the URL of
	// each request. Nil only when Elsewhere.
	Push func(ref, content string) (authorizations []string, pushed string, elsewhere []string)
}

// BrokerContract is the suite. New builds a broker for session over tokens, logging to
// logger; every test may build several, one per session, on one provider.
type BrokerContract struct {
	suite.Suite
	New func(t *testing.T, session string, tokens sandbox.Tokens, logger *slog.Logger) Subject

	ctx    context.Context
	logs   *bytes.Buffer
	logger *slog.Logger

	mu sync.Mutex
	// tokens are what each binding's connection holds now, and refused the error each
	// binding's connection gives instead, if any.
	tokens  map[string]sandbox.Token
	refused map[string]error
	// issued are every token handed out, quoted every provider answer a connection's error
	// quotes, and errs every error a broker returned: none may hold one.
	issued []string
	quoted []string
	errs   []error
}

func (s *BrokerContract) SetupTest() {
	s.ctx = context.Background()
	s.logs = &bytes.Buffer{}
	s.logger = slog.New(slog.NewMultiHandler(slog.NewJSONHandler(s.logs, nil), slog.NewTextHandler(s.logs, nil)))
	s.tokens, s.refused, s.issued, s.quoted, s.errs = map[string]sandbox.Token{}, map[string]error{}, nil, nil, nil
	s.connect("github", "gho_first", 1)
}

// TearDownTest is A13: no token, and no provider answer, reached a log line or an error. Spans
// are not checked: the provider SDKs make theirs only when told to, with an exporter of their own.
func (s *BrokerContract) TearDownTest() {
	for _, token := range append(s.issued, s.quoted...) {
		s.NotContains(s.logs.String(), token, "a log line")
		for _, err := range s.errs {
			s.NotContains(fmt.Sprintf("%v %+v", err, err), token, "an error")
		}
	}
}

// connect has binding's connection hold token at revision.
func (s *BrokerContract) connect(binding, token string, revision int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tokens[binding], s.refused[binding] = sandbox.Token{Value: token, Revision: revision}, nil
	s.issued = append(s.issued, token)
}

// disconnect makes binding's connection one that cannot be used until it is reconnected.
func (s *BrokerContract) disconnect(binding string) {
	s.refuse(binding, fmt.Errorf("%w: brokertest: the connection is not connected", sandbox.ErrNoCredential))
}

// unbind is the agent config dropping binding: its tokens give ErrNoCredential, as
// session.SandboxBound has them.
func (s *BrokerContract) unbind(binding string) {
	s.refuse(binding, fmt.Errorf("%w: brokertest: the agent config no longer binds it", sandbox.ErrNoCredential))
}

// unavailable makes binding's connection one whose credential cannot be had just now, as
// when the provider fails to renew it.
func (s *BrokerContract) unavailable(binding string) {
	s.refuse(binding, errors.New("brokertest: the provider could not renew the credential just now"))
}

// quoting makes binding's connection give an error that quotes the provider's answer, as a
// refresh the provider refused does: a reconnect is needed when reconnect is set.
func (s *BrokerContract) quoting(binding string, reconnect bool) {
	// Plain words: a log handler escapes quotes, which would hide a JSON answer from the check.
	answer := fmt.Sprintf("brokertest-quoted-%d", len(s.quoted))
	s.quoted = append(s.quoted, answer)
	err := fmt.Errorf("brokertest: the provider answered bad_refresh_token %s", answer)
	if reconnect {
		err = fmt.Errorf("%w: %w", sandbox.ErrNoCredential, err)
	}
	s.refuse(binding, err)
}

func (s *BrokerContract) refuse(binding string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refused[binding] = err
}

func (s *BrokerContract) resolve(_ context.Context, grant sandbox.Grant) (sandbox.Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.refused[grant.Binding]; err != nil {
		return sandbox.Token{}, err
	}
	return s.tokens[grant.Binding], nil
}

func (s *BrokerContract) subject(session string) Subject {
	subject := s.New(s.T(), session, s.resolve, s.logger)
	s.T().Cleanup(func() { _ = subject.Broker.Close(context.Background()) })
	return subject
}

// returned keeps err for TearDownTest.
func (s *BrokerContract) returned(err error) error {
	if err != nil {
		s.errs = append(s.errs, err)
	}
	return err
}

// ref is the value of the sandbox's env var name: a secret's name, or a placeholder.
func ref(mount sandbox.Mount, name string) string {
	if secret, ok := mount.Secrets[name]; ok {
		return secret
	}
	return mount.Env[name]
}

func grant(binding, env string) sandbox.Grant {
	return sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-" + binding},
		Binding: binding, Env: env, Hosts: []string{"api.github.com", "uploads.github.com"}}
}

// TestTheSandboxGetsAReferenceNeverTheToken is A1 and A2: what the sandbox is created with
// names the credential and never holds it, and the provider keeps it for the grant's hosts
// alone.
func (s *BrokerContract) TestTheSandboxGetsAReferenceNeverTheToken() {
	subject := s.subject("session-1")

	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})

	s.Require().NoError(s.returned(err))
	reference := ref(mount, "GITHUB_TOKEN")
	s.Require().NotEmpty(reference)
	s.NotContains(fmt.Sprintf("%+v", mount), "gho_first")
	value, hosts, ok := subject.Held(reference)
	s.Require().True(ok)
	s.Equal("gho_first", value)
	s.Equal([]string{"api.github.com", "uploads.github.com"}, hosts)
}

func (s *BrokerContract) TestANewRevisionIsHandedOverBeforeTheNextRun() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.connect("github", "gho_second", 2)
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))

	value, _, ok := subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.Require().True(ok)
	s.Equal("gho_second", value)
}

// TestClosingLeavesNothingWithTheProvider is A9 as far as the provider's API goes: once the
// session ends, no credential is kept anywhere its placeholder could reach.
func (s *BrokerContract) TestClosingLeavesNothingWithTheProvider() {
	subject := s.subject("session-1")
	_, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN"), grant("gitlab", "GITLAB_TOKEN")})
	s.Require().NoError(s.returned(err))
	s.Len(subject.Left(), 2)

	s.Require().NoError(s.returned(subject.Broker.Close(s.ctx)))
	s.Require().NoError(s.returned(subject.Broker.Close(s.ctx)), "closing twice is safe")

	s.Empty(subject.Left())
}

func (s *BrokerContract) TestADisconnectedConnectionGivesNoEnvAndAWarning() {
	subject := s.subject("session-1")
	s.disconnect("github")

	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})

	s.Require().NoError(s.returned(err))
	s.Empty(ref(mount, "GITHUB_TOKEN"))
	s.Empty(subject.Left())
	s.Contains(s.logs.String(), `"level":"WARN"`)
	s.Contains(s.logs.String(), "github")
}

func (s *BrokerContract) TestARequiredGrantWithoutACredentialRefusesTheSandbox() {
	subject := s.subject("session-1")
	s.connect("gitlab", "glpat_first", 1)
	s.disconnect("github")
	required := grant("github", "GITHUB_TOKEN")
	required.Required = true

	_, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("gitlab", "GITLAB_TOKEN"), required})

	s.Require().Error(s.returned(err))
	s.ErrorContains(err, "github")
	s.Empty(subject.Left(), "what was made before the refusal is gone")
}

// TestARequiredGrantsRefusalQuotesNoProviderAnswer is A13 on a provider's answer: it may hold
// anything, so neither the error a required grant fails with, which reaches the model, nor a
// log line holds it (TearDownTest). A connection that needs a reconnect still says so.
func (s *BrokerContract) TestARequiredGrantsRefusalQuotesNoProviderAnswer() {
	subject := s.subject("session-1")
	s.quoting("github", true)
	required := grant("github", "GITHUB_TOKEN")
	required.Required = true

	_, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{required})

	s.Require().Error(s.returned(err))
	s.ErrorIs(err, sandbox.ErrNoCredential)
	s.ErrorContains(err, "github")

	s.quoting("github", false)
	_, err = subject.Broker.Prepare(s.ctx, []sandbox.Grant{required})
	if subject.Live {
		s.Require().NoError(s.returned(err), "the proxy asks again on each request")
	} else {
		s.Require().Error(s.returned(err))
		s.NotErrorIs(err, sandbox.ErrNoCredential, "a credential that cannot be had just now needs no reconnect")
	}
}

// TestNoProviderAnswerReachesTheLogs: a grant left out, kept or withdrawn is a warning naming
// its connector, and never the provider's answer (TearDownTest).
func (s *BrokerContract) TestNoProviderAnswerReachesTheLogs() {
	subject := s.subject("session-1")
	s.connect("gitlab", "glpat_first", 1)
	s.quoting("github", true)
	_, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN"), grant("gitlab", "GITLAB_TOKEN")})
	s.Require().NoError(s.returned(err))
	s.quoting("github", false)
	_, err = subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN"), grant("gitlab", "GITLAB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.quoting("gitlab", false)
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))
	s.quoting("gitlab", true)
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))

	s.GreaterOrEqual(strings.Count(s.logs.String(), `"level":"WARN"`), 4, "one per grant left out, kept or withdrawn")
}

// TestAConnectionRevokedMidSessionIsWithdrawn is A10: after BeforeRun, the provider holds
// no credential the router would no longer send itself.
func (s *BrokerContract) TestAConnectionRevokedMidSessionIsWithdrawn() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.disconnect("github")
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))

	_, _, ok := subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.False(ok)
	s.Contains(s.logs.String(), `"level":"WARN"`)
}

// TestABindingRemovedMidSessionIsWithdrawn is A16: once the agent config drops a binding, a live
// egress sends its token to no further request, and every broker withdraws it before the next
// run. Putting the binding back gives this sandbox nothing again.
func (s *BrokerContract) TestABindingRemovedMidSessionIsWithdrawn() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.unbind("github")
	_, _, ok := subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.Equal(!subject.Live, ok, "a live egress refuses the next request")
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))
	_, _, ok = subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.False(ok, "the next run has no token")

	s.connect("github", "gho_first", 1)
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))
	_, _, ok = subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.False(ok, "a binding put back stays withdrawn")
	s.Empty(subject.Left())
}

// TestACredentialUnavailableJustNowIsKeptAndAskedForAgain: a renewal that failed just now
// withdraws nothing, and the next run asks again. A live egress has no copy to send meanwhile.
func (s *BrokerContract) TestACredentialUnavailableJustNowIsKeptAndAskedForAgain() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.unavailable("github")
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))

	value, _, ok := subject.Held(ref(mount, "GITHUB_TOKEN"))
	if subject.Live {
		s.False(ok)
	} else {
		s.Require().True(ok)
		s.Equal("gho_first", value)
	}
	s.Contains(s.logs.String(), `"level":"WARN"`)

	s.connect("github", "gho_second", 2)
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))
	value, _, _ = subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.Equal("gho_second", value)
}

// TestTwoSessionsNeverShareACredential is A8 as far as the provider's API goes: each
// session's grant is its own, and ending one leaves the other's.
func (s *BrokerContract) TestTwoSessionsNeverShareACredential() {
	// A reopened session keeps its id, so two sessions of one id are the harder case.
	first, second := s.subject("session-1"), s.subject("session-1")
	one, err := first.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))
	other, err := second.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))
	s.NotEqual(ref(one, "GITHUB_TOKEN"), ref(other, "GITHUB_TOKEN"))

	s.Require().NoError(s.returned(first.Broker.Close(s.ctx)))

	_, _, ok := second.Held(ref(other, "GITHUB_TOKEN"))
	s.True(ok)
}

func (s *BrokerContract) TestAProviderThatRefusesLeavesTheTokenInNoError() {
	subject := s.subject("session-1")
	subject.Refuse()

	_, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})

	s.Error(s.returned(err))
}

// TestARefusedRefreshIsAnErrorWithoutTheToken is A13 on the refresh path: a new revision the
// provider refuses to take fails the run, and the error holds no token. A live egress sends
// the new one all the same.
func (s *BrokerContract) TestARefusedRefreshIsAnErrorWithoutTheToken() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.connect("github", "gho_second", 2)
	subject.Refuse()

	s.Error(s.returned(subject.Broker.BeforeRun(s.ctx)))
	subject.Recover()
	value, _, ok := subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.Require().True(ok)
	if subject.Live {
		s.Equal("gho_second", value)
	} else {
		s.Equal("gho_first", value)
	}
}

// TestAWithdrawalTheProviderRefusesIsAnErrorAndIsRetried: a revoked connection's credential
// the provider would not delete fails the run, stays held, and goes on the next run or Close.
// A live egress sends a revoked connection's credential to no request even then.
func (s *BrokerContract) TestAWithdrawalTheProviderRefusesIsAnErrorAndIsRetried() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN"), grant("gitlab", "GITLAB_TOKEN")})
	s.Require().NoError(s.returned(err))
	s.Require().NotEmpty(ref(mount, "GITHUB_TOKEN"))
	s.Require().NotEmpty(ref(mount, "GITLAB_TOKEN"))

	s.disconnect("github")
	subject.Refuse()
	s.Error(s.returned(subject.Broker.BeforeRun(s.ctx)))
	_, _, ok := subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.Equal(!subject.Live, ok, "the secret stays while the provider refuses")

	subject.Recover()
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))
	_, _, ok = subject.Held(ref(mount, "GITHUB_TOKEN"))
	s.False(ok, "the next run withdraws it")
	s.Len(subject.Left(), 1)

	subject.Refuse()
	s.Error(s.returned(subject.Broker.Close(s.ctx)))
	subject.Recover()
	s.Require().NoError(s.returned(subject.Broker.Close(s.ctx)))
	s.Empty(subject.Left(), "Close deletes what an earlier try could not")
}

// TestAGitPushCarriesTheTokenInAuthorizationAlone is A15: git sends the placeholder as Basic
// auth's password, and github.com gets the token there and nowhere else. A commit holding the
// placeholder is pushed with the placeholder as it is: the body is never rewritten.
func (s *BrokerContract) TestAGitPushCarriesTheTokenInAuthorizationAlone() {
	subject := s.subject("session-1")
	if subject.Elsewhere {
		// Daytona: its egress runs in Daytona's network, which no test here reaches. What a test
		// here can check is the run_code description (daytona/secrets_test.go). Daytona replaces
		// the placeholder in git's Base64 Basic header: live staging E2E, 2026-10-22, session
		// 01a16514-923c-7843-9214-40a94453f194.
		s.T().Skip("the broker's egress runs outside this process: the live E2E checks git")
	}
	s.Require().NotNil(subject.Push, "a broker whose egress runs here must push through it")
	github := grant("github", "GITHUB_TOKEN")
	github.Hosts = append(github.Hosts, "github.com")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(s.returned(err))
	placeholder := ref(mount, "GITHUB_TOKEN")

	authorizations, pushed, elsewhere := subject.Push(placeholder, "GITHUB_TOKEN="+placeholder+"\n")

	token := "Basic " + base64.StdEncoding.EncodeToString([]byte("x-access-token:gho_first"))
	s.Contains(authorizations, token)
	for _, authorization := range authorizations {
		s.Contains([]string{"", token}, authorization, "github.com gets the token or no credential")
	}
	s.Equal("GITHUB_TOKEN="+placeholder+"\n", pushed)
	s.NotEmpty(elsewhere)
	for _, value := range elsewhere {
		s.NotContains(value, "gho_first", "the token is in Authorization alone")
	}
}
