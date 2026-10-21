// Package brokertest is the suite every sandbox.CredentialBroker passes: the contract the
// session relies on, and the attacks on it a test can make without a sandbox's real egress
// (A1, A2, A8, A9, A10 and A13 of the wave's attack table; the egress ones run live for each
// broker). It is test code: import it only from _test.go files.
package brokertest

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// Subject is one broker under the contract, and its provider as the test sees it.
type Subject struct {
	Broker sandbox.CredentialBroker
	// Held is the credential and hosts the provider keeps under ref, a value of
	// Mount.Secrets; ok is false once there is none.
	Held func(ref string) (value string, hosts []string, ok bool)
	// Left is everything the broker has left with the provider, of every session.
	Left func() []string
	// Refuse makes the provider refuse every request from now on.
	Refuse func()
}

// BrokerContract is the suite. New builds a broker for session over tokens, logging to
// logger; every test may build several, one per session, on one provider.
type BrokerContract struct {
	suite.Suite
	New func(t *testing.T, session string, tokens sandbox.Tokens, logger *slog.Logger) Subject

	ctx    context.Context
	logs   *bytes.Buffer
	logger *slog.Logger
	spans  *tracetest.SpanRecorder

	mu sync.Mutex
	// tokens are what each binding's connection holds now, and refused the error each
	// binding's connection gives instead, if any.
	tokens  map[string]sandbox.Token
	refused map[string]error
	// issued are every token handed out, and errs every error a broker returned: none may
	// hold one.
	issued []string
	errs   []error
}

// SetupSuite installs the span recorder once: the global provider takes a delegate once
// per process.
func (s *BrokerContract) SetupSuite() {
	s.spans = tracetest.NewSpanRecorder()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(s.spans)))
}

func (s *BrokerContract) SetupTest() {
	s.ctx = context.Background()
	s.logs = &bytes.Buffer{}
	s.logger = slog.New(slog.NewMultiHandler(slog.NewJSONHandler(s.logs, nil), slog.NewTextHandler(s.logs, nil)))
	s.spans.Reset()
	s.tokens, s.refused, s.issued, s.errs = map[string]sandbox.Token{}, map[string]error{}, nil, nil
	s.connect("github", "gho_first", 1)
}

// TearDownTest is A13: no token reached a log line, a span or an error.
func (s *BrokerContract) TearDownTest() {
	var spans strings.Builder
	for _, span := range s.spans.Ended() {
		fmt.Fprintf(&spans, "%s %v %v %v\n", span.Name(), span.Attributes(), span.Events(), span.Status())
	}
	for _, token := range s.issued {
		s.NotContains(s.logs.String(), token, "a log line")
		s.NotContains(spans.String(), token, "a span")
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

// unavailable makes binding's connection one whose credential cannot be had just now, as
// when the provider fails to renew it.
func (s *BrokerContract) unavailable(binding string) {
	s.refuse(binding, errors.New("brokertest: the provider could not renew the credential just now"))
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
	ref := mount.Secrets["GITHUB_TOKEN"]
	s.Require().NotEmpty(ref)
	s.NotContains(fmt.Sprintf("%+v", mount), "gho_first")
	value, hosts, ok := subject.Held(ref)
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

	value, _, ok := subject.Held(mount.Secrets["GITHUB_TOKEN"])
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
	s.NotContains(mount.Secrets, "GITHUB_TOKEN")
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

// TestAConnectionRevokedMidSessionIsWithdrawn is A10: after BeforeRun, the provider holds
// no credential the router would no longer send itself.
func (s *BrokerContract) TestAConnectionRevokedMidSessionIsWithdrawn() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.disconnect("github")
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))

	_, _, ok := subject.Held(mount.Secrets["GITHUB_TOKEN"])
	s.False(ok)
	s.Contains(s.logs.String(), `"level":"WARN"`)
}

// TestACredentialUnavailableJustNowIsKeptAndAskedForAgain: a renewal that failed just now
// withdraws nothing, and the next run asks again.
func (s *BrokerContract) TestACredentialUnavailableJustNowIsKeptAndAskedForAgain() {
	subject := s.subject("session-1")
	mount, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})
	s.Require().NoError(s.returned(err))

	s.unavailable("github")
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))

	value, _, ok := subject.Held(mount.Secrets["GITHUB_TOKEN"])
	s.Require().True(ok)
	s.Equal("gho_first", value)
	s.Contains(s.logs.String(), `"level":"WARN"`)

	s.connect("github", "gho_second", 2)
	s.Require().NoError(s.returned(subject.Broker.BeforeRun(s.ctx)))
	value, _, _ = subject.Held(mount.Secrets["GITHUB_TOKEN"])
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
	s.NotEqual(one.Secrets["GITHUB_TOKEN"], other.Secrets["GITHUB_TOKEN"])

	s.Require().NoError(s.returned(first.Broker.Close(s.ctx)))

	_, _, ok := second.Held(other.Secrets["GITHUB_TOKEN"])
	s.True(ok)
}

func (s *BrokerContract) TestAProviderThatRefusesLeavesTheTokenInNoError() {
	subject := s.subject("session-1")
	subject.Refuse()

	_, err := subject.Broker.Prepare(s.ctx, []sandbox.Grant{grant("github", "GITHUB_TOKEN")})

	s.Error(s.returned(err))
}
