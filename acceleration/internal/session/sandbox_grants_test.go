//go:build integration

package session

import (
	"bytes"
	"fmt"
	"log/slog"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SandboxGrantsSuite is what a session's sandbox gets of its config's sandbox bindings: a
// grant per binding whose connection's pinned revision has a sandbox block, and the token the
// broker hands the sandbox provider.
type SandboxGrantsSuite struct {
	connectorFixture
	logs *bytes.Buffer
}

func TestSandboxGrantsSuite(t *testing.T) {
	suite.Run(t, new(SandboxGrantsSuite))
}

func (s *SandboxGrantsSuite) SetupTest() {
	s.connectorFixture.SetupTest()
	s.T().Setenv("DAYTONA_API_KEY", "key-1")
	s.logs = &bytes.Buffer{}
	s.manager = s.managerWith(fixtureRequestTimeout)
	s.manager.logger = slog.New(slog.NewTextHandler(s.logs, nil))
	s.manager.options.Connectors.Resolver = s.resolver
	s.manager.options.SandboxBroker, s.manager.options.SandboxSecretPrefix = daytona.BrokerName, "test"
}

// sandboxRevision stores revision 2 of the test's connector, the first with a sandbox block,
// and has the connections made after it pin it.
func (s *SandboxGrantsSuite) sandboxRevision() {
	manifest, err := core.ParseManifest([]byte(s.crmManifest(s.connectorID, 2) + `
sandbox:
  env: CRM_TOKEN
  hosts: [api.crm.example, "*.crm.example"]
`))
	s.Require().NoError(err)
	definition, err := s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	s.revision = definition.Revision
}

func sandboxed(binding store.ConnectorBinding) store.ConnectorBinding {
	binding.Sandbox = true
	return binding
}

// opened is the sandbox of a daytona session of config, with its connectors attached.
func (s *SandboxGrantsSuite) opened(config store.AgentConfig) (sandbox.Sandbox, *dispatcher) {
	spec := s.spec(config, "", nil)
	spec.ID, spec.Sandbox = uuid.NewString(), daytonaProvider
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	box, err := s.manager.box(spec, d)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box, d
}

func (s *SandboxGrantsSuite) TestASandboxBindingGivesTheSandboxItsEnvVarAndHosts() {
	s.sandboxRevision()
	id := s.connection("", "primary")

	box, _ := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))

	s.Equal([]sandbox.Grant{{Connection: core.ConnectionRef{CustomerID: s.customerID, ConnectionID: id}, Binding: "crm",
		Env: "CRM_TOKEN", Hosts: []string{"api.crm.example", "*.crm.example"}}}, box.(sandbox.Granted).Grants())
	s.Contains(harness.SandboxTool(box).Description, "`Authorization: Bearer $CRM_TOKEN`")
}

func (s *SandboxGrantsSuite) TestTheBrokerIsHandedTheConnectionsBearerToken() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, d := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))

	token, err := s.manager.sandboxTokens(d.schemes)(s.ctx, box.(sandbox.Granted).Grants()[0])

	s.Require().NoError(err)
	s.Equal(tokenOf("primary"), token.Value)
	s.Positive(token.Revision)
	s.NotContains(s.logs.String(), tokenOf("primary"))
}

func (s *SandboxGrantsSuite) TestABindingWithoutSandboxGivesTheSandboxNothing() {
	s.sandboxRevision()
	id := s.connection("", "primary")

	box, _ := s.opened(s.config(s.fixed("crm", id, "whoami")))

	s.Empty(box.(sandbox.Granted).Grants())
}

// TestAConnectionSavedBeforeTheSandboxBlockGetsNoAccessAndAWarning is a GitHub connection
// still on revision 5: its tools work, its sandbox gets no env var, and the log says to
// reconnect.
func (s *SandboxGrantsSuite) TestAConnectionSavedBeforeTheSandboxBlockGetsNoAccessAndAWarning() {
	id := s.connection("", "primary")
	s.sandboxRevision()

	box, d := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))

	s.Empty(box.(sandbox.Granted).Grants())
	s.Len(d.tools, 1, "the binding's tools are still offered")
	s.Contains(s.logs.String(), "level=WARN")
	s.Contains(s.logs.String(), "reconnect to use sandbox access")
}

func (s *SandboxGrantsSuite) TestARequiredBindingSavedBeforeTheSandboxBlockRefusesTheSession() {
	id := s.connection("", "primary")
	s.sandboxRevision()
	spec := s.spec(s.config(required(sandboxed(s.fixed("crm", id, "whoami")))), "", nil)

	_, _, _, err := s.attach(spec)

	s.ErrorContains(err, fmt.Sprintf("required connector %q cannot be used: %s", "crm", unavailableSandbox))
}

func (s *SandboxGrantsSuite) TestASandboxBindingNeedsTheSecretPrefix() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	s.manager.options.SandboxSecretPrefix = ""
	spec := s.spec(s.config(sandboxed(s.fixed("crm", id, "whoami"))), "", nil)
	spec.ID, spec.Sandbox = uuid.NewString(), daytonaProvider
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)

	_, err = s.manager.box(spec, d)

	s.ErrorContains(err, "ROUTER_SANDBOX_SECRET_PREFIX")
}
