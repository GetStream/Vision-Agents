//go:build integration

package session

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/none"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
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
		Scheme: bearer.Name, Env: "CRM_TOKEN", Hosts: []string{"api.crm.example", "*.crm.example"}}}, box.(sandbox.Granted).Grants())
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

// TestAConnectionThatNeedsAReconnectGivesNoCredential: the broker withdraws its secret.
func (s *SandboxGrantsSuite) TestAConnectionThatNeedsAReconnectGivesNoCredential() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, d := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))
	s.setState(id, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })

	_, err := s.manager.sandboxTokens(d.schemes)(s.ctx, box.(sandbox.Granted).Grants()[0])

	s.ErrorIs(err, sandbox.ErrNoCredential)
}

// TestASchemeWithoutABearerTokenOrNoResolverGivesNoCredential: Daytona matches only a
// placeholder a header carries verbatim, and without a resolver there is no credential.
func (s *SandboxGrantsSuite) TestASchemeWithoutABearerTokenOrNoResolverGivesNoCredential() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, d := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))
	grant := box.(sandbox.Granted).Grants()[0]

	_, unsent := s.manager.sandboxTokens(map[string]core.Scheme{"crm": none.New()})(s.ctx, grant)
	s.manager.options.Connectors.Resolver = nil
	_, unresolved := s.manager.sandboxTokens(d.schemes)(s.ctx, grant)

	s.ErrorIs(unsent, sandbox.ErrNoCredential)
	s.ErrorIs(unresolved, sandbox.ErrNoCredential)
}

// nonBearer is a scheme that sends its credential as Basic, which Daytona would not
// replace a placeholder in.
type nonBearer struct{ core.Scheme }

func (nonBearer) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return roundTripper(func(r *http.Request) (*http.Response, error) {
		r = r.Clone(r.Context())
		r.Header.Set("Authorization", "Basic dXNlcjpwYXNz")
		return base.RoundTrip(r)
	})
}

type roundTripper func(*http.Request) (*http.Response, error)

func (f roundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// TestASchemeSendingAnotherAuthorizationGivesNoCredential: a header that is not
// "Bearer <token>" is no token to hand over, whatever it holds.
func (s *SandboxGrantsSuite) TestASchemeSendingAnotherAuthorizationGivesNoCredential() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, d := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))

	_, err := s.manager.sandboxTokens(map[string]core.Scheme{"crm": nonBearer{d.schemes["crm"]}})(s.ctx, box.(sandbox.Granted).Grants()[0])

	s.ErrorIs(err, sandbox.ErrNoCredential)
}

// TestAResolverFailureThatMayPassIsNoReasonToWithdraw: any other failure, here a cancelled
// call, leaves the broker's secret in place for the next run to refresh.
func (s *SandboxGrantsSuite) TestAResolverFailureThatMayPassIsNoReasonToWithdraw() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, d := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))
	cancelled, cancel := context.WithCancel(s.ctx)
	cancel()

	_, err := s.manager.sandboxTokens(d.schemes)(cancelled, box.(sandbox.Granted).Grants()[0])

	s.Require().Error(err)
	s.NotErrorIs(err, sandbox.ErrNoCredential)
}

// TestABindingWithoutSandboxGivesTheSandboxNothing, on a router with no secret prefix: such a
// session needs none.
func (s *SandboxGrantsSuite) TestABindingWithoutSandboxGivesTheSandboxNothing() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	s.manager.options.SandboxSecretPrefix = ""

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

// boxed is the error of making a daytona session's sandbox of options for a sandbox binding.
func (s *SandboxGrantsSuite) boxed(options sandbox.Config) error {
	s.sandboxRevision()
	id := s.connection("", "primary")
	spec := s.spec(s.config(sandboxed(s.fixed("crm", id, "whoami"))), "", nil)
	spec.ID, spec.Sandbox, spec.SandboxOptions = uuid.NewString(), daytonaProvider, options
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	_, err = s.manager.box(spec, d)
	return err
}

func (s *SandboxGrantsSuite) TestASandboxBindingNeedsTheSecretPrefix() {
	s.manager.options.SandboxSecretPrefix = ""

	s.ErrorContains(s.boxed(sandbox.Config{}), "ROUTER_CODE_SANDBOX_SECRET_PREFIX")
}

func (s *SandboxGrantsSuite) TestASandboxBindingNeedsABrokerThereIs() {
	s.manager.options.SandboxBroker = "vault"

	s.ErrorContains(s.boxed(sandbox.Config{}), `there is no sandbox broker called "vault"`)
}

// proxyCA is the proxy CA's certificate as the proxy broker is given it.
const proxyCA = "-----BEGIN CERTIFICATE-----\nY2E=\n-----END CERTIFICATE-----\n"

// TestTheProxyBrokerSendsTheSandboxThroughTheProxy: with code_sandbox.broker proxy, run_code's
// sandbox reaches the proxy alone, holds a placeholder, and the proxy finds the connection's
// token behind it, for this session. No secret prefix is needed. A connection deleted
// mid-session is withdrawn, in the session's log.
func (s *SandboxGrantsSuite) TestTheProxyBrokerSendsTheSandboxThroughTheProxy() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	redis, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(redis.Close)
	api := s.daytonaAPI()
	s.manager.options.SandboxBroker, s.manager.options.SandboxSecretPrefix = egress.BrokerName, ""
	s.manager.options.SandboxProxy = egress.BrokerOptions{Store: egress.NewStore(redis), URL: "http://proxy.stg.example:8091",
		CACert: []byte(proxyCA)}
	s.sandboxRevision()
	id := s.connection("", "primary")
	spec := s.spec(s.config(sandboxed(s.fixed("crm", id, "whoami"))), "", nil)
	spec.ID, spec.Sandbox = uuid.NewString(), daytonaProvider
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	box, err := s.manager.box(spec, d)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })

	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal("proxy.stg.example", api.created["domainAllowList"])
	proxy, err := url.Parse(api.created["outboundProxyUrl"].(string))
	s.Require().NoError(err)
	credential, _ := proxy.User.Password()
	sum := sha256.Sum256([]byte(credential))
	raw, err := redis.Do(s.ctx, redis.B().Get().Key("sandbox:egress:"+hex.EncodeToString(sum[:])).Build()).AsBytes()
	s.Require().NoError(err)
	var record egress.Record
	s.Require().NoError(json.Unmarshal(raw, &record))
	s.Equal(spec.ID, record.Session)
	s.Require().Len(record.Grants, 1)
	s.Equal(api.created["env"].(map[string]any)["CRM_TOKEN"], record.Grants[0].Placeholder)
	token, err := egress.ResolverTokens(s.resolver, s.registry.Schemes)(s.ctx, record.Grants[0])
	s.Require().NoError(err)
	s.Equal(tokenOf("primary"), token)
	s.NotContains(string(raw), tokenOf("primary"))

	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, id))
	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Contains(s.logs.String(), "credential is withdrawn from the sandbox")
}

// TestAManagerWithTheProxyBrokerNeedsItsStore: the router does not start without it, rather
// than refuse every sandbox session.
func (s *SandboxGrantsSuite) TestAManagerWithTheProxyBrokerNeedsItsStore() {
	_, err := NewManager(ManagerOptions{SandboxBroker: egress.BrokerName,
		SandboxProxy: egress.BrokerOptions{URL: "http://proxy.stg.example:8091", CACert: []byte(proxyCA)}})

	s.ErrorContains(err, "the proxy sandbox broker needs the store")
}

// TestTheProxyBrokerRefusesAConfigItCannotServe: an open network would reach the grants alone,
// and Daytona's one CIDR allow list cannot hold a proxy named by a host name. Daytona's broker
// takes both, and the proxy takes a network narrowed to domains, or ranges with its address.
func (s *SandboxGrantsSuite) TestTheProxyBrokerRefusesAConfigItCannotServe() {
	open := sandbox.Config{Network: true}
	ranges := sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8"}}
	s.NoError(s.boxed(open))
	s.NoError(s.boxed(ranges))
	s.manager.options.SandboxBroker = egress.BrokerName
	s.manager.options.SandboxProxy = egress.BrokerOptions{Store: &egress.Store{}, URL: "http://proxy.stg.example:8091", CACert: []byte(proxyCA)}

	err := s.boxed(open)
	s.ErrorContains(err, "sandbox_options.network is true")
	s.ErrorContains(err, "code_sandbox.broker is proxy")
	s.ErrorContains(s.boxed(ranges), "sandbox_options.network_allow_list must hold the egress proxy's address")
	s.NoError(s.boxed(sandbox.Config{Network: true, AllowedDomains: []string{"pypi.org"}}))
	s.manager.options.SandboxProxy.URL = "http://203.0.113.7:8091"
	s.NoError(s.boxed(ranges))
	s.NoError(s.boxed(sandbox.Config{Network: true, NetworkAllowList: []string{"10.0.0.0/8"}}),
		"a network narrowed to ranges is not open")
}

// daytonaAPI stands in for Daytona's API: it creates sandboxes, runs code, and keeps the
// organization's secrets by name. secrets maps the env vars of the last sandbox created to
// the secret names they read.
type daytonaAPI struct {
	*httptest.Server
	mu      sync.Mutex
	held    map[string]string
	secrets map[string]string
	// created is the last request for a sandbox.
	created map[string]any
}

func (s *SandboxGrantsSuite) daytonaAPI() *daytonaAPI {
	api := &daytonaAPI{held: map[string]string{}}
	api.Server = httptest.NewServer(http.HandlerFunc(api.serve))
	s.T().Cleanup(api.Close)
	s.T().Setenv("DAYTONA_API_URL", api.URL+"/api")
	return api
}

func (a *daytonaAPI) serve(w http.ResponseWriter, r *http.Request) {
	a.mu.Lock()
	defer a.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	id := strings.TrimPrefix(r.URL.Path, "/api/secret/")
	switch {
	case r.Method == http.MethodPost && r.URL.Path == "/api/secret":
		var body struct{ Name string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		a.held["secret-"+body.Name] = body.Name
		_, _ = fmt.Fprintf(w, `{"id":"secret-%s","name":%q,"placeholder":"dtn_secret_1","hosts":[],"createdAt":"2026-10-21T00:00:00Z","updatedAt":"2026-10-21T00:00:00Z"}`, body.Name, body.Name)
	case r.Method == http.MethodDelete && id != r.URL.Path:
		delete(a.held, id)
		w.WriteHeader(http.StatusNoContent)
	case r.Method == http.MethodPost && r.URL.Path == "/api/sandbox":
		var body struct{ Secrets []map[string]string }
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		_ = json.Unmarshal(raw, &a.created)
		if len(body.Secrets) > 0 {
			a.secrets = body.Secrets[0]
		}
		w.WriteHeader(http.StatusCreated)
		_, _ = fmt.Fprintf(w, `{"organizationId":"org","name":"test","user":"root","env":{},"public":false,"networkBlockAll":false,"target":"us","cpu":2,"gpu":0,"memory":4,"disk":10,"id":"sandbox-1","state":"started","toolboxProxyUrl":"%s/toolbox","labels":{"daytona.io/code-toolbox-language":"python"}}`, a.URL)
	case strings.HasSuffix(r.URL.Path, "/files/upload-v2"):
		_, _ = fmt.Fprintf(w, `{"name":"file","path":%q,"type":"file"}`, r.URL.Query().Get("path"))
	case r.Method == http.MethodDelete || strings.HasPrefix(r.URL.Path, "/api/socket.io"):
		w.WriteHeader(http.StatusNoContent)
	default:
		_, _ = w.Write([]byte(`{"result":"","exitCode":0}`))
	}
}

// offered says whether Daytona still holds a secret for the env var the sandbox reads.
func (a *daytonaAPI) offered(env string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, name := range a.held {
		if name == a.secrets[env] {
			return true
		}
	}
	return false
}

// TestADeletedConnectionIsWithdrawnBeforeTheNextRun: a connection deleted mid-session is gone
// for good, so the next run_code finds no credential behind the sandbox's env var.
func (s *SandboxGrantsSuite) TestADeletedConnectionIsWithdrawnBeforeTheNextRun() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, _ := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Require().True(api.offered("CRM_TOKEN"))

	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, id))
	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.False(api.offered("CRM_TOKEN"))
	s.Contains(s.logs.String(), "credential is withdrawn from the sandbox")
}

// TestAConnectionThatCannotBeReadJustNowKeepsItsSecret: a store that cannot answer is no
// reason to withdraw, as on base.
func (s *SandboxGrantsSuite) TestAConnectionThatCannotBeReadJustNowKeepsItsSecret() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, _ := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	closed, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	credentials, err := pgsealed.New(closed, s.sealer)
	s.Require().NoError(err)
	s.manager.options.Connectors.Resolver, err = resolver.New(resolver.Config{Store: closed, Credentials: credentials, Schemes: s.registry.Schemes})
	s.Require().NoError(err)

	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.True(api.offered("CRM_TOKEN"))
	s.Contains(s.logs.String(), "it keeps the one it has")
}
