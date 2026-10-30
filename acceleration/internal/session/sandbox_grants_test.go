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
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/none"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
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
	logs       *bytes.Buffer
	handedBack *lockedBuffer // the manager's log once a test holds a session (held)
}

func TestSandboxGrantsSuite(t *testing.T) {
	suite.Run(t, new(SandboxGrantsSuite))
}

func (s *SandboxGrantsSuite) SetupTest() {
	s.connectorFixture.SetupTest()
	s.T().Setenv("DAYTONA_API_KEY", "key-1")
	s.logs, s.handedBack = &bytes.Buffer{}, nil
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

// readsRevision stores revision 3 of the test's connector, revision 2's sandbox block with reads,
// and returns them. Connections keep pinning revision 2.
func (s *SandboxGrantsSuite) readsRevision() []core.SandboxRead {
	manifest, err := core.ParseManifest([]byte(s.crmManifest(s.connectorID, 3) + `
sandbox:
  env: CRM_TOKEN
  hosts: [api.crm.example, "*.crm.example"]
  reads:
    - {host: api.crm.example, method: POST, path: /search}
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	return manifest.Sandbox.Reads
}

// TestAGrantTakesItsReadsFromTheLatestRevision: a connection pinned to revision 2 keeps its env
// var and hosts, and gets revision 3's reads without a reconnect.
func (s *SandboxGrantsSuite) TestAGrantTakesItsReadsFromTheLatestRevision() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	reads := s.readsRevision()

	box, _ := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))

	grants := box.(sandbox.Granted).Grants()
	s.Require().Len(grants, 1)
	s.Equal(reads, grants[0].Reads)
	s.Equal("CRM_TOKEN", grants[0].Env)
	s.Equal([]string{"api.crm.example", "*.crm.example"}, grants[0].Hosts)
}

// TestALateGrantTakesTheLatestRevisionsReads: a grant waiting for its user's login reads the
// latest revision's block, its reads included.
func (s *SandboxGrantsSuite) TestALateGrantTakesTheLatestRevisionsReads() {
	s.daytonaAPI()
	s.sandboxRevision()
	reads := s.readsRevision()
	s.connection("bob", "primary")

	box, _ := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), Consent{})

	grants := box.(sandbox.Granted).Grants()
	s.Require().Len(grants, 1)
	s.True(grants[0].Late)
	s.Equal(reads, grants[0].Reads)
}

func sandboxed(binding store.ConnectorBinding) store.ConnectorBinding {
	binding.Sandbox = true
	return binding
}

// opened is the sandbox of a daytona session of config, with its connectors attached.
func (s *SandboxGrantsSuite) opened(config store.AgentConfig) (sandbox.Sandbox, *dispatcher) {
	spec := s.spec(config, "", nil)
	spec.ID, spec.Sandbox = uuid.NewString(), daytonaProvider
	spec.Attribution.SessionID = spec.ID
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	box, err := s.manager.box(s.ctx, spec, d)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box, d
}

func (s *SandboxGrantsSuite) TestASandboxBindingGivesTheSandboxItsEnvVarAndHosts() {
	s.sandboxRevision()
	id := s.connection("", "primary")

	config := s.config(sandboxed(s.fixed("crm", id, "whoami")))

	box, _ := s.opened(config)

	s.Equal([]sandbox.Grant{{Connection: core.ConnectionRef{CustomerID: s.customerID, ConnectionID: id}, Binding: "crm",
		Config: config.ID, Connector: s.connectorID, Selection: selectionFixed, Scheme: bearer.Name, Env: "CRM_TOKEN", Hosts: []string{"api.crm.example", "*.crm.example"}}}, box.(sandbox.Granted).Grants())
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

// TestTheBrokerIsHandedTheTokenForTheGrantsFirstHost: Hosts[0] is the host the scheme wraps a
// request for, with a leading *. taken off, whatever the hosts after it are.
func (s *SandboxGrantsSuite) TestTheBrokerIsHandedTheTokenForTheGrantsFirstHost() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, _ := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))
	schemes := map[string]core.Scheme{"crm": hostBearer{Scheme: bearer.New()}}

	for hosts, asked := range map[string]string{"api.crm.example,*.crm.example": "api.crm.example",
		"*.crm.example,api.crm.example": "crm.example"} {
		grant := box.(sandbox.Granted).Grants()[0]
		grant.Hosts = strings.Split(hosts, ",")

		token, err := s.manager.sandboxTokens(schemes)(s.ctx, grant)

		s.Require().NoError(err, hosts)
		s.Equal(asked, token.Value, hosts)
	}
}

// hostBearer is a scheme that sends the host a request is for as its bearer token.
type hostBearer struct{ core.Scheme }

func (hostBearer) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return roundTripper(func(r *http.Request) (*http.Response, error) {
		r = r.Clone(r.Context())
		r.Header.Set("Authorization", "Bearer "+r.URL.Host)
		return base.RoundTrip(r)
	})
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

// TestABindingOpenedWithoutADriftedToolStillGivesTheSandboxAccess: secret's schema changed
// since the config granted it, so the binding opens with whoami alone, and its sandbox still
// gets the connection's grant.
func (s *SandboxGrantsSuite) TestABindingOpenedWithoutADriftedToolStillGivesTheSandboxAccess() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	binding := sandboxed(s.fixed("crm", id, "whoami", "secret"))
	binding.Tools[1].SchemaDigest = staleDigest

	d, tools, unavailable, err := s.attach(s.spec(s.config(binding), "", nil))

	s.Require().NoError(err)
	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
	s.Len(d.grants, 1)
}

// TestARequiredBindingWithADriftedToolSavedBeforeTheSandboxBlockRefusesTheSession: the
// sandbox the binding needs is why it fails, before a tool it is used without.
func (s *SandboxGrantsSuite) TestARequiredBindingWithADriftedToolSavedBeforeTheSandboxBlockRefusesTheSession() {
	id := s.connection("", "primary")
	s.sandboxRevision()
	binding := required(sandboxed(s.fixed("crm", id, "whoami", "secret")))
	binding.Tools[1].SchemaDigest = staleDigest

	_, _, _, err := s.attach(s.spec(s.config(binding), "", nil))

	s.ErrorContains(err, fmt.Sprintf("required connector %q cannot be used: %s", "crm", unavailableSandbox))
}

// boxed is the error of making a daytona session's sandbox of options for a sandbox binding.
func (s *SandboxGrantsSuite) boxed(options sandbox.Config) error {
	s.sandboxRevision()
	id := s.connection("", "primary")
	spec := s.spec(s.config(sandboxed(s.fixed("crm", id, "whoami"))), "", nil)
	spec.ID, spec.Sandbox, spec.SandboxOptions = uuid.NewString(), daytonaProvider, options
	spec.Attribution.SessionID = spec.ID
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	_, err = s.manager.box(s.ctx, spec, d)
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
// sandbox may reach the proxy alone, its runs go there, it holds a placeholder, and the proxy
// finds the connection's token behind it, for this session, and the allowed domains. No secret
// prefix is needed. A binding the config drops gets no token at the next request, and gets it
// again once put back before a run (A16). A connection deleted mid-session is withdrawn, in the
// session's log.
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
	s.manager.options.SandboxProxy = egress.BrokerOptions{Store: egress.NewStore(redis), URL: "http://203.0.113.7:8091",
		CACert: []byte(proxyCA)}
	s.sandboxRevision()
	id := s.connection("", "primary")
	binding := sandboxed(s.fixed("crm", id, "whoami"))
	config := s.config(binding)
	spec := s.spec(config, "", nil)
	spec.ID, spec.Sandbox, spec.SandboxOptions = uuid.NewString(), daytonaProvider, sandbox.Config{AllowedDomains: []string{"PyPI.org", "Files.PythonHosted.org"}}
	spec.Attribution.SessionID = spec.ID
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	box, err := s.manager.box(s.ctx, spec, d)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })

	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal("203.0.113.7/32", api.created["networkAllowList"])
	s.NotContains(api.created, "domainAllowList")
	proxy, err := url.Parse(api.ran["envs"].(map[string]any)["HTTPS_PROXY"].(string))
	s.Require().NoError(err)
	s.Equal("203.0.113.7:8091", proxy.Host)
	credential, _ := proxy.User.Password()
	sum := sha256.Sum256([]byte(credential))
	raw, err := redis.Do(s.ctx, redis.B().Get().Key("sandbox:egress:"+hex.EncodeToString(sum[:])).Build()).AsBytes()
	s.Require().NoError(err)
	var record egress.Record
	s.Require().NoError(json.Unmarshal(raw, &record))
	s.Equal(spec.ID, record.Session)
	s.Equal([]string{"pypi.org", "files.pythonhosted.org"}, record.Allowed)
	s.Require().Len(record.Grants, 1)
	s.Equal(api.created["env"].(map[string]any)["CRM_TOKEN"], record.Grants[0].Placeholder)
	tokens := SandboxProxyTokens(s.resolver, s.store, s.registry.Schemes)
	token, err := tokens(s.ctx, record.Grants[0])
	s.Require().NoError(err)
	s.Equal(tokenOf("primary"), token)
	s.NotContains(string(raw), tokenOf("primary"))

	s.rebind(config)
	_, err = tokens(s.ctx, record.Grants[0])
	s.Require().ErrorIs(err, sandbox.ErrNoCredential)
	s.rebind(config, binding)
	token, err = tokens(s.ctx, record.Grants[0])
	s.Require().NoError(err)
	s.Equal(tokenOf("primary"), token)

	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, id))
	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Contains(s.logs.String(), "credential is withdrawn from the sandbox")
}

// TestAManagerWithTheProxyBrokerNeedsItsStore: the router does not start without it, rather
// than refuse every sandbox session.
func (s *SandboxGrantsSuite) TestAManagerWithTheProxyBrokerNeedsItsStore() {
	_, err := NewManager(ManagerOptions{SandboxBroker: egress.BrokerName,
		SandboxProxy: egress.BrokerOptions{URL: "http://203.0.113.7:8091", CACert: []byte(proxyCA)}})

	s.ErrorContains(err, "the proxy sandbox broker needs the store")
}

// TestTheProxyBrokerRefusesAConfigItCannotServe: the proxy reaches hosts by name, so an open
// network or CIDR ranges would reach the grants alone. Daytona's broker takes both, and the
// proxy takes a network narrowed to domains.
func (s *SandboxGrantsSuite) TestTheProxyBrokerRefusesAConfigItCannotServe() {
	open := sandbox.Config{Network: true}
	ranges := sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8"}}
	s.NoError(s.boxed(open))
	s.NoError(s.boxed(ranges))
	s.manager.options.SandboxBroker = egress.BrokerName
	s.manager.options.SandboxProxy = egress.BrokerOptions{Store: &egress.Store{}, URL: "http://203.0.113.7:8091", CACert: []byte(proxyCA)}

	err := s.boxed(open)
	s.ErrorContains(err, "sandbox_options.network is true")
	s.ErrorContains(err, "code_sandbox.broker is proxy")
	s.ErrorContains(s.boxed(ranges), "sandbox_options.network_allow_list is set")
	s.ErrorContains(s.boxed(sandbox.Config{Network: true, NetworkAllowList: []string{"10.0.0.0/8"}}), "sandbox_options.network_allow_list is set")
	s.NoError(s.boxed(sandbox.Config{Network: true, AllowedDomains: []string{"pypi.org"}}))
	s.NoError(s.boxed(sandbox.Config{}))
}

// daytonaAPI stands in for Daytona's API: it creates sandboxes, runs code, and keeps the
// organization's secrets by name, and values their values. secrets maps the env vars of the
// last sandbox created to the secret names they read.
type daytonaAPI struct {
	*httptest.Server
	mu      sync.Mutex
	held    map[string]string
	values  map[string]string
	secrets map[string]string
	// created is the last request for a sandbox, and ran for a run of code.
	created, ran map[string]any
}

func (s *SandboxGrantsSuite) daytonaAPI() *daytonaAPI {
	api := &daytonaAPI{held: map[string]string{}, values: map[string]string{}}
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
		var body struct{ Name, Value string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		a.held["secret-"+body.Name], a.values["secret-"+body.Name] = body.Name, body.Value
		_, _ = fmt.Fprintf(w, `{"id":"secret-%s","name":%q,"placeholder":"dtn_secret_1","hosts":[],"createdAt":"2026-10-21T00:00:00Z","updatedAt":"2026-10-21T00:00:00Z"}`, body.Name, body.Name)
	case r.Method == http.MethodPatch && id != r.URL.Path:
		var body struct{ Value string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		a.values[id] = body.Value
		_, _ = fmt.Fprintf(w, `{"id":%q,"name":%q,"placeholder":"dtn_secret_1","hosts":[],"createdAt":"2026-10-21T00:00:00Z","updatedAt":"2026-10-21T00:00:00Z"}`, id, a.held[id])
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
	case strings.HasSuffix(r.URL.Path, "/process/code-run"):
		_ = json.NewDecoder(r.Body).Decode(&a.ran)
		_, _ = w.Write([]byte(`{"result":"","exitCode":0}`))
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

// value is what Daytona's secret for the env var the sandbox reads holds.
func (a *daytonaAPI) value(env string) string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.values["secret-"+a.secrets[env]]
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
	s.Equal("va_withdrawn", api.value("CRM_TOKEN"), "Daytona's secret holds the stand-in until Close")
	s.Contains(s.logs.String(), "credential is withdrawn from the sandbox")
}

// TestABindingRemovedMidSessionIsWithdrawnBeforeTheNextRun is A16 under Daytona's broker: once
// the config drops the binding, the next run_code finds no credential behind the sandbox's env
// var. Putting the binding back gives this sandbox nothing again: the next session gets it.
func (s *SandboxGrantsSuite) TestABindingRemovedMidSessionIsWithdrawnBeforeTheNextRun() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	id := s.connection("", "primary")
	binding := sandboxed(s.fixed("crm", id, "whoami"))
	config := s.config(binding)
	box, _ := s.opened(config)
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Require().True(api.offered("CRM_TOKEN"))

	s.rebind(config)
	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal("va_withdrawn", api.value("CRM_TOKEN"), "Daytona's secret holds the stand-in until Close")
	s.Contains(s.logs.String(), "credential is withdrawn from the sandbox")
	s.rebind(config, binding)
	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Equal("va_withdrawn", api.value("CRM_TOKEN"))
}

// TestABindingTheConfigNoLongerHoldsAsOpenedGivesNoCredential: each way the config can stop
// binding a grant as the session opened it withdraws it, as recheck refuses a tool call. A
// config that cannot be read just now does not.
func (s *SandboxGrantsSuite) TestABindingTheConfigNoLongerHoldsAsOpenedGivesNoCredential() {
	s.sandboxRevision()
	id := s.connection("", "primary")
	config := s.config(sandboxed(s.fixed("crm", id, "whoami")))
	box, _ := s.opened(config)
	grant := box.(sandbox.Granted).Grants()[0]
	s.Require().NoError(SandboxBound(s.ctx, s.store, grant))
	connector, selection, connection := grant, grant, grant
	connector.Connector = uuid.NewString()
	selection.Selection = selectionSession
	connection.Connection.ConnectionID = s.connection("", "secondary")
	for name, changed := range map[string]sandbox.Grant{"connector": connector, "selection": selection, "connection": connection} {
		s.ErrorIs(SandboxBound(s.ctx, s.store, changed), sandbox.ErrNoCredential, name)
	}
	chosen := s.config(sandboxed(s.chosen("crm", "whoami")))
	selection.Config = chosen.ID
	s.NoError(SandboxBound(s.ctx, s.store, selection), "a session binding takes the caller's connection")

	s.rebind(config, s.fixed("crm", id, "whoami"))
	s.ErrorIs(SandboxBound(s.ctx, s.store, grant), sandbox.ErrNoCredential, "sandbox unset")
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, s.customerID, config.ID))
	s.ErrorIs(SandboxBound(s.ctx, s.store, grant), sandbox.ErrNoCredential, "config deleted")
	closed, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	err = SandboxBound(s.ctx, closed, grant)
	s.Require().Error(err)
	s.NotErrorIs(err, sandbox.ErrNoCredential)
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

// TestAConfigThatCannotBeReadJustNowGivesNoTokenAndKeepsTheSecret: the check that the config
// still binds the grant reads it first. When it cannot, the broker gets no token and no
// withdrawal, so it keeps the secret it has without renewing it, though the connection holds a
// newer credential.
func (s *SandboxGrantsSuite) TestAConfigThatCannotBeReadJustNowGivesNoTokenAndKeepsTheSecret() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	id := s.connection("", "primary")
	box, d := s.opened(s.config(sandboxed(s.fixed("crm", id, "whoami"))))
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Require().True(api.offered("CRM_TOKEN"))
	grant := box.(sandbox.Granted).Grants()[0]
	closed, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	s.manager.options.Store = closed

	token, err := s.manager.sandboxTokens(d.schemes)(s.ctx, grant)
	s.Require().Error(err)
	s.NotErrorIs(err, sandbox.ErrNoCredential)
	s.Empty(token.Value)
	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.True(api.offered("CRM_TOKEN"))
	s.Contains(s.logs.String(), "it keeps the one it has")
}

// chat is a daytona session of config for alice, kept in a conversation she can be asked to
// log in in, on a router whose consents begin begun.
func (s *SandboxGrantsSuite) chat(config store.AgentConfig, begun Consent) Spec {
	s.manager.options.Connectors.Consents = func(context.Context, ConsentRequest) (Consent, error) { return begun, nil }
	spec := s.spec(config, "alice", nil)
	spec.ID, spec.Sandbox, spec.PersistConversation, spec.Text = uuid.NewString(), daytonaProvider, true, true
	spec.Attribution.SessionID = spec.ID
	return spec
}

// chatSandbox is the sandbox of chat's session, with its connectors attached, whose
// conversation shows every consent.
func (s *SandboxGrantsSuite) chatSandbox(config store.AgentConfig, begun Consent) (sandbox.Sandbox, *dispatcher) {
	spec := s.chat(config, begun)
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	s.T().Cleanup(d.closeLogins)
	if d.logins != nil {
		d.logins.canAsk = func(string) bool { return true }
		d.logins.ask = func(string, persistent.ConnectorAuthorization) bool { return true }
	}
	box, err := s.manager.box(s.ctx, spec, d)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box, d
}

// consent is a consent of alice's on connection, which her chat shows.
func consent(connection string) Consent {
	return Consent{ConnectionID: connection, AuthorizationID: "attempt-alice", Name: "CRM", ExpiresAt: time.Now().Add(time.Minute)}
}

// bindings are the grants' bindings.
func bindings(grants []sandbox.Grant) []string {
	var named []string
	for _, grant := range grants {
		named = append(named, grant.Binding)
	}
	return named
}

// TestASessionBindingGivesTheSandboxTheCallersOwnConnection: the caller's one connected
// connection is the one the binding opens on, and its token is the one Daytona gets.
func (s *SandboxGrantsSuite) TestASessionBindingGivesTheSandboxTheCallersOwnConnection() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	s.connection("bob", "secondary")
	id := s.connection("alice", "primary")
	config := s.config(sandboxed(s.chosen("crm", "whoami")))

	box, _ := s.chatSandbox(config, Consent{})
	result, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	// Both as on base 68b5eb8c2 (probe, 2026-10-26).
	s.Equal(sandbox.Result{}, result)
	said, _ := harness.HandBack(s.ctx, nil, result, s.manager.logger)
	s.Equal("The code ran and printed nothing.", said)
	s.Equal([]sandbox.Grant{{Connection: core.ConnectionRef{CustomerID: s.customerID, ConnectionID: id}, Binding: "crm",
		Config: config.ID, Connector: s.connectorID, Selection: selectionSession, Scheme: bearer.Name, Env: "CRM_TOKEN",
		Hosts: []string{"api.crm.example", "*.crm.example"}}}, box.(sandbox.Granted).Grants())
	s.Equal(tokenOf("primary"), api.value("CRM_TOKEN"))
	s.NotContains(harness.SandboxTool(box).Description, "has not connected")
}

// TestACallerWithNoConnectionGetsALateGrantAndIsToldHowToConnect: the sandbox has the env var
// from the start, holding no token, and run_code says what a 401 means and which tool connects.
func (s *SandboxGrantsSuite) TestACallerWithNoConnectionGetsALateGrantAndIsToldHowToConnect() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	s.connection("bob", "primary")
	config := s.config(sandboxed(s.chosen("crm", "whoami")))

	box, d := s.chatSandbox(config, Consent{})
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	result, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal(sandbox.Result{Unconnected: box.(sandbox.Granted).Grants()}, result, "every run says so until the user connects")
	said, _ := harness.HandBack(s.ctx, nil, result, s.manager.logger)
	s.Equal("The code ran and printed nothing.\n\nIf this code needed crm (a 401 from api.crm.example, *.crm.example), the "+
		"user has not connected it: call crm__list_tools so they can, then run the code again; if you cannot call it, tell "+
		"the user they must connect crm.", said)
	s.Equal([]sandbox.Grant{{Binding: "crm", Config: config.ID, Connector: s.connectorID, Selection: selectionSession,
		Env: "CRM_TOKEN", Hosts: []string{"api.crm.example", "*.crm.example"}, Late: true}}, box.(sandbox.Granted).Grants())
	s.Equal([]string{"crm__list_tools", "crm__call_tool"}, names(d.tools))
	s.True(api.offered("CRM_TOKEN"))
	s.NotEmpty(api.value("CRM_TOKEN"))
	s.NotContains(api.value("CRM_TOKEN"), "token-of-")
	s.Contains(harness.SandboxTool(box).Description, " A 401 from api.crm.example, *.crm.example means the user has not "+
		"connected crm yet: crm__list_tools shows them a button to connect it and waits; if you cannot call it, tell the "+
		"user they must connect crm. Once they have, run the code again; if it still gets 401, wait 20 seconds and run it once more.")
}

// TestARunNamesOnlyTheLateGrantsNotConnected: the app's own connection, bound from the start,
// needs no login.
func (s *SandboxGrantsSuite) TestARunNamesOnlyTheLateGrantsNotConnected() {
	s.daytonaAPI()
	s.sandboxRevision()
	app := s.connection("", "secondary")
	box, _ := s.chatSandbox(s.config(sandboxed(s.fixed("app", app, "whoami")), sandboxed(s.chosen("crm", "whoami"))), Consent{})

	result, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal([]string{"app", "crm"}, bindings(box.(sandbox.Granted).Grants()))
	s.Equal(sandbox.Result{Unconnected: box.(sandbox.Granted).Grants()[1:]}, result)
}

// TestALoginFinishedInTheChatBindsTheGrantBeforeTheNextRun: the consent the model's
// crm__list_tools began connects alice's account, and the next run's secret holds its token.
func (s *SandboxGrantsSuite) TestALoginFinishedInTheChatBindsTheGrantBeforeTheNextRun() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	mine := s.pending("alice", "primary")
	box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), consent(mine))
	asked, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)
	s.Require().Contains(asked, `"status":"authorization_required"`)
	unbound, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Require().NotEqual(tokenOf("primary"), api.value("CRM_TOKEN"))
	connectedBefore := box.(sandbox.Connecting).Connected()

	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	connected := box.(sandbox.Connecting).Connected()
	_, err = box.Run(s.ctx, " ", nil)
	s.Require().Error(err, "a run that never ran says nothing, and leaves the line to the next")
	bound, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	later, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.True(finished)
	s.Equal(tokenOf("primary"), api.value("CRM_TOKEN"))
	s.Equal([]string{"crm"}, bindings(unbound.Unconnected))
	s.Empty(connectedBefore)
	s.Equal([]string{"crm"}, connected, "the agent runs the code that got the 401 again")
	s.Equal(sandbox.Result{Connected: []string{"crm"}}, bound, "a Daytona secret's token is still on its way")
	s.Equal(sandbox.Result{}, later, "and said once")
	s.NotContains(s.logs.String(), tokenOf("primary"))
}

// TestALoginFinishedWhileCodeRunsIsSaidAgainByTheRunThatBindsIt: the login lands during the
// code-run request, after that run's BeforeRun, so its secret still has no token; the next
// run's BeforeRun puts it there and says so again.
func (s *SandboxGrantsSuite) TestALoginFinishedWhileCodeRunsIsSaidAgainByTheRunThatBindsIt() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	mine := s.pending("alice", "primary")
	box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), consent(mine))
	_, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)
	var finished bool
	var once sync.Once
	api.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/process/code-run") {
			once.Do(func() {
				s.connect(mine, "primary")
				_, finished, _ = d.loginFinished(s.ctx, "attempt-alice", mine)
			})
		}
		api.serve(w, r)
	})

	during, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	unboundDuring := api.value("CRM_TOKEN") != tokenOf("primary")
	next, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	later, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.True(finished)
	s.True(unboundDuring, "the run during the login bound nothing")
	s.Equal(sandbox.Result{Connected: []string{"crm"}}, during, "connected, not to be connected again")
	s.Equal(sandbox.Result{Connected: []string{"crm"}}, next, "this run put the token in the secret")
	s.Equal(tokenOf("primary"), api.value("CRM_TOKEN"))
	s.Equal(sandbox.Result{}, later)
}

// TestALoginOnARevisionWithoutTheBlockLeavesTheGrantUnbound: alice's connection pins revision
// 1, which gives the sandbox nothing, so her login opens the tools and binds no token.
func (s *SandboxGrantsSuite) TestALoginOnARevisionWithoutTheBlockLeavesTheGrantUnbound() {
	api := s.daytonaAPI()
	mine := s.pending("alice", "primary")
	s.sandboxRevision()
	box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), consent(mine))
	_, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)

	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	result, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.True(finished)
	s.NotEqual(tokenOf("primary"), api.value("CRM_TOKEN"))
	s.Contains(s.logs.String(), "reconnect to use sandbox access")
	s.Equal(sandbox.Result{}, result, "connected, so not to be connected again, and not bound")
}

// TestALoginOnARevisionWithOtherHostsLeavesTheGrantUnbound: revision 3 names other hosts
// after alice's connection was saved on revision 2. The late grant reads revision 3, her
// connection's revision decides what the sandbox may reach, so her token goes to neither.
func (s *SandboxGrantsSuite) TestALoginOnARevisionWithOtherHostsLeavesTheGrantUnbound() {
	for name, block := range map[string]string{
		"hosts": "  env: CRM_TOKEN\n  hosts: [api.other.example]\n",
		"env":   "  env: OTHER_TOKEN\n  hosts: [api.crm.example, \"*.crm.example\"]\n",
	} {
		s.Run(name, func() {
			s.SetupTest()
			api := s.daytonaAPI()
			s.sandboxRevision()
			mine := s.pending("alice", "primary")
			manifest, err := core.ParseManifest([]byte(s.crmManifest(s.connectorID, 3) + "\nsandbox:\n" + block))
			s.Require().NoError(err)
			_, err = s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
			s.Require().NoError(err)
			box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), consent(mine))
			_, err = s.call(d, "crm__list_tools", `{}`)
			s.Require().NoError(err)

			s.connect(mine, "primary")
			_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
			result, err := box.Run(s.ctx, "print(1)", nil)

			s.Require().NoError(err)
			s.True(finished)
			s.Equal(sandbox.Result{}, result)
			env := box.(sandbox.Granted).Grants()[0].Env
			s.NotEqual(tokenOf("primary"), api.value(env))
			s.Contains(s.logs.String(), "reconnect to use sandbox access")
		})
	}
}

// TestALoginBindsOnlyItsOwnBinding: alice's login on crm opens crm alone; the late grant of
// crm2, a second binding of the same connector with the same env var and hosts, stays unbound.
// The run after it opens crm2 on her one connected connection, as list_tools would (openConnected).
func (s *SandboxGrantsSuite) TestALoginBindsOnlyItsOwnBinding() {
	s.daytonaAPI()
	s.sandboxRevision()
	mine := s.pending("alice", "primary")
	box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami")), sandboxed(s.chosen("crm2", "whoami"))), consent(mine))
	_, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)

	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	grants := box.(sandbox.Granted).Grants()
	_, crm := d.boundLate(s.ctx, grants[0])
	_, crm2 := d.boundLate(s.ctx, grants[1])
	result, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Equal(sandbox.Result{Unconnected: box.(sandbox.Granted).Grants()[1:], Connected: []string{"crm"}}, result)
	_, crm2Later := d.boundLate(s.ctx, grants[1])
	opened := d.logins.byAlias["crm2"].opened
	d.offerConsent(s.ctx, d.logins.byAlias["crm2"])

	s.True(finished)
	s.Equal([]string{"crm", "crm2"}, []string{grants[0].Binding, grants[1].Binding})
	s.True(crm)
	s.False(crm2)
	s.True(crm2Later, "the run opened crm2 on her connection with no consent")
	s.Same(opened, d.logins.byAlias["crm2"].opened, "and a later one opens it no second time")
}

// TestAnotherUsersConnectionNeverBindsTheGrant: a consent that came back on bob's connection,
// or bob's own consent finishing, opens nothing in alice's session, so her sandbox gets no
// token though bob's connection is connected.
func (s *SandboxGrantsSuite) TestAnotherUsersConnectionNeverBindsTheGrant() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	bobs := s.pending("bob", "primary")
	box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), consent(bobs))
	_, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)

	s.connect(bobs, "primary")
	_, onBobs, _ := d.loginFinished(s.ctx, "attempt-alice", bobs)
	_, bobsOwn, _ := d.loginFinished(s.ctx, "attempt-bob", bobs)
	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.False(onBobs)
	s.False(bobsOwn)
	s.NotEqual(tokenOf("primary"), api.value("CRM_TOKEN"))
}

// TestARequiredSessionBindingWithNoConnectionStillRefusesTheSession: a late grant is never
// required; the session fails as on base.
func (s *SandboxGrantsSuite) TestARequiredSessionBindingWithNoConnectionStillRefusesTheSession() {
	s.sandboxRevision()

	_, _, _, err := s.attach(s.chat(s.config(required(sandboxed(s.chosen("crm", "whoami")))), Consent{}))

	s.ErrorContains(err, fmt.Sprintf("required connector %q cannot be used: %s", "crm", unavailableNoSelection))
}

// TestASessionWithNoConversationGetsNoLateGrant: a voice session has nowhere to show a
// consent, so the binding is not available and the sandbox gets nothing, as on base.
func (s *SandboxGrantsSuite) TestASessionWithNoConversationGetsNoLateGrant() {
	s.sandboxRevision()
	spec := s.chat(s.config(sandboxed(s.chosen("crm", "whoami"))), Consent{})
	spec.PersistConversation = false

	d, tools, unavailable, err := s.attach(spec)

	s.Require().NoError(err)
	s.Empty(d.grants)
	s.Empty(tools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableNoSelection}}, unavailable)
}

// TestALoginWithoutSandboxOrABlockGivesTheSandboxNothing: a binding without sandbox, and one
// whose connector has no sandbox block or was deleted, wait for a login with no grant.
func (s *SandboxGrantsSuite) TestALoginWithoutSandboxOrABlockGivesTheSandboxNothing() {
	noBlock, _ := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), Consent{})
	s.sandboxRevision()
	unset, _ := s.chatSandbox(s.config(s.chosen("crm", "whoami")), Consent{})
	gone := s.config(sandboxed(s.chosen("crm", "whoami")))
	_, err := s.store.DeleteConnectorDefinition(s.ctx, s.customerID, s.connectorID, true)
	s.Require().NoError(err)
	deleted, d := s.chatSandbox(gone, Consent{})

	s.Empty(unset.(sandbox.Granted).Grants())
	s.Empty(noBlock.(sandbox.Granted).Grants())
	s.Empty(deleted.(sandbox.Granted).Grants())
	s.Equal([]string{"crm__list_tools", "crm__call_tool"}, names(d.tools))
	s.Contains(s.logs.String(), "its connector offers none")
}

// TestALoginFinishingWhileCodeRunsBindsOnce, under -race: runs, and the broker's bind alone,
// ask for the binding while the login opens it. The bare bind shares no lock with the login but
// the login's own, so -race sees an unguarded read; a run also takes the store's.
func (s *SandboxGrantsSuite) TestALoginFinishingWhileCodeRunsBindsOnce() {
	api := s.daytonaAPI()
	s.sandboxRevision()
	mine := s.pending("alice", "primary")
	box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), consent(mine))
	_, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)
	s.connect(mine, "primary")

	var wg sync.WaitGroup
	errs := make(chan error, 10)
	connected := make(chan []string, 11)
	stop := make(chan struct{})
	for range 2 {
		wg.Go(func() {
			for range 5 {
				result, err := box.Run(s.ctx, "print(1)", nil)
				errs <- err
				connected <- result.Connected
			}
		})
	}
	wg.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
				d.boundLate(s.ctx, sandbox.Grant{Binding: "crm", Late: true})
			}
		}
	})
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	close(stop)
	wg.Wait()
	close(errs)
	result, err := box.Run(s.ctx, "print(1)", nil)
	connected <- result.Connected
	close(connected)

	for err := range errs {
		s.NoError(err)
	}
	var told []string
	for bindings := range connected {
		told = append(told, bindings...)
	}
	after, afterErr := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.Require().NoError(afterErr)
	s.True(finished)
	s.NotEmpty(told, "a run during the login, and the one that binds it, say it was just connected")
	s.Subset([]string{"crm"}, told)
	s.Equal(sandbox.Result{}, after, "and none after the one that bound it")
	s.Equal(tokenOf("primary"), api.value("CRM_TOKEN"))
}

// TestTheProxyBrokerBindsALateGrantOnceTheLoginFinishes: the proxy's record holds the late
// grant with no connection, so the proxy refuses its placeholder, until the run after the
// login, which puts alice's connection behind the same placeholder.
func (s *SandboxGrantsSuite) TestTheProxyBrokerBindsALateGrantOnceTheLoginFinishes() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	redis, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(redis.Close)
	api := s.daytonaAPI()
	s.manager.options.SandboxBroker, s.manager.options.SandboxSecretPrefix = egress.BrokerName, ""
	s.manager.options.SandboxProxy = egress.BrokerOptions{Store: egress.NewStore(redis), URL: "http://203.0.113.7:8091",
		CACert: []byte(proxyCA)}
	s.sandboxRevision()
	mine := s.pending("alice", "primary")
	box, d := s.chatSandbox(s.config(sandboxed(s.chosen("crm", "whoami"))), consent(mine))
	_, err = s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)
	var result sandbox.Result
	record := func() egress.Record {
		result, err = box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
		proxy, err := url.Parse(api.ran["envs"].(map[string]any)["HTTPS_PROXY"].(string))
		s.Require().NoError(err)
		credential, _ := proxy.User.Password()
		sum := sha256.Sum256([]byte(credential))
		raw, err := redis.Do(s.ctx, redis.B().Get().Key("sandbox:egress:"+hex.EncodeToString(sum[:])).Build()).AsBytes()
		s.Require().NoError(err)
		var record egress.Record
		s.Require().NoError(json.Unmarshal(raw, &record))
		s.Require().Len(record.Grants, 1)
		return record
	}
	unbound := record()
	s.Equal(sandbox.Result{Unconnected: box.(sandbox.Granted).Grants()}, result)

	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	bound := record()

	s.True(finished)
	s.Equal(sandbox.Result{}, result, "the proxy sends the token from the run's first request")
	s.Empty(unbound.Grants[0].Connection.ConnectionID)
	s.Equal(core.ConnectionRef{CustomerID: s.customerID, ConnectionID: mine}, bound.Grants[0].Connection)
	s.Equal(bearer.Name, bound.Grants[0].Scheme)
	s.Equal(api.created["env"].(map[string]any)["CRM_TOKEN"], bound.Grants[0].Placeholder)
	s.Equal(unbound.Grants[0].Placeholder, bound.Grants[0].Placeholder)
	token, err := SandboxProxyTokens(s.resolver, s.store, s.registry.Schemes)(s.ctx, bound.Grants[0])
	s.Require().NoError(err)
	s.Equal(tokenOf("primary"), token)
}
