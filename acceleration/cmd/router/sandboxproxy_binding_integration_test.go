//go:build integration

package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// SandboxProxyBindingSuite is the proxy as the router builds it (newSandboxProxy), over a
// real store and connections resolver: a placeholder request is checked against the agent
// config on every request.
type SandboxProxyBindingSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	proxy      *httptest.Server
	trusted    *x509.CertPool
	config     store.AgentConfig
	binding    store.ConnectorBinding
	credential string
	// placeholder is what the sandbox sends in place of the token.
	placeholder string
}

func TestSandboxProxyBindingSuite(t *testing.T) {
	suite.Run(t, new(SandboxProxyBindingSuite))
}

// forgeHost is a host no DNS answers (RFC 6761), so a request let through ends in a 502
// without reaching anything.
const forgeHost = "api.forge.invalid"

func (s *SandboxProxyBindingSuite) SetupSuite() {
	dsn, address := os.Getenv("ROUTER_POSTGRES_DSN"), os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR are not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = db.Close() })
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	sealer, err := auth.NewSealerWithKeyring(1, map[int]string{1: "sandbox proxy binding test key"})
	s.Require().NoError(err)
	credentials, err := pgsealed.New(db, sealer)
	s.Require().NoError(err)
	schemes := map[string]core.Scheme{bearer.Name: bearer.New()}
	connections, err := resolver.New(resolver.Config{Store: db, Credentials: credentials, Schemes: schemes})
	s.Require().NoError(err)
	redis, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.T().Cleanup(redis.Close)

	settings := config.Defaults().CodeSandbox
	settings.ProxyAddr = ":8091"
	settings.ProxyCACert, settings.ProxyCAKey = newCA(s.T())
	proxy, err := newSandboxProxy(settings, redis, connections, db, schemes, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.proxy = httptest.NewServer(proxy)
	s.T().Cleanup(s.proxy.Close)
	s.trusted = x509.NewCertPool()
	s.Require().True(s.trusted.AppendCertsFromPEM([]byte(settings.ProxyCACert)))

	customer := "w16bind-" + uuid.NewString()
	manifest, err := core.ParseManifest([]byte(`
id: custom_forge
revision: 1
name: Forge
endpoints:
  mcp: https://forge.example/mcp
schemes: [bearer]
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	definition, err := db.CreateConnectorDefinition(s.ctx, customer, manifest)
	s.Require().NoError(err)
	connection := &store.ConnectorConnection{CustomerID: customer, ConnectorID: definition.ID, DefinitionRevision: definition.Revision,
		OwnerType: store.OwnerApp, AuthScheme: bearer.Name}
	s.Require().NoError(db.CreateConnectorConnection(s.ctx, core.Registry{Schemes: schemes}, connection))
	ref := core.ConnectionRef{CustomerID: customer, ConnectionID: connection.ID}
	stored, _, err := bearer.New().Complete(s.ctx, core.CompleteInput{Supplied: map[string]string{bearer.SuppliedToken: "gho_canary"}})
	s.Require().NoError(err)
	s.Require().NoError(credentials.Update(s.ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials, state.Status = stored, store.ConnectionConnected
		return true, nil
	}))
	s.binding = store.ConnectorBinding{Name: "forge", ConnectorID: definition.ID, Sandbox: true,
		Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: connection.ID}}
	s.config = store.AgentConfig{CustomerID: customer, Name: "agent-" + uuid.NewString(), Connectors: []store.ConnectorBinding{s.binding}}
	s.Require().NoError(db.CreateAgentConfig(s.ctx, &s.config))

	s.credential, s.placeholder = "w16bind-"+egress.NewCredential(), "va_ph_"+uuid.NewString()
	records := sandboxStore(redis)
	s.Require().NoError(records.Put(s.ctx, s.credential, egress.Record{Session: "w16bind-session", Grants: []egress.Grant{{
		Connection: ref, Binding: "forge", Config: s.config.ID, Connector: definition.ID, Selection: "fixed",
		Scheme: bearer.Name, Placeholder: s.placeholder, Hosts: []string{forgeHost}}}}, time.Minute))
	s.T().Cleanup(func() { _ = records.Delete(context.Background(), s.credential) })
}

// send is a placeholder request of the sandbox's, through the proxy, with its status and body.
func (s *SandboxProxyBindingSuite) send() (int, string) {
	proxyURL, err := url.Parse(s.proxy.URL)
	s.Require().NoError(err)
	proxyURL.User = url.UserPassword("sbx", s.credential)
	client := &http.Client{Timeout: 20 * time.Second,
		Transport: &http.Transport{Proxy: http.ProxyURL(proxyURL), TLSClientConfig: &tls.Config{RootCAs: s.trusted}}}
	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, "https://"+forgeHost+"/user", nil)
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer "+s.placeholder)
	response, err := client.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(body)
}

// TestAConfigThatNoLongerBindsTheGrantStopsThePlaceholderAtOnce: the proxy newSandboxProxy
// builds asks the store for the config on every request. While the config binds the grant
// the request gets as far as the host (the 502 of a host no DNS answers, not a refusal for
// want of a token); once it drops the binding, or sets another connector, it is a 401.
func (s *SandboxProxyBindingSuite) TestAConfigThatNoLongerBindsTheGrantStopsThePlaceholderAtOnce() {
	status, body := s.send()
	s.Equal(http.StatusBadGateway, status)
	s.NotContains(body, "gives no token")

	other := s.config
	other.Connectors = []store.ConnectorBinding{{Name: "forge", ConnectorID: uuid.NewString(), Sandbox: true,
		Connection: s.binding.Connection}}
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &other))
	status, body = s.send()
	s.Equal(http.StatusUnauthorized, status, "a connector the config no longer binds")
	s.Contains(body, "gives no token: forge")

	unbound := s.config
	unbound.Connectors = nil
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &unbound))
	status, _ = s.send()
	s.Equal(http.StatusUnauthorized, status, "a binding the config dropped")

	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &s.config))
	status, _ = s.send()
	s.Equal(http.StatusBadGateway, status, "the binding put back reaches the host again: the proxy keeps no copy")
}
