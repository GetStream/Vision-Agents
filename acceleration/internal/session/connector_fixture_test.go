//go:build integration

package session

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// fixtureRequestTimeout bounds every request a connection's client sends in these suites, as
// connectorHTTPTimeout does in the router. A choice: loopback answers in milliseconds.
const fixtureRequestTimeout = 5 * time.Second

// connectorFixture is what the connector suites run against: Postgres, the router's resolver
// over it with the bearer scheme, core.Transports, the mcp source, and an MCP server of the
// official SDK on loopback TLS that keeps one account per path. Each account takes its own
// bearer token and answers whoami with its own name, so a test can tell whose credential a
// call went with and which account it reached.
type connectorFixture struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	sealer   *auth.Sealer
	provider *accountsProvider
	manager  *Manager
	registry core.Registry
	resolver *resolver.Resolver

	// customerID, connectorID and revision are the test's own tenant and connector.
	customerID  string
	connectorID string
	revision    int
}

func (s *connectorFixture) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })

	s.sealer, err = auth.NewSealerWithKeyring(1, map[int]string{1: "connector session test key"})
	s.Require().NoError(err)
	s.provider = newAccountsProvider(s)
	s.registry = core.Registry{
		Schemes:     map[string]core.Scheme{bearer.Name: bearer.New()},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	credentials, err := pgsealed.New(db, s.sealer)
	s.Require().NoError(err)
	s.resolver, err = resolver.New(resolver.Config{Store: db, Credentials: credentials, Schemes: s.registry.Schemes})
	s.Require().NoError(err)
	s.manager = s.managerWith(fixtureRequestTimeout)
}

// managerWith is a manager whose connections' clients bound each request by requestTimeout,
// as connectorHTTPTimeout does in the router. egress refuses loopback, so the clients reach
// the provider over its own transport.
func (s *connectorFixture) managerWith(requestTimeout time.Duration) *Manager {
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: s.resolver, Timeout: requestTimeout,
		NewClient: func(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
			return &http.Client{Transport: wrap(s.provider.Client().Transport), Timeout: timeout}
		}})
	s.Require().NoError(err)
	logger := slog.New(slog.DiscardHandler)
	invocations := newInvocationRecorder(s.store, logger)
	s.T().Cleanup(invocations.Close)
	return &Manager{logger: logger, invocations: invocations, options: ManagerOptions{
		Store: s.store, Logger: logger, Connectors: Connectors{Registry: s.registry, Transports: transports},
	}}
}

// SetupTest makes the test's own tenant and a connector of it whose MCP endpoint is the
// provider's, at the path its account input names.
func (s *connectorFixture) SetupTest() {
	s.provider.forget()
	s.customerID = "connectors-" + uuid.NewString()
	s.connectorID = "custom_crm"
	manifest, err := core.ParseManifest([]byte(fmt.Sprintf(`
id: %s
revision: 1
name: CRM
inputs:
  - name: account
    enum: [primary, secondary, moved]
endpoints:
  mcp: %s/{account}/mcp
schemes: [bearer]
sources:
  - kind: mcp
    endpoint: mcp
`, s.connectorID, s.provider.URL)))
	s.Require().NoError(err)
	definition, err := s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	s.revision = definition.Revision
}

// connection is a connected bearer connection to account, owned by the app when user is
// empty and by user otherwise, holding the account's token.
func (s *connectorFixture) connection(user, account string) string {
	connection := &store.ConnectorConnection{
		CustomerID: s.customerID, ConnectorID: s.connectorID, DefinitionRevision: s.revision,
		OwnerType: store.OwnerApp, AuthScheme: bearer.Name, Inputs: map[string]string{"account": account},
	}
	if user != "" {
		connection.OwnerType, connection.OwnerID = store.OwnerUser, user
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, s.registry, connection))
	stored, _, err := bearer.New().Complete(s.ctx, core.CompleteInput{Supplied: map[string]string{bearer.SuppliedToken: tokenOf(account)}})
	s.Require().NoError(err)
	s.setState(connection.ID, func(state *core.CredentialState) {
		state.Credentials, state.Status = stored, store.ConnectionConnected
	})
	return connection.ID
}

// setState changes a connection's credential state under the lock, as the router does.
func (s *connectorFixture) setState(id string, change func(state *core.CredentialState)) {
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	s.Require().NoError(credentials.Update(s.ctx, core.ConnectionRef{CustomerID: s.customerID, ConnectionID: id},
		func(state *core.CredentialState, _ func() error) (bool, error) {
			change(state)
			return true, nil
		}))
}

// config stores an agent config of the test's tenant with bindings.
func (s *connectorFixture) config(bindings ...store.ConnectorBinding) store.AgentConfig {
	config := store.AgentConfig{CustomerID: s.customerID, Name: "agent-" + uuid.NewString(), Connectors: bindings}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))
	return config
}

// rebind stores the config with bindings in place of its own.
func (s *connectorFixture) rebind(config store.AgentConfig, bindings ...store.ConnectorBinding) {
	config.Connectors = bindings
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &config))
}

// fixed and chosen are bindings of the test's connector under alias that grant tools: one
// to the app's connection id, one to the connection the session picks.
func (s *connectorFixture) fixed(alias, id string, tools ...string) store.ConnectorBinding {
	return store.ConnectorBinding{Name: alias, ConnectorID: s.connectorID,
		Connection: store.ConnectionBinding{Type: selectionFixed, ConnectionID: id}, Tools: grants(tools...)}
}

func (s *connectorFixture) chosen(alias string, tools ...string) store.ConnectorBinding {
	return store.ConnectorBinding{Name: alias, ConnectorID: s.connectorID,
		Connection: store.ConnectionBinding{Type: selectionSession}, Tools: grants(tools...)}
}

// required is binding, required.
func required(binding store.ConnectorBinding) store.ConnectorBinding {
	binding.Required = true
	return binding
}

// spec is a session of config for a verified end user, choosing connections by alias.
func (s *connectorFixture) spec(config store.AgentConfig, user string, chosen map[string]string) Spec {
	spec := Spec{CustomerID: s.customerID, ConfigID: config.ID, ConnectorBindings: config.Connectors,
		Caller: routing.Caller{UserID: user}, CallerKind: auth.KindAuthenticated}
	for alias, id := range chosen {
		spec.ConnectorSelections = append(spec.ConnectorSelections, ConnectorSelection{Name: alias, ConnectionID: id})
	}
	return spec
}

// attach opens spec's connectors and closes them when the test ends.
func (s *connectorFixture) attach(spec Spec) (*dispatcher, []harness.Tool, []ConnectorUnavailable, error) {
	d, tools, unavailable, err := s.manager.attachConnectors(s.ctx, &spec)
	if d != nil {
		s.T().Cleanup(d.Close)
	}
	return d, tools, unavailable, err
}

// call runs one tool through d as the model would ask for it.
func (s *connectorFixture) call(d *dispatcher, name, arguments string) (string, error) {
	parts, err := d.Run(s.ctx, llm.ToolCall{ID: uuid.NewString(), Name: name, Arguments: arguments})
	return llm.TextOf(parts), err
}

// names are the names of tools.
func names(tools []harness.Tool) []string {
	var named []string
	for _, tool := range tools {
		named = append(named, tool.Name)
	}
	return named
}

// The provider's tools, as it lists them, so a grant pins the digest the source computes.
var (
	toolWhoami = &mcpsdk.Tool{Name: "whoami", Description: "Says which account this is.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}
	toolSecret = &mcpsdk.Tool{Name: "secret", Description: "Something no grant names.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}
	toolSlow = &mcpsdk.Tool{Name: "slow", Description: "Takes as long as it is let.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}
	toolEcho = &mcpsdk.Tool{Name: "echo", Description: "Answers with the note it was sent.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{"note": map[string]any{"type": "string"}}}}
	toolFails = &mcpsdk.Tool{Name: "fails", Description: "Reports its own failure.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}
	// toolGuarded is never run: the provider refuses every call of it with a 403 that asks for
	// the scope the call names, or a 401 with the claims challenge it names (newAccountsProvider).
	toolGuarded = &mcpsdk.Tool{Name: "guarded", Description: "Needs a scope no grant has.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{
			"scope": map[string]any{"type": "string"}, "claims": map[string]any{"type": "string"}}}}
	providerTools = map[string]*mcpsdk.Tool{"whoami": toolWhoami, "secret": toolSecret, "slow": toolSlow,
		"echo": toolEcho, "fails": toolFails, "guarded": toolGuarded}
)

// grants grant each of the provider's tools by name, at its digest.
func grants(tools ...string) []store.ToolGrant {
	granted := []store.ToolGrant{}
	for _, name := range tools {
		digest, err := mcp.ToolSchemaDigest(providerTools[name])
		if err != nil {
			panic(err)
		}
		granted = append(granted, store.ToolGrant{Name: name, SchemaDigest: digest})
	}
	return granted
}

// tokenOf is the bearer token an account takes. Synthetic.
func tokenOf(account string) string {
	return "token-of-" + account
}

// accountsProvider is an MCP server per account, at /{account}/mcp, each taking only its
// account's token. It keeps what it was sent, so a test can tell whether a call reached it.
type accountsProvider struct {
	*httptest.Server
	mu sync.Mutex
	// called counts tools/call by account; methods is every JSON-RPC method by account.
	called  map[string]int
	methods map[string][]string
	// streams answers each request as an SSE stream whose headers go out at once, before the
	// tool has answered, as a server that streams progress does. Off answers in one JSON
	// object when the tool is done.
	streams bool
}

// slowFor is how long slow takes before it answers, longer than any test lets it run; it
// answers sooner when its request is cancelled.
const slowFor = 2 * time.Second

func newAccountsProvider(s *connectorFixture) *accountsProvider {
	p := &accountsProvider{called: map[string]int{}, methods: map[string][]string{}}
	servers := map[string]*mcpsdk.Server{}
	for _, account := range []string{"primary", "secondary", "moved"} {
		server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: account, Version: "1"}, nil)
		server.AddTool(toolWhoami, func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: account}}}, nil
		})
		server.AddTool(toolSecret, func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "the secret"}}}, nil
		})
		server.AddTool(toolSlow, func(ctx context.Context, _ *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			select {
			case <-time.After(slowFor):
			case <-ctx.Done():
			}
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "done"}}}, nil
		})
		server.AddTool(toolEcho, func(_ context.Context, request *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			var sent struct {
				Note string `json:"note"`
			}
			_ = json.Unmarshal(request.Params.Arguments, &sent)
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "you said " + sent.Note}}}, nil
		})
		server.AddTool(toolFails, func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return &mcpsdk.CallToolResult{IsError: true, Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "the record is locked"}}}, nil
		})
		server.AddTool(toolGuarded, func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "guarded ran"}}}, nil
		})
		servers[account] = server
	}
	p.Server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		account := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/"), "/mcp")
		server, found := servers[account]
		if !found {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+tokenOf(account) {
			// RFC 6750 section 3.1: a token that is not this account's is invalid_token.
			w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(raw))
		var message struct {
			Method string `json:"method"`
			Params struct {
				Name      string `json:"name"`
				Arguments struct {
					Scope  string `json:"scope"`
					Claims string `json:"claims"`
				} `json:"arguments"`
			} `json:"params"`
		}
		_ = json.Unmarshal(raw, &message)
		p.mu.Lock()
		p.methods[account] = append(p.methods[account], message.Method)
		if message.Method == "tools/call" {
			p.called[account]++
		}
		streams := p.streams
		p.mu.Unlock()
		if message.Method == "tools/call" && message.Params.Name == toolGuarded.Name && message.Params.Arguments.Claims != "" {
			// Microsoft's claims challenge: a 401 insufficient_claims with the base64 claims
			// request (oauth2code's Classify).
			w.Header().Set("WWW-Authenticate", `Bearer error="insufficient_claims", claims="`+message.Params.Arguments.Claims+`"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if message.Method == "tools/call" && message.Params.Name == toolGuarded.Name {
			// RFC 6750 section 3.1: insufficient_scope, with the scope the request needs.
			w.Header().Set("WWW-Authenticate", `Bearer error="insufficient_scope", scope="`+message.Params.Arguments.Scope+`"`)
			w.WriteHeader(http.StatusForbidden)
			return
		}
		if streams && message.Method == "tools/call" {
			// The SSE headers go out now; the SDK's own WriteHeader after them is ignored, and
			// its events follow on the stream once the tool answers.
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.(http.Flusher).Flush()
		}
		// Stateless, as the mcp source's own tests run the SDK (go-sdk v1.8.0); JSON unless
		// the provider streams.
		mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server },
			&mcpsdk.StreamableHTTPOptions{Stateless: true, JSONResponse: !streams}).ServeHTTP(w, r)
	}))
	s.T().Cleanup(p.Close)
	return p
}

// streamFirst has the provider answer tools/call as an SSE stream whose headers go out first.
func (p *accountsProvider) streamFirst() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.streams = true
}

// forget clears what the provider was sent, so a test reads its own.
func (p *accountsProvider) forget() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.called, p.methods, p.streams = map[string]int{}, map[string][]string{}, false
}

// calls is how many tools/call reached account.
func (p *accountsProvider) calls(account string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.called[account]
}

// sent is every JSON-RPC method account was sent.
func (p *accountsProvider) sent(account string) []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.methods[account]...)
}
