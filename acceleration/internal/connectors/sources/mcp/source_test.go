package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core/contracttest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// requestTimeout bounds every request a test's client sends, so a broken source fails the
// test instead of hanging it. A choice: loopback answers in milliseconds.
const requestTimeout = 5 * time.Second

// shortStartup is the startup timeout of the test that proves it bounds Discover: long enough
// for nothing, since the provider never answers. A choice.
const shortStartup = 200 * time.Millisecond

// errNotConnected is the memory resolver's refusal.
var errNotConnected = errors.New("not connected")

func TestMCPSourceContract(t *testing.T) {
	suite.Run(t, &contracttest.SourceContract{New: func(t *testing.T) contracttest.SourceSubject {
		p := newProvider(t)
		return contracttest.SourceSubject{
			Source:       New(),
			Binding:      p.binding(t, newResolver("token-1"), "crm"),
			Calls:        p.calls,
			ChangeSchema: p.changeSchema,
		}
	}})
}

// SourceSuite is what the mcp source does beyond the contract, against an MCP server of the
// official SDK behind a TLS server on loopback that takes a bearer token, reached through
// core.Transports with the bearer scheme and a resolver that keeps one connection in memory.
type SourceSuite struct {
	suite.Suite
	ctx      context.Context
	provider *provider
	resolver *memoryResolver
	source   *Source
}

func TestSourceSuite(t *testing.T) {
	suite.Run(t, new(SourceSuite))
}

func (s *SourceSuite) SetupTest() {
	s.ctx = context.Background()
	s.provider = newProvider(s.T())
	s.resolver = newResolver("token-1")
	s.source = New()
}

func (s *SourceSuite) TestDiscoverReadsEveryPage() {
	specs, err := s.source.Discover(s.ctx, s.binding())

	s.Require().NoError(err)
	var names []string
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	s.ElementsMatch([]string{contracttest.ToolEcho, contracttest.ToolLarge, contracttest.ToolBroken}, names)
	s.Equal(3, s.provider.pages(), "the server answers one tool per page")
}

func (s *SourceSuite) TestTheScopesAToolNeedsComeFromTheManifest() {
	specs, err := s.source.Discover(s.ctx, s.binding())

	s.Require().NoError(err)
	for _, spec := range specs {
		if spec.Name == contracttest.ToolEcho {
			s.Equal([]string{"channels:read"}, spec.NeedsScopes)
		} else {
			s.Empty(spec.NeedsScopes, spec.Name)
		}
	}
}

func (s *SourceSuite) TestEveryRequestCarriesTheConnectionsCredential() {
	set := s.open(contracttest.ToolEcho)
	_, err := set.Call(s.ctx, llm.ToolCall{Name: "crm__echo", Arguments: `{"text":"hi"}`})
	s.Require().NoError(err)

	s.NotEmpty(s.provider.tokens())
	for _, token := range s.provider.tokens() {
		s.Equal("token-1", token)
	}
}

// TestARefusedCredentialIsRenewedOnceAndDiscoverGoesOn: the provider ended token-1, and the
// resolver renews it past the refusal, so the request goes once more through
// core.Transports and the listing completes with token-2.
func (s *SourceSuite) TestARefusedCredentialIsRenewedOnceAndDiscoverGoesOn() {
	s.provider.accept("token-2")
	s.resolver.renewTo("token-2")

	specs, err := s.source.Discover(s.ctx, s.binding())

	s.Require().NoError(err)
	s.Len(specs, 3)
	s.Equal("token-1", s.provider.tokens()[0])
	s.Equal("token-2", s.provider.tokens()[len(s.provider.tokens())-1])
	s.False(s.resolver.invalidated())
}

func (s *SourceSuite) TestACredentialNothingRenewsIsInvalidatedAndDiscoverFails() {
	s.provider.accept("token-2")

	_, err := s.source.Discover(s.ctx, s.binding())

	s.Error(err)
	s.True(s.resolver.invalidated())
}

func (s *SourceSuite) TestAResolverRefusalSendsNothing() {
	s.resolver.disconnect()

	_, err := s.source.Discover(s.ctx, s.binding())

	s.ErrorIs(err, errNotConnected)
	s.Empty(s.provider.tokens())
}

func (s *SourceSuite) TestAResponseOverTheCapIsRefused() {
	s.provider.addTool(&mcp.Tool{Name: "huge", Description: strings.Repeat("x", maxResponseBytes),
		InputSchema: map[string]any{"type": "object"}})

	_, err := s.source.Discover(s.ctx, s.binding())

	s.Error(err)
}

func (s *SourceSuite) TestTheStartupTimeoutBoundsDiscover() {
	s.provider.hang()
	s.source.startupTimeout = shortStartup
	started := time.Now()

	_, err := s.source.Discover(s.ctx, s.binding())

	s.Error(err)
	s.Less(time.Since(started), requestTimeout, "Discover gave up at its own timeout, not the client's")
}

func (s *SourceSuite) TestTheStartupTimeoutIsTenSeconds() {
	s.Equal(10*time.Second, New().startupTimeout)
}

func (s *SourceSuite) TestABindingNameHoldingTheSeparatorIsRefused() {
	binding := s.binding()
	binding.Binding.Name = "crm__extra"

	_, err := s.source.Open(s.ctx, binding, nil)

	s.ErrorContains(err, `must not hold "__"`)
}

// TestToolsOfTwoBindingsNeverShareAName: one binding's tool bar__baz and another's alias
// foo__bar would both be foo__bar__baz, which is why an alias never holds the separator.
func (s *SourceSuite) TestToolsOfTwoBindingsNeverShareAName() {
	s.provider.addTool(&mcp.Tool{Name: "bar__baz", InputSchema: map[string]any{"type": "object"}})
	first := s.binding()
	first.Binding.Name = "foo"

	set, err := s.source.Open(s.ctx, first, s.grants("bar__baz"))
	s.Require().NoError(err)
	defer set.Close()

	s.Equal("foo__bar__baz", set.Tools()[0].Name)
	second := s.binding()
	second.Binding.Name = "foo__bar"
	_, err = s.source.Open(s.ctx, second, nil)
	s.Error(err)
}

func (s *SourceSuite) TestASchemaReferringToAnotherURLIsNotOfferedAndNotFetched() {
	var fetched int
	var mu sync.Mutex
	external := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		fetched++
		mu.Unlock()
		_, _ = w.Write([]byte(`{"type":"object"}`))
	}))
	defer external.Close()
	s.provider.addTool(&mcp.Tool{Name: "referring", InputSchema: map[string]any{"type": "object", "$ref": external.URL + "/schema.json"}})

	set := s.open("referring")

	s.Empty(set.Tools())
	mu.Lock()
	defer mu.Unlock()
	s.Zero(fetched)
}

func (s *SourceSuite) TestAStructuredResultWithNoTextIsReadAsJSON() {
	s.provider.addTool(&mcp.Tool{Name: "count", InputSchema: map[string]any{"type": "object"}})
	set := s.open("count")

	result, err := set.Call(s.ctx, llm.ToolCall{Name: "crm__count", Arguments: "{}"})

	s.Require().NoError(err)
	s.JSONEq(`{"count":3}`, llm.TextOf(result.Parts))
}

func (s *SourceSuite) TestAToolErrorCarriesTheToolsText() {
	s.provider.addTool(&mcp.Tool{Name: "refuses", InputSchema: map[string]any{"type": "object"}})
	set := s.open("refuses")

	_, err := set.Call(s.ctx, llm.ToolCall{Name: "crm__refuses", Arguments: "{}"})

	var reported *core.ToolError
	s.Require().ErrorAs(err, &reported)
	s.Equal("the record is locked", reported.Message)
}

// TestTheFakeProvidersServerIsDiscovered runs the source against T5's fake provider, which
// speaks MCP 2026-07-28 and 2025-11-25 and lists its two tools on two pages, with an access
// token from its client credentials grant.
func (s *SourceSuite) TestTheFakeProvidersServerIsDiscovered() {
	fake := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	form := url.Values{"grant_type": {"client_credentials"}}
	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, fake.URL+fakeprovider.PathToken, strings.NewReader(form.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.SetBasicAuth(fake.ClientID, fake.ClientSecret)
	response, err := fake.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	var token struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&token))
	s.Require().NotEmpty(token.AccessToken)
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: newResolver(token.AccessToken),
		Timeout: requestTimeout, NewClient: loopback(fake.Client())})
	s.Require().NoError(err)
	binding := core.ResolvedBinding{
		Binding:  core.Binding{Name: "fake"},
		Manifest: manifest(fake.URL + fakeprovider.PathMCP),
		HTTP:     transports.Client(core.ConnectionRef{CustomerID: "app", ConnectionID: "fake-1"}, bearer.New()),
	}

	specs, err := s.source.Discover(s.ctx, binding)

	s.Require().NoError(err)
	var names []string
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	s.Equal([]string{"echo", "fail"}, names)
}

func (s *SourceSuite) binding() core.ResolvedBinding {
	return s.provider.binding(s.T(), s.resolver, "crm")
}

func (s *SourceSuite) grants(tools ...string) []core.ToolGrant {
	specs, err := s.source.Discover(s.ctx, s.binding())
	s.Require().NoError(err)
	var grants []core.ToolGrant
	for _, spec := range specs {
		for _, tool := range tools {
			if spec.Name == tool {
				grants = append(grants, core.ToolGrant{Name: tool, SchemaDigest: spec.SchemaDigest})
			}
		}
	}
	return grants
}

func (s *SourceSuite) open(tools ...string) core.Toolset {
	set, err := s.source.Open(s.ctx, s.binding(), s.grants(tools...))
	s.Require().NoError(err)
	s.T().Cleanup(set.Close)
	return set
}

// manifest is a connector whose mcp source runs at endpoint, saying echo needs channels:read.
func manifest(endpoint string) core.ResolvedManifest {
	return core.ResolvedManifest{
		ConnectorID: "custom_crm",
		Endpoints:   map[string]string{"mcp": endpoint},
		Sources: []core.SourceRule{{Kind: Kind, Endpoint: "mcp", Tools: []core.ToolRule{
			{Name: contracttest.ToolEcho, NeedsScopes: []string{"channels:read"}},
		}}},
	}
}

// provider is an MCP server of the official SDK, one tool per tools/list page, behind a TLS
// server that takes only the bearer tokens it accepts.
type provider struct {
	*httptest.Server
	mu       sync.Mutex
	server   *mcp.Server
	called   map[string]int
	accepted map[string]bool
	seen     []string
	listed   int
	hanging  chan struct{}
}

func newProvider(t *testing.T) *provider {
	p := &provider{called: map[string]int{}, accepted: map[string]bool{"token-1": true}}
	p.server = mcp.NewServer(&mcp.Implementation{Name: "provider", Version: "1"}, &mcp.ServerOptions{PageSize: 1})
	p.server.AddTool(echoTool(map[string]any{"text": map[string]any{"type": "string"}}), p.echo)
	p.server.AddTool(&mcp.Tool{Name: contracttest.ToolLarge, Description: "Answers at length.",
		InputSchema: map[string]any{"type": "object"}}, p.large)
	p.server.AddTool(&mcp.Tool{Name: contracttest.ToolBroken, Description: "Fails with a picture.",
		InputSchema: map[string]any{"type": "object"}}, p.broken)
	// Stateless: the SDK answers MCP 2026-07-28's per-request _meta only without sessions
	// (StreamableHTTPOptions.Stateless in go-sdk v1.8.0). JSON: one object per answer.
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return p.server },
		&mcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	p.Server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		raw, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(raw))
		p.mu.Lock()
		p.seen = append(p.seen, token)
		if p.accepted[token] && bytes.Contains(raw, []byte(`"method":"tools/list"`)) {
			p.listed++
		}
		accepted, hanging := p.accepted[token], p.hanging
		p.mu.Unlock()
		if hanging != nil {
			select {
			case <-hanging:
			case <-r.Context().Done():
			}
			return
		}
		if !accepted {
			// RFC 6750 section 3.1: an expired or revoked token is invalid_token.
			w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(func() {
		p.mu.Lock()
		if p.hanging != nil {
			close(p.hanging)
		}
		p.mu.Unlock()
		p.Close()
	})
	return p
}

// binding reaches the provider under alias, through core.Transports with the bearer scheme.
func (p *provider) binding(t *testing.T, resolver core.Resolver, alias string) core.ResolvedBinding {
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: resolver, Timeout: requestTimeout, NewClient: loopback(p.Client())})
	if err != nil {
		t.Fatal(err)
	}
	return core.ResolvedBinding{
		Binding:  core.Binding{Name: alias},
		Manifest: manifest(p.URL + "/mcp"),
		HTTP:     transports.Client(core.ConnectionRef{CustomerID: "app", ConnectionID: "crm-1"}, bearer.New()),
	}
}

func (p *provider) calls(tool string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.called[tool]
}

func (p *provider) tokens() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string{}, p.seen...)
}

func (p *provider) pages() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.listed
}

func (p *provider) accept(token string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.accepted = map[string]bool{token: true}
}

func (p *provider) hang() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.hanging = make(chan struct{})
}

func (p *provider) changeSchema() {
	p.server.RemoveTools(contracttest.ToolEcho)
	p.server.AddTool(echoTool(map[string]any{"text": map[string]any{"type": "string"}, "loud": map[string]any{"type": "boolean"}}), p.echo)
}

// addTool adds a tool whose answer the test names it for: count answers structured content,
// refuses an error with text, anything else an empty text.
func (p *provider) addTool(tool *mcp.Tool) {
	p.server.AddTool(tool, func(_ context.Context, request *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		p.count(request.Params.Name)
		switch request.Params.Name {
		case "count":
			return &mcp.CallToolResult{StructuredContent: map[string]any{"count": 3}}, nil
		case "refuses":
			return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "the record is locked"}}}, nil
		}
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: ""}}}, nil
	})
}

func (p *provider) count(tool string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.called[tool]++
}

func (p *provider) echo(_ context.Context, request *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	p.count(contracttest.ToolEcho)
	var arguments struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(request.Params.Arguments, &arguments); err != nil {
		return nil, err
	}
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: arguments.Text}}}, nil
}

// large answers with two bytes per character, so the cut can land inside one.
func (p *provider) large(context.Context, *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	p.count(contracttest.ToolLarge)
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: strings.Repeat("é", core.MaxResultBytes)}}}, nil
}

// broken is MCP «Tools», Error Handling: a tool execution error is a result with isError, here
// with an image and no text.
func (p *provider) broken(context.Context, *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	p.count(contracttest.ToolBroken)
	return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.ImageContent{Data: []byte("png"), MIMEType: "image/png"}}}, nil
}

func echoTool(properties map[string]any) *mcp.Tool {
	return &mcp.Tool{Name: contracttest.ToolEcho, Description: "Returns the text it is given.",
		InputSchema: map[string]any{"type": "object", "properties": properties, "required": []string{"text"}}}
}

// loopback is egress.NewClient for a test: the same redirect policy, but the TLS server's
// transport, because egress refuses loopback, where httptest listens.
func loopback(server *http.Client) core.NewClientFunc {
	policy := egress.NewClient(0, nil).CheckRedirect
	return func(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
		return &http.Client{Transport: wrap(server.Transport), Timeout: timeout, CheckRedirect: policy}
	}
}

// memoryResolver hands out one bearer token. renewTo makes a renewal past a refused credential
// give another; without it the same comes back, as a static token does.
type memoryResolver struct {
	mu          sync.Mutex
	token       string
	revision    int
	next        string
	connected   bool
	invalidates int
}

func newResolver(token string) *memoryResolver {
	return &memoryResolver{token: token, revision: 1, connected: true}
}

func (r *memoryResolver) Resolve(_ context.Context, _ core.ConnectionRef, req core.CredentialRequest) (core.AccessCredential, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.connected {
		return core.AccessCredential{}, errNotConnected
	}
	if req.Refused != nil && req.Refused.Revision == r.revision && r.next != "" {
		r.token, r.revision, r.next = r.next, r.revision+1, ""
	}
	secret, err := json.Marshal(map[string]string{"token": r.token})
	if err != nil {
		return core.AccessCredential{}, err
	}
	credential := core.NewAccessCredential(bearer.Name, time.Time{}, secret)
	credential.Revision = r.revision
	return credential, nil
}

func (r *memoryResolver) Invalidate(context.Context, core.ConnectionRef, core.AccessCredential, core.Outcome) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.invalidates++
	r.connected = false
	return nil
}

func (r *memoryResolver) Revoke(context.Context, core.ConnectionRef, core.SignalKind, time.Time) error {
	return nil
}

func (r *memoryResolver) renewTo(token string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.next = token
}

func (r *memoryResolver) disconnect() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.connected = false
}

func (r *memoryResolver) invalidated() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.invalidates > 0
}
