// Package mcp is the core.ToolSource for a connector's MCP server: it lists the server's tools
// with the official Go SDK and runs the granted ones for a session. Every request leaves
// through the connection's own client (core.ResolvedBinding.HTTP), which applies its
// credential and the egress policy.
package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Kind is the name a manifest's sources[].kind gives this source, and its key in
// core.Registry.ToolSources.
const Kind = "mcp"

// Separator joins a binding's alias and a tool's name into the name the model is offered,
// so a connector's tools never collide with built-in or caller tools. The prototype's
// PrefixSeparator (internal/mcp/mcp.go:22 on codex/connector-support at cf62af0d).
const Separator = "__"

// defaultStartupTimeout bounds Discover and Open: connecting and reading every tools/list
// page. 10 s is the prototype's startupTimeout (internal/mcp/mcp.go:24 at cf62af0d), which
// subtasks T14 keeps («startupTimeout of 10 s bounds Discover»). Unverified, not measured.
const defaultStartupTimeout = 10 * time.Second

// defaultCallTimeout bounds each request the client of an opened Toolset sends, the most a
// tools/call may take before the client cuts it: longer than the longest call a binding may
// ask for, so the caller's deadline always ends a call first and the session reads a cut call
// as outcome_unknown, even from a server that sent its SSE headers first. 30 s is the
// binding's maximum timeout_ms (api.AgentConnectorBinding.TimeoutMs), and the 5 s above it is
// a margin, a choice: unverified. It is still a bound, so a reply the SDK posts with no
// deadline of its own (its answer to a server's ping, go-sdk v1.8.0 internal/jsonrpc2/conn.go)
// cannot hold Toolset.Close forever.
const defaultCallTimeout = 35 * time.Second

// maxResponseBytes caps the body of one HTTP response from the MCP server, so a server that
// answers without end cannot exhaust the router's memory. 4 MiB is the prototype's
// maxMCPResponseBytes (internal/mcp/mcp.go:25 at cf62af0d), which subtasks T14 keeps.
// Unverified, not measured.
const maxResponseBytes = 4 << 20

// schemaLocation is the URL each tool's input schema is compiled under. .invalid is reserved
// and never resolves (RFC 6761 section 6.4), and no loader fetches anything (denyLoader).
const schemaLocation = "https://vision-agents.invalid/connector-tool-schema.json"

// implementation is how the router names itself to an MCP server: the prototype's
// (internal/mcp/mcp.go:183 at cf62af0d).
var implementation = &mcp.Implementation{Name: "vision-agents", Version: "0"}

// Source is the mcp core.ToolSource. It holds no state, so one serves every connection.
type Source struct {
	startupTimeout time.Duration
	callTimeout    time.Duration
}

var _ core.ToolSource = (*Source)(nil)

// New is the mcp source.
func New() *Source {
	return &Source{startupTimeout: defaultStartupTimeout, callTimeout: defaultCallTimeout}
}

// Kind is Kind.
func (*Source) Kind() string {
	return Kind
}

// Discover lists every tool the connection's MCP server offers, reading every tools/list page,
// each with its schema digest and the scopes the manifest says it needs.
func (s *Source) Discover(ctx context.Context, b core.ResolvedBinding) ([]core.ToolSpec, error) {
	// The connection's client as it is: validate's requests keep its timeout.
	session, listed, err := s.open(ctx, b, b.HTTP)
	if err != nil {
		return nil, err
	}
	defer func() { _ = session.Close() }()
	rule := sourceRule(b.Manifest)
	specs := make([]core.ToolSpec, 0, len(listed))
	for _, tool := range listed {
		digest, err := ToolSchemaDigest(tool)
		if err != nil {
			return nil, err
		}
		schema, err := schemaObject(tool.InputSchema)
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("mcp: tool %q: %w", tool.Name, err))
		}
		specs = append(specs, core.ToolSpec{
			Name:         tool.Name,
			Description:  tool.Description,
			InputSchema:  schema,
			SchemaDigest: digest,
			NeedsScopes:  needsScopes(rule, tool.Name),
		})
	}
	return specs, nil
}

// Open lists the server's tools and keeps only the granted ones whose schema digest still
// matches the grant, each offered to the model as the binding's alias, Separator and its
// name. A tool not granted, or whose schema changed since it was granted, is not offered and
// never called; nor is one whose input schema does not compile.
//
// The alias must not hold Separator, so the name a model calls splits back into one alias
// and one tool: two bindings cannot offer the same name.
func (s *Source) Open(ctx context.Context, b core.ResolvedBinding, grants []core.ToolGrant) (core.Toolset, error) {
	alias := b.Binding.Name
	if alias == "" || strings.Contains(alias, Separator) {
		return nil, stack.Wrap(fmt.Errorf("mcp: binding name %q must be set and must not hold %q", alias, Separator))
	}
	session, listed, err := s.open(ctx, b, s.callClient(b.HTTP))
	if err != nil {
		return nil, err
	}
	granted := make(map[string]string, len(grants))
	for _, grant := range grants {
		granted[grant.Name] = grant.SchemaDigest
	}
	set := &Toolset{session: session, timeout: b.Binding.Timeout, targets: map[string]target{}}
	for _, tool := range listed {
		want, found := granted[tool.Name]
		if !found {
			continue
		}
		digest, err := ToolSchemaDigest(tool)
		if err != nil || digest != want {
			continue
		}
		schema, err := schemaObject(tool.InputSchema)
		if err != nil {
			continue
		}
		validator, err := compileSchema(schema)
		if err != nil {
			continue
		}
		name := alias + Separator + tool.Name
		set.targets[name] = target{name: tool.Name, validator: validator}
		set.tools = append(set.tools, offered(name, tool.Description, schema))
	}
	return set, nil
}

// ToolSchemaDigest is the SHA-256, in hex, of the JSON of a tool's name, description and input
// schema: everything the model is shown. encoding/json writes map keys sorted, so the same
// schema has the same digest whatever order the server sent it in. The prototype's
// (internal/mcp/mcp.go:166-178 at cf62af0d).
func ToolSchemaDigest(tool *mcp.Tool) (string, error) {
	data, err := json.Marshal(struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		InputSchema any    `json:"input_schema"`
	}{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema})
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("mcp: tool %q schema: %w", tool.Name, err))
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

// callClient is the connection's client for an opened Toolset: the same client, with a
// timeout no shorter than callTimeout, so it never cuts a tools/call before the caller's
// deadline. A nil client stays nil, for open to refuse.
func (s *Source) callClient(client *http.Client) *http.Client {
	if client == nil {
		return nil
	}
	copied := *client
	if copied.Timeout < s.callTimeout {
		copied.Timeout = s.callTimeout
	}
	return &copied
}

// open connects to the connection's MCP endpoint with client, a copy of the binding's, and
// reads every tools/list page, both within the startup timeout. The session is the caller's
// to close.
func (s *Source) open(ctx context.Context, b core.ResolvedBinding, client *http.Client) (*mcp.ClientSession, []*mcp.Tool, error) {
	if b.HTTP == nil || client == nil {
		return nil, nil, stack.Wrap(errors.New("mcp: the binding has no client: build it with core.Transports"))
	}
	rule := sourceRule(b.Manifest)
	if rule == nil {
		return nil, nil, stack.Wrap(fmt.Errorf("mcp: connector %q has no mcp source", b.Manifest.ConnectorID))
	}
	endpoint := b.Manifest.Endpoints[rule.Endpoint]
	if endpoint == "" {
		return nil, nil, stack.Wrap(fmt.Errorf("mcp: connector %q has no %s endpoint resolved", b.Manifest.ConnectorID, rule.Endpoint))
	}
	startup, cancel := context.WithTimeout(ctx, s.startupTimeout)
	defer cancel()
	type opened struct {
		session *mcp.ClientSession
		listed  []*mcp.Tool
		err     error
	}
	done := make(chan opened, 1)
	go func() {
		session, listed, err := connect(startup, b, client, endpoint)
		done <- opened{session, listed, err}
	}()
	select {
	case got := <-done:
		return got.session, got.listed, got.err
	case <-startup.Done():
		// The SDK's Connect can outlast its context: against a server that never answers it
		// returns only when the client's Timeout cuts the request (go-sdk v1.8.0), 10 s for
		// Discover and callTimeout for Open, so the caller does not wait for it
		// (TestTheStartupTimeoutBoundsDiscover). A session it opens late is closed.
		go func() {
			if got := <-done; got.session != nil {
				_ = got.session.Close()
			}
		}()
		return nil, nil, stack.Wrap(fmt.Errorf("mcp: %s did not list its tools within %s: %w",
			b.Manifest.ConnectorID, s.startupTimeout, context.Cause(startup)))
	}
}

// connect opens a session with the MCP server at endpoint through base, and reads every
// tools/list page.
func connect(ctx context.Context, b core.ResolvedBinding, base *http.Client, endpoint string) (*mcp.ClientSession, []*mcp.Tool, error) {
	// A copy whose transport caps each response and then hands the request to the
	// connection's own transport, unchanged: the credential, the 401 renewal and the egress
	// checks all still run. The redirect policy and the timeout stay base's.
	client := *base
	client.Transport = capped{base: base.Transport}
	session, err := mcp.NewClient(implementation, nil).Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint:   endpoint,
		HTTPClient: &client,
		// Only answers to the router's own requests are read: tools/list and tools/call. A
		// standalone stream would be one more open request per session for notifications
		// nothing here reads.
		DisableStandaloneSSE: true,
		// A failed request is the caller's to retry, so a refused credential is not sent
		// again by the SDK behind the transport's back.
		MaxRetries: -1,
	}, nil)
	if err != nil {
		return nil, nil, stack.Wrap(fmt.Errorf("mcp: connect to %s: %w", b.Manifest.ConnectorID, err))
	}
	listed, err := listTools(ctx, session)
	if err != nil {
		_ = session.Close()
		return nil, nil, stack.Wrap(fmt.Errorf("mcp: %s: %w", b.Manifest.ConnectorID, err))
	}
	return session, listed, nil
}

// listTools reads every tools/list page: MCP 2025-11-25 «Pagination» has clients «Treat a
// missing nextCursor as the end of results» and treat cursors as opaque. A cursor seen before
// ends the list with an error, so a server cannot keep the router paging in a loop; one that
// sends new cursors forever is stopped by the startup timeout.
func listTools(ctx context.Context, session *mcp.ClientSession) ([]*mcp.Tool, error) {
	var listed []*mcp.Tool
	seen := map[string]bool{}
	cursor := ""
	for {
		result, err := session.ListTools(ctx, &mcp.ListToolsParams{Cursor: cursor})
		if err != nil {
			return nil, fmt.Errorf("tools/list: %w", err)
		}
		listed = append(listed, result.Tools...)
		if result.NextCursor == "" {
			return listed, nil
		}
		if seen[result.NextCursor] {
			return nil, errors.New("tools/list repeated a pagination cursor")
		}
		seen[result.NextCursor] = true
		cursor = result.NextCursor
	}
}

// sourceRule is the manifest's mcp source, or nil.
func sourceRule(m core.ResolvedManifest) *core.SourceRule {
	index := slices.IndexFunc(m.Sources, func(rule core.SourceRule) bool { return rule.Kind == Kind })
	if index < 0 {
		return nil
	}
	return &m.Sources[index]
}

// needsScopes is what the manifest says the tool needs.
func needsScopes(rule *core.SourceRule, name string) []string {
	if rule == nil {
		return nil
	}
	index := slices.IndexFunc(rule.Tools, func(tool core.ToolRule) bool { return tool.Name == name })
	if index < 0 {
		return nil
	}
	return slices.Clone(rule.Tools[index].NeedsScopes)
}

// schemaObject is the input schema as a JSON object.
func schemaObject(value any) (map[string]any, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil || schema == nil {
		return nil, errors.New("the input schema is not a JSON object")
	}
	return schema, nil
}

// compileSchema compiles a tool's input schema, with every external reference refused, so a
// schema cannot make the router fetch a URL. A schema without $schema is draft 2020-12, as
// MCP 2025-11-25 «Tools» says («Defaults to 2020-12 if no $schema field is present»). Set
// here rather than left to jsonschema/v6, whose default is its newest draft and «will not
// stay the same overtime» (Compiler.DefaultDraft, v6.0.2).
func compileSchema(schema map[string]any) (*jsonschema.Schema, error) {
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(denyLoader{})
	if err := compiler.AddResource(schemaLocation, schema); err != nil {
		return nil, err
	}
	return compiler.Compile(schemaLocation)
}

// denyLoader refuses every schema a tool's schema refers to by URL.
type denyLoader struct{}

func (denyLoader) Load(string) (any, error) {
	return nil, errors.New("mcp: a tool schema may not refer to an external schema")
}

// capped hands each request to base and caps what can be read of the response at
// maxResponseBytes.
type capped struct {
	base http.RoundTripper
}

func (c capped) RoundTrip(request *http.Request) (*http.Response, error) {
	base := c.base
	if base == nil {
		base = http.DefaultTransport
	}
	response, err := base.RoundTrip(request)
	if err != nil {
		return nil, err
	}
	response.Body = http.MaxBytesReader(nil, response.Body, maxResponseBytes)
	return response, nil
}
