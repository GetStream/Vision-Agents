package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// PrefixSeparator keeps a connector's tools from colliding with built-in or caller tools.
const PrefixSeparator = "__"

const startupTimeout = 10 * time.Second
const maxMCPResponseBytes = 4 << 20
const maxMCPToolResultBytes = 32 << 10
const truncatedToolResultNotice = "\n[connector result truncated]"

// Connection is what a session needs to open one MCP server.
type Connection struct {
	ConnectorID            string
	ToolPrefix          string
	Endpoint            string
	AccessToken         string
	AllowedTools        []string
	ExpectedToolDigests map[string]string
	ResolveToken        func(context.Context) (string, error)
	Authorize           func(context.Context, *http.Request) error
	AuthorizeTool       func(context.Context, string, string) error
	Timeout             time.Duration
}

// Runtime owns the MCP sessions opened for one agent session.
type Runtime struct {
	clients []*mcp.ClientSession
	owned   map[string]toolTarget
}

type toolTarget struct {
	session       *mcp.ClientSession
	name          string
	schema        *jsonschema.Schema
	digest        string
	authorizeTool func(context.Context, string, string) error
	timeout       time.Duration
}

// Open connects each authorized account, discovers all pages of its tools, and exposes
// only names present in AllowedTools when the connection has an explicit allowlist.
func Open(ctx context.Context, conns []Connection, transport *http.Client) (*Runtime, []harness.Tool, []error) {
	if transport == nil {
		transport = egress.NewPublicHTTPClient(0)
	}
	runtime := &Runtime{owned: map[string]toolTarget{}}
	var tools []harness.Tool
	var failures []error
	toolOwners := make(map[string]string)
	blockedToolNames := make(map[string]struct{})
	for _, conn := range conns {
		startupCtx, cancel := context.WithTimeout(ctx, startupTimeout)
		opened, err := dial(startupCtx, conn, transport)
		if err != nil {
			cancel()
			failures = append(failures, fmt.Errorf("%s: %w", conn.ConnectorID, err))
			continue
		}
		listed, err := listTools(startupCtx, opened)
		cancel()
		if err != nil {
			_ = opened.Close()
			failures = append(failures, fmt.Errorf("%s: %w", conn.ConnectorID, err))
			continue
		}
		runtime.clients = append(runtime.clients, opened)
		var allowed map[string]struct{}
		if conn.AllowedTools != nil {
			allowed = make(map[string]struct{}, len(conn.AllowedTools))
			for _, name := range conn.AllowedTools {
				allowed[name] = struct{}{}
			}
		}
		foundDigests := make(map[string]struct{}, len(conn.ExpectedToolDigests))
		for _, tool := range listed {
			if allowed != nil {
				if _, ok := allowed[tool.Name]; !ok {
					continue
				}
			}
			digest, err := ToolSchemaDigest(tool)
			if err != nil {
				failures = append(failures, fmt.Errorf("%s: tool %s schema could not be fingerprinted", conn.ConnectorID, tool.Name))
				continue
			}
			if expected, hasExpectation := conn.ExpectedToolDigests[tool.Name]; hasExpectation {
				foundDigests[tool.Name] = struct{}{}
				if digest != expected {
					failures = append(failures, fmt.Errorf("%s: tool %s schema changed", conn.ConnectorID, tool.Name))
					continue
				}
			}
			parameters, err := schemaObject(tool.InputSchema)
			if err != nil {
				failures = append(failures, fmt.Errorf("%s.%s: invalid input schema: %w", conn.ConnectorID, tool.Name, err))
				continue
			}
			validator, err := compileToolSchema(parameters)
			if err != nil {
				failures = append(failures, fmt.Errorf("%s.%s: invalid input schema: %w", conn.ConnectorID, tool.Name, err))
				continue
			}
			prefix := conn.ToolPrefix
			if prefix == "" {
				prefix = conn.ConnectorID
			}
			prefixed := Prefix(prefix, tool.Name)
			if _, blocked := blockedToolNames[prefixed]; blocked {
				failures = append(failures, fmt.Errorf("%s: exposed tool name %q collides with another connector", conn.ConnectorID, prefixed))
				continue
			}
			if previous, collision := toolOwners[prefixed]; collision {
				delete(runtime.owned, prefixed)
				filtered := tools[:0]
				for _, exposed := range tools {
					if exposed.Name != prefixed {
						filtered = append(filtered, exposed)
					}
				}
				tools = filtered
				delete(toolOwners, prefixed)
				blockedToolNames[prefixed] = struct{}{}
				failures = append(failures,
					fmt.Errorf("%s: exposed tool name %q collides with another connector", previous, prefixed),
					fmt.Errorf("%s: exposed tool name %q collides with another connector", conn.ConnectorID, prefixed),
				)
				continue
			}
			toolOwners[prefixed] = conn.ConnectorID
			runtime.owned[prefixed] = toolTarget{
				session: opened, name: tool.Name, schema: validator, digest: digest,
				authorizeTool: conn.AuthorizeTool, timeout: conn.Timeout,
			}
			tools = append(tools, harness.Tool{
				Name:         prefixed,
				Description:  fmt.Sprintf("%s (via %s)", tool.Description, conn.ConnectorID),
				Parameters:   parameters,
				SchemaDigest: digest,
			})
		}
		for name := range conn.ExpectedToolDigests {
			if _, found := foundDigests[name]; !found {
				failures = append(failures, fmt.Errorf("%s: granted tool %s is unavailable", conn.ConnectorID, name))
			}
		}
	}
	if len(runtime.clients) == 0 {
		return nil, nil, failures
	}
	return runtime, tools, failures
}

// ToolSchemaDigest fingerprints an MCP tool's exact model-visible contract.
func ToolSchemaDigest(tool *mcp.Tool) (string, error) {
	data, err := json.Marshal(struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		InputSchema any    `json:"input_schema"`
	}{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema})
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func dial(ctx context.Context, conn Connection, base *http.Client) (*mcp.ClientSession, error) {
	clientHTTP := *base
	clientHTTP.Transport = authorizedTransport{
		base:      base.Transport,
		token:     conn.AccessToken,
		resolve:   conn.ResolveToken,
		authorize: conn.Authorize,
	}
	clientHTTP.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "vision-agents", Version: "0"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint:             conn.Endpoint,
		HTTPClient:           &clientHTTP,
		DisableStandaloneSSE: true,
		MaxRetries:           -1,
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp: %s: connect: %w", conn.ConnectorID, err)
	}
	return session, nil
}

func listTools(ctx context.Context, session *mcp.ClientSession) ([]*mcp.Tool, error) {
	var listed []*mcp.Tool
	seen := map[string]struct{}{}
	cursor := ""
	for {
		result, err := session.ListTools(ctx, &mcp.ListToolsParams{Cursor: cursor})
		if err != nil {
			return nil, fmt.Errorf("mcp: tools/list: %w", err)
		}
		listed = append(listed, result.Tools...)
		if result.NextCursor == "" {
			return listed, nil
		}
		if _, duplicate := seen[result.NextCursor]; duplicate {
			return nil, fmt.Errorf("mcp: tools/list repeated a pagination cursor")
		}
		seen[result.NextCursor] = struct{}{}
		cursor = result.NextCursor
	}
}

func schemaObject(value any) (map[string]any, error) {
	var schema map[string]any
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil, err
	}
	if schema == nil {
		return nil, fmt.Errorf("schema is not a JSON object")
	}
	return schema, nil
}

func compileToolSchema(schema map[string]any) (*jsonschema.Schema, error) {
	const schemaLocation = "https://vision-agents.invalid/connector-tool-schema.json"
	compiler := jsonschema.NewCompiler()
	compiler.UseLoader(denyExternalSchemaLoader{})
	if err := compiler.AddResource(schemaLocation, schema); err != nil {
		return nil, err
	}
	return compiler.Compile(schemaLocation)
}

type denyExternalSchemaLoader struct{}

func (denyExternalSchemaLoader) Load(location string) (any, error) {
	return nil, fmt.Errorf("external JSON Schema references are not permitted")
}

// Owns reports whether this runtime runs the named tool.
func (r *Runtime) Owns(name string) bool {
	if r == nil {
		return false
	}
	_, ok := r.owned[name]
	return ok
}

// Call runs a prefixed tool against the MCP server that offered it.
func (r *Runtime) Call(ctx context.Context, call llm.ToolCall) (string, error) {
	if r == nil {
		return "", fmt.Errorf("mcp: no mcp runtime")
	}
	target, ok := r.owned[call.Name]
	if !ok {
		return "", fmt.Errorf("mcp: %s is not a connector tool", call.Name)
	}
	arguments := map[string]any{}
	if strings.TrimSpace(call.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
			return "", fmt.Errorf("mcp: arguments: %w", err)
		}
	}
	if err := target.schema.Validate(arguments); err != nil {
		return "", fmt.Errorf("mcp: arguments do not match the accepted tool schema")
	}
	callCtx := ctx
	if target.timeout > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, target.timeout)
		defer cancel()
	}
	if target.authorizeTool != nil {
		if err := target.authorizeTool(callCtx, target.name, target.digest); err != nil {
			return "", err
		}
	}
	result, err := target.session.CallTool(callCtx, &mcp.CallToolParams{Name: target.name, Arguments: arguments})
	if err != nil {
		return "", fmt.Errorf("mcp: tool call: %w", err)
	}
	var text strings.Builder
	truncated := false
	for _, part := range result.Content {
		if value, ok := part.(*mcp.TextContent); ok && value.Text != "" {
			appendToolResult(&text, value.Text, &truncated)
		}
	}
	if text.Len() == 0 && result.StructuredContent != nil {
		raw, err := json.Marshal(result.StructuredContent)
		if err != nil {
			return "", fmt.Errorf("mcp: encode tool result: %w", err)
		}
		appendToolResult(&text, string(raw), &truncated)
	}
	if text.Len() == 0 && len(result.Content) > 0 {
		text.WriteString("MCP tool returned non-text content that the agent cannot display")
	}
	if truncated {
		text.WriteString(truncatedToolResultNotice)
	}
	if result.IsError {
		if text.Len() == 0 {
			return "", fmt.Errorf("mcp: MCP tool returned an error without text details")
		}
		return "", fmt.Errorf("%s", text.String())
	}
	return text.String(), nil
}

func appendToolResult(output *strings.Builder, value string, truncated *bool) {
	if value == "" {
		return
	}
	if output.Len() > 0 {
		appendToolResultPart(output, "\n", truncated)
	}
	appendToolResultPart(output, value, truncated)
}

func appendToolResultPart(output *strings.Builder, value string, truncated *bool) {
	limit := maxMCPToolResultBytes - len(truncatedToolResultNotice)
	remaining := limit - output.Len()
	if remaining <= 0 {
		*truncated = true
		return
	}
	if len(value) > remaining {
		value = value[:remaining]
		for !utf8.ValidString(value) {
			value = value[:len(value)-1]
		}
		*truncated = true
	}
	output.WriteString(value)
}

// Close drops every MCP session. Safe on a nil runtime.
func (r *Runtime) Close() {
	if r == nil {
		return
	}
	for _, client := range r.clients {
		_ = client.Close()
	}
	r.clients = nil
	r.owned = nil
}

type authorizedTransport struct {
	base      http.RoundTripper
	token     string
	resolve   func(context.Context) (string, error)
	authorize func(context.Context, *http.Request) error
}

func (t authorizedTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	clone := request.Clone(request.Context())
	clone.Header = request.Header.Clone()
	if t.authorize != nil {
		if err := t.authorize(clone.Context(), clone); err != nil {
			return nil, err
		}
	} else {
		token := t.token
		if t.resolve != nil {
			var err error
			token, err = t.resolve(clone.Context())
			if err != nil {
				return nil, err
			}
		}
		if token != "" {
			clone.Header.Set("Authorization", "Bearer "+token)
		}
	}
	response, err := base.RoundTrip(clone)
	if err != nil {
		return nil, err
	}
	if response.Body != nil {
		response.Body = http.MaxBytesReader(nil, response.Body, maxMCPResponseBytes)
	}
	return response, nil
}

// Prefix is how a connector tool is offered to the model.
func Prefix(connectorID, tool string) string {
	return connectorID + PrefixSeparator + tool
}

// Split undoes Prefix.
func Split(name string) (connectorID, tool string, ok bool) {
	connectorID, tool, ok = strings.Cut(name, PrefixSeparator)
	return connectorID, tool, ok && connectorID != "" && tool != ""
}
