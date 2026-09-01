package plugins

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// PrefixSeparator keeps a plugin's tools from colliding with lookup, search or transfer.
const PrefixSeparator = "__"

// Connection is what a session needs to open one MCP server.
type Connection struct {
	PluginID    string
	Endpoint    string
	AccessToken string
}

// Runtime is the MCP sessions a conversation opened, and the tools they offered.
type Runtime struct {
	clients []*client
	owned   map[string]*client
}

type client struct {
	pluginID string
	endpoint string
	token    string
	http     *http.Client
	nextID   int
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *rpcError       `json:"error"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type toolsListResult struct {
	Tools []mcpTool `json:"tools"`
}

type mcpTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type toolsCallResult struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError"`
}

type mcpContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Open connects each login and lists its tools. A server that will not start is skipped
// rather than failing the call.
func Open(ctx context.Context, conns []Connection, transport *http.Client) (*Runtime, []harness.Tool, []error) {
	if transport == nil {
		transport = http.DefaultClient
	}
	runtime := &Runtime{owned: map[string]*client{}}
	var tools []harness.Tool
	var failures []error
	for _, conn := range conns {
		opened, listed, err := dial(ctx, conn, transport)
		if err != nil {
			failures = append(failures, fmt.Errorf("%s: %w", conn.PluginID, err))
			continue
		}
		runtime.clients = append(runtime.clients, opened)
		for _, tool := range listed {
			prefixed := Prefix(conn.PluginID, tool.Name)
			runtime.owned[prefixed] = opened
			tools = append(tools, harness.Tool{
				Name:        prefixed,
				Description: fmt.Sprintf("%s (via %s)", tool.Description, conn.PluginID),
				Parameters:  tool.InputSchema,
			})
		}
	}
	if len(runtime.clients) == 0 {
		return nil, nil, failures
	}
	return runtime, tools, failures
}

func dial(ctx context.Context, conn Connection, transport *http.Client) (*client, []mcpTool, error) {
	opened := &client{
		pluginID: conn.PluginID,
		endpoint: conn.Endpoint,
		token:    conn.AccessToken,
		http:     transport,
		nextID:   1,
	}
	_, err := opened.call(ctx, "initialize", map[string]any{
		"protocolVersion": "2025-03-26",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]string{"name": "vision-agents", "version": "0"},
	})
	if err != nil {
		return nil, nil, err
	}
	if err := opened.notify(ctx, "notifications/initialized", nil); err != nil {
		return nil, nil, err
	}
	raw, err := opened.call(ctx, "tools/list", map[string]any{})
	if err != nil {
		return nil, nil, err
	}
	var listed toolsListResult
	if err := json.Unmarshal(raw, &listed); err != nil {
		return nil, nil, fmt.Errorf("plugins: tools/list: %w", err)
	}
	return opened, listed.Tools, nil
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
		return "", fmt.Errorf("plugins: no mcp runtime")
	}
	opened, ok := r.owned[call.Name]
	if !ok {
		return "", fmt.Errorf("plugins: %s is not a plugin tool", call.Name)
	}
	_, tool, ok := Split(call.Name)
	if !ok {
		return "", fmt.Errorf("plugins: %s is not a plugin tool", call.Name)
	}
	var arguments any
	if strings.TrimSpace(call.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
			return "", fmt.Errorf("plugins: arguments: %w", err)
		}
	}
	raw, err := opened.call(ctx, "tools/call", map[string]any{
		"name":      tool,
		"arguments": arguments,
	})
	if err != nil {
		return "", err
	}
	var result toolsCallResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return string(raw), nil
	}
	var text strings.Builder
	for _, part := range result.Content {
		if part.Type == "text" && part.Text != "" {
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(part.Text)
		}
	}
	if text.Len() == 0 {
		return string(raw), nil
	}
	if result.IsError {
		return "", fmt.Errorf("%s", text.String())
	}
	return text.String(), nil
}

// Close drops every MCP session. Safe on a nil runtime.
func (r *Runtime) Close() {
	if r == nil {
		return
	}
	r.clients = nil
	r.owned = nil
}

func (c *client) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := c.nextID
	c.nextID++
	body, err := json.Marshal(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params})
	if err != nil {
		return nil, err
	}
	raw, err := c.roundTrip(ctx, body)
	if err != nil {
		return nil, err
	}
	var response rpcResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, fmt.Errorf("plugins: %s: %w", method, err)
	}
	if response.Error != nil {
		return nil, fmt.Errorf("plugins: %s: %s", method, response.Error.Message)
	}
	return response.Result, nil
}

func (c *client) notify(ctx context.Context, method string, params any) error {
	body, err := json.Marshal(rpcRequest{JSONRPC: "2.0", Method: method, Params: params})
	if err != nil {
		return err
	}
	_, err = c.roundTrip(ctx, body)
	return err
}

func (c *client) roundTrip(ctx context.Context, body []byte) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	if c.token != "" {
		request.Header.Set("Authorization", "Bearer "+c.token)
	}
	response, err := c.http.Do(request)
	if err != nil {
		return nil, fmt.Errorf("plugins: %s: %w", c.pluginID, err)
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("plugins: %s: %w", c.pluginID, err)
	}
	if response.StatusCode >= 300 {
		return nil, fmt.Errorf("plugins: %s: %s", c.pluginID, strings.TrimSpace(string(raw)))
	}
	if strings.Contains(response.Header.Get("Content-Type"), "text/event-stream") {
		return sseData(raw)
	}
	return raw, nil
}

func sseData(raw []byte) ([]byte, error) {
	var last []byte
	for _, line := range bytes.Split(raw, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if bytes.HasPrefix(line, []byte("data:")) {
			last = bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		}
	}
	if len(last) == 0 {
		return nil, fmt.Errorf("plugins: empty event stream")
	}
	return last, nil
}

// Prefix is how a plugin tool is offered to the model.
func Prefix(pluginID, tool string) string {
	return pluginID + PrefixSeparator + tool
}

// Split undoes Prefix.
func Split(name string) (pluginID, tool string, ok bool) {
	pluginID, tool, ok = strings.Cut(name, PrefixSeparator)
	return pluginID, tool, ok && pluginID != "" && tool != ""
}
