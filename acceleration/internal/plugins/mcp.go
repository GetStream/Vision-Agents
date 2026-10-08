package plugins

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// PrefixSeparator keeps a plugin's tools from colliding with lookup, search or transfer.
const PrefixSeparator = "__"

const sessionHeader = "Mcp-Session-Id"

// publicClient is how plugin traffic leaves the router when a caller gives no client. A
// config names its MCP servers, a login its shop and the servers' metadata their auth
// servers, so it only reaches public hosts.
var publicClient = egress.NewClient(0, nil)

// ErrUnauthorized is a server refusing the token a connection was opened with: it expired
// early, was revoked, or the account was disconnected at the provider. The login has to be
// made again.
var ErrUnauthorized = errors.New("plugins: the server refused the login")

// Connection is what a session needs to open one MCP server.
type Connection struct {
	PluginID    string
	Endpoint    string
	AccessToken string
	// Renew mints a new access token when the server refuses the one held, and the request
	// is sent once more with it. Nil leaves the refusal as ErrUnauthorized.
	Renew func(ctx context.Context) (string, error)
	// Tools offer only the server's tools matching these names or path.Match patterns.
	// Empty offers every tool.
	Tools []string
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
	renew    func(ctx context.Context) (string, error)
	http     *http.Client
	nextID   int
	// session is the Mcp-Session-Id the server gave at initialize, sent back on every
	// request after it (MCP 2025-03-26, Streamable HTTP, «Session Management»).
	session string
	// version is sent as MCP-Protocol-Version by a client that skips initialize, as an
	// MCP 2.0 one does.
	version string
	// instructions are what the server said at initialize about using its tools.
	instructions string
}

type initializeResult struct {
	Instructions string     `json:"instructions"`
	ServerInfo   serverInfo `json:"serverInfo"`
}

// serverInfo is the server's Implementation (MCP 2025-11-25, «Lifecycle»). Everything but
// name and version is optional, and older servers send only those.
type serverInfo struct {
	Name        string    `json:"name"`
	Title       string    `json:"title"`
	Version     string    `json:"version"`
	Description string    `json:"description"`
	WebsiteURL  string    `json:"websiteUrl"`
	Icons       []mcpIcon `json:"icons"`
}

type mcpIcon struct {
	Src string `json:"src"`
}

// Branding is how an MCP server describes itself at initialize, for showing it to people.
// Any of it may be empty.
type Branding struct {
	// Title is the server's display title, or its name when it gives none.
	Title       string
	Description string
	Version     string
	// IconURL is the first of its icons served over https; a data: icon is not kept.
	IconURL    string
	WebsiteURL string
}

var initializeParams = map[string]any{
	"protocolVersion": "2025-03-26",
	"capabilities":    map[string]any{},
	"clientInfo":      map[string]string{"name": "vision-agents", "version": "0"},
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
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
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
		transport = publicClient
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
			if !Offered(conn.Tools, tool.Name) {
				continue
			}
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

// Offered reports whether a server's tool is in an allowlist of names and path.Match
// patterns. An empty allowlist offers every tool.
func Offered(allowed []string, tool string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, pattern := range allowed {
		if matched, _ := path.Match(pattern, tool); matched {
			return true
		}
	}
	return false
}

// CheckToolPatterns refuses an allowlist entry path.Match cannot read, which would
// otherwise match nothing and hide a tool without saying why.
func CheckToolPatterns(allowed []string) error {
	for _, pattern := range allowed {
		if _, err := path.Match(pattern, ""); err != nil {
			return stack.Wrap(fmt.Errorf("plugins: %q is not a tool name or pattern", pattern))
		}
	}
	return nil
}

func dial(ctx context.Context, conn Connection, transport *http.Client) (*client, []mcpTool, error) {
	opened := &client{
		pluginID: conn.PluginID,
		endpoint: conn.Endpoint,
		token:    conn.AccessToken,
		renew:    conn.Renew,
		http:     transport,
		nextID:   1,
	}
	raw, err := opened.call(ctx, "initialize", initializeParams)
	if err != nil {
		return nil, nil, err
	}
	var initialized initializeResult
	if json.Unmarshal(raw, &initialized) == nil {
		opened.instructions = strings.TrimSpace(initialized.Instructions)
	}
	if err := opened.notify(ctx, "notifications/initialized", nil); err != nil {
		return nil, nil, err
	}
	raw, err = opened.call(ctx, "tools/list", map[string]any{})
	if err != nil {
		return nil, nil, err
	}
	var listed toolsListResult
	if err := json.Unmarshal(raw, &listed); err != nil {
		return nil, nil, stack.Wrap(fmt.Errorf("plugins: tools/list: %w", err))
	}
	return opened, listed.Tools, nil
}

// Describe asks a server how it describes itself, with an initialize and nothing after it.
func Describe(ctx context.Context, conn Connection, transport *http.Client) (Branding, error) {
	if transport == nil {
		transport = publicClient
	}
	opened := &client{pluginID: conn.PluginID, endpoint: conn.Endpoint, token: conn.AccessToken, http: transport, nextID: 1}
	raw, err := opened.call(ctx, "initialize", initializeParams)
	if err != nil {
		return Branding{}, err
	}
	var initialized initializeResult
	if err := json.Unmarshal(raw, &initialized); err != nil {
		return Branding{}, fmt.Errorf("plugins: initialize: %w", err)
	}
	info := initialized.ServerInfo
	branding := Branding{
		Title:       clip(info.Title, 128),
		Description: clip(info.Description, 1024),
		Version:     clip(info.Version, 64),
		WebsiteURL:  httpsURL(info.WebsiteURL),
	}
	if branding.Title == "" {
		branding.Title = clip(info.Name, 128)
	}
	for _, icon := range info.Icons {
		if branding.IconURL = httpsURL(icon.Src); branding.IconURL != "" {
			break
		}
	}
	return branding, nil
}

// clip trims text a server sent and cuts it to at most limit runes.
func clip(text string, limit int) string {
	runes := []rune(strings.TrimSpace(text))
	if len(runes) > limit {
		runes = runes[:limit]
	}
	return string(runes)
}

// httpsURL is raw when it is an https URL short enough to keep, and "" otherwise.
func httpsURL(raw string) string {
	raw = strings.TrimSpace(raw)
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil || len(raw) > 2048 {
		return ""
	}
	return raw
}

// Instructions are what the server opened for pluginID said at initialize about using its
// tools, or "" when it said nothing or is not open.
func (r *Runtime) Instructions(pluginID string) string {
	if r == nil {
		return ""
	}
	for _, opened := range r.clients {
		if opened.pluginID == pluginID {
			return opened.instructions
		}
	}
	return ""
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
		return nil, stack.Wrap(err)
	}
	raw, err := c.roundTrip(ctx, body)
	if err != nil {
		return nil, err
	}
	var response rpcResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, stack.Wrap(fmt.Errorf("plugins: %s: %w", method, err))
	}
	if response.Error != nil {
		if len(response.Error.Data) > 0 {
			return nil, stack.Wrap(fmt.Errorf("plugins: %s: %s %s", method, response.Error.Message, response.Error.Data))
		}
		return nil, stack.Wrap(fmt.Errorf("plugins: %s: %s", method, response.Error.Message))
	}
	return response.Result, nil
}

func (c *client) notify(ctx context.Context, method string, params any) error {
	body, err := json.Marshal(rpcRequest{JSONRPC: "2.0", Method: method, Params: params})
	if err != nil {
		return stack.Wrap(err)
	}
	_, err = c.roundTrip(ctx, body)
	return err
}

// roundTrip sends body, and once more with a renewed token when the server refuses the one
// held. A renewal that fails is joined to the refusal, which still reads as ErrUnauthorized.
func (c *client) roundTrip(ctx context.Context, body []byte) ([]byte, error) {
	raw, err := c.send(ctx, body)
	if c.renew == nil || !errors.Is(err, ErrUnauthorized) {
		return raw, err
	}
	token, renewErr := c.renew(ctx)
	if renewErr != nil {
		return nil, errors.Join(err, renewErr)
	}
	c.token = token
	return c.send(ctx, body)
}

func (c *client) send(ctx context.Context, body []byte) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, stack.Wrap(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	if c.token != "" {
		request.Header.Set("Authorization", "Bearer "+c.token)
	}
	if c.session != "" {
		request.Header.Set(sessionHeader, c.session)
	}
	if c.version != "" {
		request.Header.Set("MCP-Protocol-Version", c.version)
	}
	response, err := c.http.Do(request)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("plugins: %s: %w", c.pluginID, err))
	}
	defer response.Body.Close()
	if session := response.Header.Get(sessionHeader); session != "" {
		c.session = session
	}
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("plugins: %s: %w", c.pluginID, err))
	}
	if response.StatusCode == http.StatusUnauthorized {
		return nil, stack.Wrap(fmt.Errorf("%w: %s: %s", ErrUnauthorized, c.pluginID, strings.TrimSpace(string(raw))))
	}
	if response.StatusCode >= 300 {
		return nil, stack.Wrap(fmt.Errorf("plugins: %s: %s", c.pluginID, strings.TrimSpace(string(raw))))
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
		return nil, stack.Wrap(fmt.Errorf("plugins: empty event stream"))
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
