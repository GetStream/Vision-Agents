package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// attachPlugins opens the MCP servers this config is logged into, and the ones it names by
// URL, and returns their tools. A server that will not start is skipped so a broken Slack
// login does not refuse the call.
func attachPlugins(ctx context.Context, spec Spec, db *store.Store, logger *slog.Logger) (*plugins.Runtime, []harness.Tool) {
	var wanted []plugins.Connection
	if db != nil && spec.ConfigID != "" {
		conns, err := db.ConnectedPlugins(ctx, spec.CustomerID, spec.ConfigID)
		if err != nil {
			logger.Warn("not loading plugin connections", "config", spec.ConfigID, "error", err)
		}
		for _, conn := range conns {
			plugin, err := ConfiguredPlugin(conn.PluginID, spec.PluginOptions)
			if err != nil {
				logger.Warn("plugin is not usable", "plugin", conn.PluginID, "error", err)
				continue
			}
			endpoint, err := plugin.Endpoint(conn.InstanceURL)
			if err != nil {
				logger.Warn("plugin has no endpoint", "plugin", conn.PluginID, "error", err)
				continue
			}
			wanted = append(wanted, plugins.Connection{
				PluginID:    conn.PluginID,
				Endpoint:    endpoint,
				AccessToken: FreshToken(ctx, db, &conn, logger),
				Tools:       plugin.Tools,
			})
		}
	}
	for _, server := range spec.MCPServers {
		wanted = append(wanted, plugins.Connection{PluginID: server.Name, Endpoint: server.URL, Tools: server.Tools})
	}
	if len(wanted) == 0 {
		return nil, nil
	}

	runtime, tools, failures := plugins.Open(ctx, wanted, nil)
	for _, failure := range failures {
		logger.Warn("plugin did not connect", "error", failure)
	}
	return runtime, tools
}

// ConfiguredPlugin is a catalog plugin as the config's plugin options ask for it.
func ConfiguredPlugin(id string, options []store.PluginOptions) (plugins.Plugin, error) {
	plugin, ok := plugins.Lookup(id)
	if !ok {
		return plugins.Plugin{}, fmt.Errorf("session: no plugin called %s", id)
	}
	for _, option := range options {
		if option.Plugin == id {
			return plugin.Configured(plugins.Options{
				Readonly: option.Readonly, Scopes: option.Scopes, Toolsets: option.Toolsets,
				Tools: option.Tools,
			})
		}
	}
	return plugin, nil
}

// serverInstructionsLimit caps what one server named by URL adds to the agent's
// instructions, since nobody vetted how much it says.
const serverInstructionsLimit = 4000

// serverInstructions are what the servers a config names by URL said about using their
// tools, for the agent's instructions. Catalog plugins are left out: their tools are
// described well enough to use without.
func serverInstructions(servers []store.MCPServer, mcp *plugins.Runtime) string {
	var said []string
	for _, server := range servers {
		text := mcp.Instructions(server.Name)
		if text == "" {
			continue
		}
		if len(text) > serverInstructionsLimit {
			text = strings.ToValidUTF8(text[:serverInstructionsLimit], "")
		}
		said = append(said, fmt.Sprintf("The %s tools (%s) come with these instructions from their server:\n\n%s",
			server.Name, plugins.Prefix(server.Name, "*"), text))
	}
	return strings.Join(said, "\n\n")
}

// FreshToken is a login's access token, renewed first when it is about to expire. A
// renewal that fails keeps the old one, which the server is then the judge of.
func FreshToken(ctx context.Context, db *store.Store, conn *store.PluginConnection, logger *slog.Logger) string {
	if conn.ExpiresAt == nil || time.Until(*conn.ExpiresAt) >= time.Minute || conn.RefreshToken == "" {
		return conn.AccessToken
	}
	renewer := &plugins.Auth{}
	refreshed, err := renewer.Refresh(ctx, conn.PluginID, conn.TokenEndpoint, conn.ClientID, conn.RefreshToken)
	if err != nil {
		logger.Warn("could not refresh a plugin token", "plugin", conn.PluginID, "error", err)
		return conn.AccessToken
	}
	conn.AccessToken = refreshed.AccessToken
	conn.RefreshToken = refreshed.RefreshToken
	conn.ExpiresAt = refreshed.ExpiresAt
	if err := db.SavePluginConnection(ctx, conn); err != nil {
		logger.Warn("could not store a refreshed plugin token", "plugin", conn.PluginID, "error", err)
	}
	return conn.AccessToken
}

// userPlugins runs the plugins the caller connects with their own account, in front of
// next. Nil when the session has none to offer: none named, no store to keep logins in, or
// no end user whose account it would be. An anonymous caller goes by a name nobody checked,
// so a login made under it would be anybody's who used the same name.
func (m *Manager) userPlugins(spec Spec, next agent.ToolRunner) *userPluginRunner {
	if len(spec.UserPlugins) == 0 || m.options.Store == nil || spec.ConfigID == "" {
		return nil
	}
	if spec.Caller.UserID == "" || spec.CallerKind == auth.KindAnonymous {
		m.logger.Warn("not offering user plugins to a session with no verified end user",
			"config", spec.ConfigID, "plugins", spec.UserPlugins)
		return nil
	}
	named := map[string]plugins.Plugin{}
	for _, id := range spec.UserPlugins {
		if plugin, err := ConfiguredPlugin(id, spec.PluginOptions); err == nil {
			named[id] = plugin
		}
	}
	if len(named) == 0 {
		return nil
	}
	signer := m.options.PluginAuth
	if signer == nil {
		signer = &plugins.Auth{}
	}
	return &userPluginRunner{
		customerID: spec.CustomerID,
		configID:   spec.ConfigID,
		userID:     spec.Caller.UserID,
		named:      named,
		db:         m.options.Store,
		auth:       signer,
		logger:     m.logger,
		next:       next,
		open:       map[string]*plugins.Runtime{},
	}
}

// userPluginRunner answers a user plugin's list_tools and call_tool with the caller's own
// login, or with a request that they make one, and hands everything else to next.
type userPluginRunner struct {
	customerID, configID, userID string
	named                        map[string]plugins.Plugin
	db                           *store.Store
	auth                         *plugins.Auth
	logger                       *slog.Logger
	next                         agent.ToolRunner

	mu sync.Mutex
	// open is the MCP session per plugin, opened at the first call after the user logged
	// in, with the tools it listed.
	open  map[string]*plugins.Runtime
	tools map[string][]harness.Tool
}

type userToolCall struct {
	Tool      string         `json:"tool"`
	Arguments map[string]any `json:"arguments"`
}

func (r *userPluginRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	pluginID, verb, ok := plugins.Split(call.Name)
	plugin, named := r.named[pluginID]
	if !ok || !named || verb != plugins.ListToolsSuffix && verb != plugins.CallToolSuffix {
		return r.next.Run(ctx, call)
	}

	runtime, tools, prompt, err := r.connect(ctx, plugin)
	if errors.Is(err, plugins.ErrUnauthorized) {
		prompt, err = r.reconnect(ctx, plugin)
	}
	if err != nil {
		return nil, err
	}
	if prompt != "" {
		return llm.TextParts(prompt), nil
	}
	if verb == plugins.ListToolsSuffix {
		return llm.TextParts(plugins.ListedTools(plugin.ID, tools)), nil
	}

	var wanted userToolCall
	if err := json.Unmarshal([]byte(call.Arguments), &wanted); err != nil || wanted.Tool == "" {
		return nil, fmt.Errorf("session: %s needs the name of a tool %s lists",
			call.Name, plugins.Prefix(plugin.ID, plugins.ListToolsSuffix))
	}
	arguments, err := json.Marshal(wanted.Arguments)
	if err != nil {
		return nil, err
	}
	text, err := runtime.Call(ctx, llm.ToolCall{
		ID:        call.ID,
		Name:      plugins.Prefix(plugin.ID, wanted.Tool),
		Arguments: string(arguments),
	})
	if errors.Is(err, plugins.ErrUnauthorized) {
		text, err = r.reconnect(ctx, plugin)
	}
	return llm.TextParts(text), err
}

// reconnect forgets a login the provider refused and asks the caller to make it again.
func (r *userPluginRunner) reconnect(ctx context.Context, plugin plugins.Plugin) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if runtime, ok := r.open[plugin.ID]; ok {
		runtime.Close()
		delete(r.open, plugin.ID)
	}
	r.logger.Info("a plugin refused an end user's login", "plugin", plugin.ID, "config", r.configID)
	return r.authorize(ctx, plugin)
}

// connect is the caller's MCP session with a plugin, opened on first use. Without a login
// it starts one and returns the result that asks the caller to finish it instead.
func (r *userPluginRunner) connect(ctx context.Context, plugin plugins.Plugin) (*plugins.Runtime, []harness.Tool, string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if runtime, ok := r.open[plugin.ID]; ok {
		return runtime, r.tools[plugin.ID], "", nil
	}

	conn, err := r.db.UserPluginConnection(ctx, r.customerID, r.configID, r.userID, plugin.ID)
	if err != nil || conn.Status != store.PluginConnected || conn.AccessToken == "" {
		prompt, err := r.authorize(ctx, plugin)
		return nil, nil, prompt, err
	}
	endpoint, err := plugin.Endpoint(conn.InstanceURL)
	if err != nil {
		return nil, nil, "", err
	}
	runtime, tools, failures := plugins.Open(ctx, []plugins.Connection{{
		PluginID:    plugin.ID,
		Endpoint:    endpoint,
		AccessToken: FreshToken(ctx, r.db, &conn, r.logger),
		Tools:       plugin.Tools,
	}}, nil)
	if runtime == nil {
		return nil, nil, "", errors.Join(failures...)
	}
	if r.tools == nil {
		r.tools = map[string][]harness.Tool{}
	}
	r.open[plugin.ID] = runtime
	r.tools[plugin.ID] = tools
	return runtime, tools, "", nil
}

// authorize starts the caller's login and returns the tool result asking them to finish it.
func (r *userPluginRunner) authorize(ctx context.Context, plugin plugins.Plugin) (string, error) {
	pending, err := r.auth.StartAuthorize(ctx, plugin, "")
	if err != nil {
		return "", err
	}
	conn := store.PluginConnection{
		CustomerID:    r.customerID,
		ConfigID:      r.configID,
		PluginID:      plugin.ID,
		UserID:        r.userID,
		Status:        store.PluginPending,
		OAuthState:    pending.State,
		CodeVerifier:  pending.CodeVerifier,
		ClientID:      pending.ClientID,
		TokenEndpoint: pending.TokenEndpoint,
	}
	if err := r.db.UpsertPluginConnection(ctx, &conn); err != nil {
		return "", err
	}
	r.logger.Info("asked an end user to connect a plugin", "plugin", plugin.ID, "config", r.configID)
	return plugins.AuthorizationResult(plugin, pending.AuthorizeURL, r.auth.LogoURL(plugin.ID)), nil
}

// Close drops every MCP session the caller opened.
func (r *userPluginRunner) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, runtime := range r.open {
		runtime.Close()
	}
	r.open = map[string]*plugins.Runtime{}
}

// pluginRunner runs prefixed MCP tools itself and hands everything else to the caller bridge.
type pluginRunner struct {
	mcp  *plugins.Runtime
	next agent.ToolRunner
}

func (r *pluginRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if r.mcp != nil && r.mcp.Owns(call.Name) {
		text, err := r.mcp.Call(ctx, call)
		return llm.TextParts(text), err
	}
	if r.next != nil {
		return r.next.Run(ctx, call)
	}
	return nil, errUnknownTool(call.Name)
}

// videoRunner marks the session as a video one when the caller hands back frames of the
// user's video.
type videoRunner struct {
	next    agent.ToolRunner
	session *Session
}

func (r *videoRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	parts, err := r.next.Run(ctx, call)
	if err == nil && call.Name == agent.VideoFramesTool {
		for _, part := range parts {
			if part.Image != nil {
				r.session.SawVideo()
				break
			}
		}
	}
	return parts, err
}

func errUnknownTool(name string) error {
	return &toolError{name: name}
}

type toolError struct{ name string }

func (e *toolError) Error() string {
	return "session: " + e.name + " is not a tool this session can run"
}
