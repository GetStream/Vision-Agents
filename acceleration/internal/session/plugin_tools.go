package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// attachPlugins opens the MCP servers this config is logged into, and the ones it names by
// URL that need no login or the app's, and returns their tools. A server that will not start
// is skipped so a broken Slack login does not refuse the call. A server named by URL that
// needs the app's login and has none is returned in unconnected. A nil transport reaches
// only public hosts. A plugin the config names with user set is each end user's, so an
// app's login to it is not used.
func attachPlugins(ctx context.Context, spec Spec, db *store.Store, pluginAuth *plugins.Auth, logger *slog.Logger) (runtime *plugins.Runtime, tools []harness.Tool, unconnected []string) {
	var transport *http.Client
	if pluginAuth != nil {
		transport = pluginAuth.HTTP
	}
	var wanted []plugins.Connection
	logins := map[string]store.PluginConnection{}
	if db != nil && spec.ConfigID != "" {
		conns, err := db.ConnectedPlugins(ctx, spec.CustomerID, spec.ConfigID)
		if err != nil {
			logger.Warn("not loading plugin connections", "config", spec.ConfigID, "error", err)
		}
		for _, conn := range conns {
			if _, listed := plugins.Lookup(conn.PluginID); !listed {
				logins[conn.PluginID] = conn
				continue
			}
			entry := EntryFor(conn.PluginID, spec.Plugins)
			if entry.User {
				continue
			}
			// A connector binding for the same provider wins (Spec.Normalize).
			if spec.boundProvider(conn.PluginID) {
				continue
			}
			plugin, err := ConfiguredPlugin(entry)
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
				AccessToken: FreshToken(ctx, db, pluginAuth, &conn, logger),
				Renew:       Renewal(db, pluginAuth, &conn, logger),
				Tools:       plugin.Tools,
			})
		}
	}
	for _, server := range spec.MCPServers {
		if server.User {
			continue
		}
		if server.NeedsLogin == nil {
			needs, err := askLogin(ctx, server, transport)
			if err != nil {
				logger.Warn("mcp server could not be asked whether it needs a login", "server", server.Name, "error", err)
				continue
			}
			server.NeedsLogin = &needs
		}
		connection := plugins.Connection{PluginID: server.Name, Endpoint: server.URL, Tools: server.Tools}
		if server.AppLogin() {
			conn, ok := logins[server.Name]
			if !ok || conn.InstanceURL != server.URL {
				logger.Warn("mcp server is not connected", "server", server.Name, "config", spec.ConfigID)
				unconnected = append(unconnected, server.Name)
				continue
			}
			connection.AccessToken = FreshToken(ctx, db, pluginAuth, &conn, logger)
			connection.Renew = Renewal(db, pluginAuth, &conn, logger)
		}
		wanted = append(wanted, connection)
	}
	if len(wanted) == 0 {
		return nil, nil, unconnected
	}

	runtime, tools, failures := plugins.Open(ctx, wanted, transport)
	for _, failure := range failures {
		logger.Warn("plugin did not connect", "error", failure)
	}
	return runtime, tools, unconnected
}

// loginQuestionTimeout bounds asking a server that could not be asked when its config was
// saved whether it needs a login.
const loginQuestionTimeout = 5 * time.Second

func askLogin(ctx context.Context, server store.MCPServer, transport *http.Client) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, loginQuestionTimeout)
	defer cancel()
	return (&plugins.Auth{HTTP: transport}).NeedsLogin(ctx, server.URL)
}

// unconnectedTools stand in for the tools of servers that need the app's login and have
// none, so the model can say what to do rather than not know the server is there.
func unconnectedTools(servers []string) []harness.Tool {
	tools := make([]harness.Tool, 0, len(servers))
	for _, server := range servers {
		tools = append(tools, harness.Tool{
			Name: plugins.Prefix(server, plugins.ListToolsSuffix),
			Description: fmt.Sprintf("List what %s can do. It fails until the app connects %s "+
				"on the dashboard; say so if it does.", server, server),
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		})
	}
	return tools
}

// notConnected is what a server that needs the app's login and has none fails with.
func notConnected(server string) error {
	return fmt.Errorf("%s is not connected: connect %s on the dashboard", server, server)
}

// ConfiguredPlugin is a catalog plugin as the config's entry for it asks for it.
func ConfiguredPlugin(entry store.PluginEntry) (plugins.Plugin, error) {
	plugin, ok := plugins.Lookup(entry.Name)
	if !ok {
		return plugins.Plugin{}, stack.Wrap(fmt.Errorf("session: no plugin called %s", entry.Name))
	}
	return plugin.Configured(plugins.Options{
		Readonly: entry.Readonly, Scopes: entry.Scopes, Toolsets: entry.Toolsets, Tools: entry.Tools,
	})
}

// EntryFor is the entry naming id, or one with nothing but its name. An app's login may be
// for a plugin its config does not name yet, which is the catalog's.
func EntryFor(id string, entries []store.PluginEntry) store.PluginEntry {
	for _, entry := range entries {
		if entry.Name == id {
			return entry
		}
	}
	return store.PluginEntry{Name: id}
}

// ServerPlugin is an MCP server a config names by URL, as the plugin its login is made for.
func ServerPlugin(server store.MCPServer) plugins.Plugin {
	return plugins.Plugin{
		ID: server.Name, Name: server.Name, URL: server.URL,
		Scopes: server.Scopes, Tools: server.Tools, ByURL: true,
	}
}

// Logins are the servers each end user logs into in the conversation: the plugins and the
// servers named by URL with user set. Only they may ask for a login there; the app logs
// into the rest on the dashboard.
func Logins(spec Spec) []string {
	ids := store.PluginNames(store.UserPlugins(spec.Plugins))
	for _, server := range spec.MCPServers {
		if server.User {
			ids = append(ids, server.Name)
		}
	}
	return ids
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
func FreshToken(ctx context.Context, db *store.Store, renewer *plugins.Auth, conn *store.PluginConnection, logger *slog.Logger) string {
	if conn.ExpiresAt == nil || time.Until(*conn.ExpiresAt) >= time.Minute || conn.RefreshToken == "" {
		return conn.AccessToken
	}
	if err := renewToken(ctx, db, renewer, conn, logger); err != nil {
		logger.Warn("could not refresh a plugin token", "plugin", conn.PluginID, "error", err)
	}
	return conn.AccessToken
}

// Renewal renews a login whenever the server refuses its token, whatever its expiry says,
// as plugins.Connection.Renew. A login the provider will not renew is marked failed, so it
// is no longer offered as connected. Nil for a login with no refresh token.
func Renewal(db *store.Store, renewer *plugins.Auth, conn *store.PluginConnection, logger *slog.Logger) func(context.Context) (string, error) {
	if conn.RefreshToken == "" {
		return nil
	}
	return func(ctx context.Context) (string, error) {
		err := renewToken(ctx, db, renewer, conn, logger)
		if errors.Is(err, plugins.ErrRefreshRefused) {
			conn.Status = store.PluginFailed
			if saveErr := db.SavePluginConnection(ctx, conn); saveErr != nil {
				logger.Warn("could not mark a plugin login failed", "plugin", conn.PluginID, "error", saveErr)
			}
		}
		if err != nil {
			return "", err
		}
		logger.Info("renewed a plugin token the server refused", "plugin", conn.PluginID, "config", conn.ConfigID)
		return conn.AccessToken, nil
	}
}

// renewToken swaps conn's tokens for new ones from its refresh token and stores them.
func renewToken(ctx context.Context, db *store.Store, renewer *plugins.Auth, conn *store.PluginConnection, logger *slog.Logger) error {
	owner := plugins.Owner{CustomerID: conn.CustomerID, ConfigID: conn.ConfigID}
	refreshed, err := renewer.Refresh(ctx, owner, conn.PluginID, conn.TokenEndpoint, conn.ClientID, conn.RefreshToken)
	if err != nil {
		return err
	}
	conn.AccessToken = refreshed.AccessToken
	conn.RefreshToken = refreshed.RefreshToken
	conn.ExpiresAt = refreshed.ExpiresAt
	if err := db.SavePluginConnection(ctx, conn); err != nil {
		logger.Warn("could not store a refreshed plugin token", "plugin", conn.PluginID, "error", err)
	}
	return nil
}

// userPlugins runs the plugins the caller connects with their own account, in front of
// next. Nil when the session has none to offer: none named, no store to keep logins in, or
// no end user whose account it would be. An anonymous caller goes by a name nobody checked,
// so a login made under it would be anybody's who used the same name.
func (m *Manager) userPlugins(spec Spec, next agent.ToolRunner) *userPluginRunner {
	var offered []plugins.Plugin
	for _, entry := range store.UserPlugins(spec.Plugins) {
		if plugin, err := ConfiguredPlugin(entry); err == nil {
			offered = append(offered, plugin)
		}
	}
	for _, server := range spec.MCPServers {
		if server.User {
			offered = append(offered, ServerPlugin(server))
		}
	}
	if len(offered) == 0 || m.options.Store == nil || spec.ConfigID == "" {
		return nil
	}
	if spec.Caller.UserID == "" || spec.CallerKind == auth.KindAnonymous {
		m.logger.Warn("not offering user plugins to a session with no verified end user",
			"config", spec.ConfigID, "plugins", store.PluginNames(store.UserPlugins(spec.Plugins)))
		return nil
	}
	named := map[string]plugins.Plugin{}
	for _, plugin := range offered {
		named[plugin.ID] = plugin
	}
	signer := m.options.PluginAuth
	if signer == nil {
		signer = &plugins.Auth{}
	}
	return &userPluginRunner{
		customerID: spec.CustomerID,
		configID:   spec.ConfigID,
		userID:     spec.Caller.UserID,
		offered:    offered,
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
	// offered are the plugins the caller may connect, in the order their tools are offered.
	offered []plugins.Plugin

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
	if err != nil || conn.Status != store.PluginConnected || conn.AccessToken == "" ||
		plugin.ByURL && conn.InstanceURL != plugin.URL {
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
		AccessToken: FreshToken(ctx, r.db, r.auth, &conn, r.logger),
		Renew:       Renewal(r.db, r.auth, &conn, r.logger),
		Tools:       plugin.Tools,
	}}, r.auth.HTTP)
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
// A plugin the agent was given no client for cannot be logged into, which the model is told
// to pass on as the plugin being unavailable here rather than as a fault to fix.
func (r *userPluginRunner) authorize(ctx context.Context, plugin plugins.Plugin) (string, error) {
	owner := plugins.Owner{CustomerID: r.customerID, ConfigID: r.configID}
	pending, err := r.auth.StartAuthorize(ctx, owner, plugin, "")
	if errors.Is(err, plugins.ErrClientRequired) {
		r.logger.Warn("an end user asked for a plugin this agent has no client for", "plugin", plugin.ID, "config", r.configID)
		return plugins.UnavailableResult(plugin), nil
	}
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
	logo := r.auth.LogoURL(plugin.ID)
	if plugin.ByURL {
		conn.InstanceURL = plugin.URL
		logo = ""
	}
	if err := r.db.UpsertPluginConnection(ctx, &conn); err != nil {
		return "", err
	}
	r.logger.Info("asked an end user to connect a plugin", "plugin", plugin.ID, "config", r.configID)
	return plugins.AuthorizationResult(plugin, pending.AuthorizeURL, logo), nil
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
	mcp *plugins.Runtime
	// unconnected are the servers whose stand-in tools fail with how to connect them.
	unconnected []string
	next        agent.ToolRunner
}

func (r *pluginRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	for _, server := range r.unconnected {
		if call.Name == plugins.Prefix(server, plugins.ListToolsSuffix) {
			return nil, notConnected(server)
		}
	}
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
// user's video, and the frames as video.
type videoRunner struct {
	next    agent.ToolRunner
	session *Session
}

func (r *videoRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	parts, err := r.next.Run(ctx, call)
	if err != nil || call.Name != agent.VideoFramesTool {
		return parts, err
	}
	saw := false
	for _, part := range parts {
		if part.Image != nil {
			part.Image.Video = true
			saw = true
		}
	}
	if saw {
		r.session.SawVideo()
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
