package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"reflect"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

var errUnknownPlugin = APIError{Type: ErrorTypeNotFound, Code: codePluginNotFound, Message: "no such plugin"}

// listPlugins returns the built-in catalog, optionally filtered.
func (s *Server) listPlugins(ctx context.Context, request *listPluginsRequest) (*listPluginsResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}

	found := plugins.Search(value(request.Q.ptr()))
	listed := make([]Plugin, 0, len(found))
	for _, plugin := range found {
		rendered := pluginOf(plugin, s.auth().LogoURL(plugin.ID))
		if plugin.ClientRequired {
			rendered.RedirectUri = optional(s.auth().CallbackURL())
		}
		listed = append(listed, rendered)
	}
	return &listPluginsResponse{Body: listed}, nil
}

// servePluginLogo is the unauthenticated image a card draws a plugin with. It is open for
// the same reason the callback is: whoever renders the card, a chat client or a browser,
// has no credential of this API's to send with an <img>.
func (s *Server) servePluginLogo(w http.ResponseWriter, r *http.Request) {
	raw, ok := plugins.Logo(r.PathValue("plugin_id"))
	if !ok {
		writeError(w, errUnknownPlugin)
		return
	}
	w.Header().Set("Content-Type", "image/svg+xml")
	w.Header().Set("Cache-Control", "public, max-age=86400, immutable")
	_, _ = w.Write(raw)
}

// listConfigPlugins returns the catalog as this agent has it: the app's logins with their
// status, then every plugin the config names that has none yet, as not_connected, which is
// what a dashboard reminds the app to finish, then its user_plugins, which the app never
// logs into. The rest of the catalog is implied absent.
func (s *Server) listConfigPlugins(ctx context.Context, request *listConfigPluginsRequest) (*listConfigPluginsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	config, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}

	conns, err := s.store.PluginConnections(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	clients, err := s.store.PluginClients(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}

	listed := make([]PluginConnection, 0, len(conns)+len(config.AgentPlugins)+len(config.UserPlugins))
	held := map[string]bool{}
	for _, conn := range conns {
		plugin, ok := plugins.Lookup(conn.PluginID)
		if !ok || userOnly(config, plugin.ID) {
			continue
		}
		held[plugin.ID] = true
		listed = append(listed, pluginConnectionOf(plugin, conn, clients, s.auth().LogoURL(plugin.ID)))
	}
	for _, id := range store.PluginNames(config.AgentPlugins) {
		plugin, ok := plugins.Lookup(id)
		if !ok || held[id] {
			continue
		}
		held[id] = true
		listed = append(listed, pluginConnectionOf(plugin,
			store.PluginConnection{Status: string(PluginConnectionStatusNotConnected)},
			clients, s.auth().LogoURL(plugin.ID)))
	}
	for _, id := range store.PluginNames(config.UserPlugins) {
		plugin, ok := plugins.Lookup(id)
		if !ok || held[id] {
			continue
		}
		held[id] = true
		rendered := pluginConnectionOf(plugin,
			store.PluginConnection{Status: string(PluginConnectionStatusNotConnected)},
			clients, s.auth().LogoURL(plugin.ID))
		user := true
		rendered.User = &user
		listed = append(listed, rendered)
	}
	for _, server := range config.MCPServers {
		if !server.AppLogin() {
			continue
		}
		status := PluginConnectionStatusNotConnected
		for _, conn := range conns {
			if conn.PluginID == server.Name && conn.InstanceURL == server.URL {
				status = PluginConnectionStatus(conn.Status)
				break
			}
		}
		name := server.Name
		if server.Branding != nil && server.Branding.Title != "" {
			name = server.Branding.Title
		}
		listed = append(listed, PluginConnection{PluginId: server.Name, Name: name, Status: status})
	}
	return &listConfigPluginsResponse{Body: listed}, nil
}

// appPlugin is what the app logs into once for a config: a catalog plugin as the config's
// entry asks for it, or an MCP server the config names by URL that needs a login and not
// each end user's. One that could not be asked at save is tried, and fails at discovery if
// it has no login.
func appPlugin(config store.AgentConfig, id string) (plugins.Plugin, error) {
	if _, ok := plugins.Lookup(id); ok {
		if userOnly(config, id) {
			return plugins.Plugin{}, invalidRequest(id + " is connected by each end user, in the conversation")
		}
		plugin, err := session.ConfiguredPlugin(session.EntryFor(id, config.AgentPlugins, config.UserPlugins))
		if err != nil {
			return plugins.Plugin{}, invalidRequest(err.Error())
		}
		return plugin, nil
	}
	for _, server := range config.MCPServers {
		if server.Name != id || (server.NeedsLogin != nil && !*server.NeedsLogin) {
			continue
		}
		if server.User {
			return plugins.Plugin{}, invalidRequest(id + " is connected by each end user, in the conversation")
		}
		return session.ServerPlugin(server), nil
	}
	return plugins.Plugin{}, errUnknownPlugin
}

// userOnly reports whether the config names a catalog plugin under user_plugins and not
// agent_plugins, so that each end user logs into it and the app does not.
func userOnly(config store.AgentConfig, id string) bool {
	return store.NamesPlugin(config.UserPlugins, id) && !store.NamesPlugin(config.AgentPlugins, id)
}

// pluginClientWarnings names the user_plugins nobody can connect yet: the provider
// registers no client on the fly and the config has none of the app's own. An end user who
// asks for one is told it is not available.
func (s *Server) pluginClientWarnings(ctx context.Context, config store.AgentConfig) ([]string, error) {
	var warnings []string
	for _, id := range store.PluginNames(config.UserPlugins) {
		plugin, ok := plugins.Lookup(id)
		if !ok || !plugin.ClientRequired {
			continue
		}
		set, err := s.auth().HasClient(ctx, plugins.Owner{CustomerID: config.CustomerID, ConfigID: config.ID}, id)
		if err != nil {
			return nil, err
		}
		if !set {
			warnings = append(warnings, "user_plugins: "+plugin.Name+" needs an OAuth client of this app's own "+
				"before anybody can connect it: set one with PUT /v1/agents/configs/{id}/plugins/"+id+"/client")
		}
	}
	return warnings, nil
}

// setPluginClient stores the OAuth client a config logs into a plugin with, its secret
// sealed. With user set it also names the plugin under user_plugins, which has no login of
// the app's to name it at.
func (s *Server) setPluginClient(ctx context.Context, request *setPluginClientRequest) (*setPluginClientResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	if s.secrets == nil {
		return nil, notConfigured("this deployment has no key to seal a client secret with: set auth.kek")
	}
	config, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}
	plugin, ok := plugins.Lookup(request.PluginId)
	if !ok {
		return nil, errUnknownPlugin
	}
	if plugin.Auth != "oauth" {
		return nil, invalidRequest(plugin.Name + " has no OAuth login to set a client for")
	}

	owner := plugins.Owner{CustomerID: customerID, ConfigID: request.Id}
	client := store.PluginClient{
		CustomerID: customerID,
		ConfigID:   request.Id,
		PluginID:   plugin.ID,
		ClientID:   request.Body.ClientId,
	}
	if secret := value(request.Body.ClientSecret); secret != "" {
		sealed, err := session.SealPluginClientSecret(s.secrets, owner, plugin.ID, secret)
		if err != nil {
			return nil, err
		}
		client.SecretSealed = sealed
		client.SecretKEKVersion = s.secrets.CurrentVersion()
	}
	if err := s.store.SavePluginClient(ctx, &client); err != nil {
		return nil, err
	}
	if value(request.Body.User) && !store.NamesPlugin(config.UserPlugins, plugin.ID) {
		config.UserPlugins = append(config.UserPlugins, store.PluginEntry{Name: plugin.ID})
		if err := s.configs.UpdateAgentConfig(ctx, &config); err != nil {
			return nil, err
		}
		s.pluginEvents.Changed(customerID, request.Id)
	}
	return &setPluginClientResponse{Body: pluginClientOf(client)}, nil
}

// deletePluginClient drops the OAuth client a config set for a plugin.
func (s *Server) deletePluginClient(ctx context.Context, request *deletePluginClientRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	if _, err := s.configs.AgentConfig(ctx, customerID, request.Id); err != nil {
		return nil, errUnknownConfig
	}
	err := s.store.DeletePluginClient(ctx, customerID, request.Id, request.PluginId)
	if errors.Is(err, store.ErrUnknownPluginClient) {
		return nil, notFound("no client is set for this plugin")
	}
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// authorizePlugin starts a plugin login and returns the URL the browser should open.
func (s *Server) authorizePlugin(ctx context.Context, request *authorizePluginRequest) (*authorizePluginResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}

	if s.store == nil {
		return nil, errNoConfigs
	}
	config, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}
	plugin, err := appPlugin(config, string(request.PluginId))
	if err != nil {
		return nil, err
	}

	instance := ""
	if request.Body != nil {
		instance = value(request.Body.InstanceUrl)
	}
	if _, err := plugin.Endpoint(instance); err != nil {
		return nil, invalidRequest(err.Error())
	}

	pending, err := s.auth().StartAuthorize(ctx, plugins.Owner{CustomerID: customerID, ConfigID: request.Id}, plugin, instance)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	conn := store.PluginConnection{
		CustomerID:    customerID,
		ConfigID:      request.Id,
		PluginID:      plugin.ID,
		InstanceURL:   instance,
		Status:        store.PluginPending,
		OAuthState:    pending.State,
		CodeVerifier:  pending.CodeVerifier,
		ClientID:      pending.ClientID,
		TokenEndpoint: pending.TokenEndpoint,
	}
	if plugin.ByURL {
		conn.InstanceURL = plugin.URL
	}
	if err := s.store.UpsertPluginConnection(ctx, &conn); err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &authorizePluginResponse{Body: PluginAuthorization{AuthorizeUrl: pending.AuthorizeURL}}, nil
}

// disconnectPlugin drops a login and unnames the plugin on the config.
func (s *Server) disconnectPlugin(ctx context.Context, request *disconnectPluginRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	config, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}
	named := func(server store.MCPServer) bool { return server.Name == string(request.PluginId) }
	if _, ok := plugins.Lookup(string(request.PluginId)); !ok && !slices.ContainsFunc(config.MCPServers, named) {
		return nil, errUnknownPlugin
	}
	// A user plugin has no login of the app's to drop, only its name and its client.
	user := userOnly(config, string(request.PluginId))
	if err := s.store.DeletePluginConnection(ctx, customerID, request.Id, string(request.PluginId)); err != nil && !user {
		return nil, errUnknownPlugin
	}
	unnamed := func(entry store.PluginEntry) bool { return entry.Name == string(request.PluginId) }
	config.AgentPlugins = slices.DeleteFunc(config.AgentPlugins, unnamed)
	config.UserPlugins = slices.DeleteFunc(config.UserPlugins, unnamed)
	if err := s.configs.UpdateAgentConfig(ctx, &config); err != nil {
		return nil, err
	}
	err = s.store.DeletePluginClient(ctx, customerID, request.Id, string(request.PluginId))
	if err != nil && !errors.Is(err, store.ErrUnknownPluginClient) {
		return nil, err
	}
	s.pluginEvents.Changed(customerID, request.Id)
	return nil, nil
}

// receivePluginEvent is the unauthenticated callback a plugin's server delivers events to.
// The token in the path names the subscription, and the signature is checked against its
// secret, so it is the server that subscription was made with or nobody.
func (s *Server) receivePluginEvent(w http.ResponseWriter, r *http.Request) {
	if s.pluginEvents == nil {
		writeError(w, gone("this deployment subscribes to no plugin events"))
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, plugins.MaxEventBytes+1))
	if err != nil {
		writeError(w, invalidRequest(err.Error()))
		return
	}
	if len(body) > plugins.MaxEventBytes {
		writeError(w, payloadTooLarge("a delivery is at most 256 KiB"))
		return
	}
	reply := s.pluginEvents.Receive(r.Context(), r.PathValue("token"), r.Header, body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(reply.Status)
	_ = json.NewEncoder(w).Encode(reply.Body)
}

// finishPluginLogin is the unauthenticated callback the provider redirects to.
func (s *Server) finishPluginLogin(w http.ResponseWriter, r *http.Request) {
	auth := s.auth()
	query := r.URL.Query()
	if query.Get("error") != "" {
		writeError(w, invalidRequest(query.Get("error")))
		return
	}
	state := query.Get("state")
	code := query.Get("code")
	if state == "" || code == "" {
		writeError(w, invalidRequest("a code and a state are required"))
		return
	}
	if s.store == nil {
		writeError(w, errNoConfigs)
		return
	}

	conn, err := s.store.PluginConnectionByState(r.Context(), state)
	if err != nil {
		writeError(w, notFound("no such login"))
		return
	}

	token, err := auth.Exchange(r.Context(), plugins.Owner{CustomerID: conn.CustomerID, ConfigID: conn.ConfigID}, plugins.Pending{
		PluginID:      conn.PluginID,
		State:         conn.OAuthState,
		CodeVerifier:  conn.CodeVerifier,
		ClientID:      conn.ClientID,
		TokenEndpoint: conn.TokenEndpoint,
	}, code)
	if err != nil {
		conn.Status = store.PluginFailed
		conn.OAuthState = ""
		conn.CodeVerifier = ""
		_ = s.store.SavePluginConnection(r.Context(), &conn)
		writeError(w, invalidRequest(err.Error()))
		return
	}

	conn.AccessToken = token.AccessToken
	conn.RefreshToken = token.RefreshToken
	conn.ExpiresAt = token.ExpiresAt
	conn.Status = store.PluginConnected
	conn.OAuthState = ""
	conn.CodeVerifier = ""
	if err := s.store.SavePluginConnection(r.Context(), &conn); err != nil {
		writeFailure(w, r, err)
		return
	}
	// An end user connected their own account from a conversation: there is no editor to
	// go back to, and the config already names the plugin for every user.
	s.pluginEvents.Changed(conn.CustomerID, conn.ConfigID)
	plugin, listed := plugins.Lookup(conn.PluginID)
	if conn.UserID != "" {
		if s.sessions != nil {
			if conversations, err := s.sessions.Conversations(); err == nil {
				conversations.Connected(state)
			}
		}
		name := conn.PluginID
		if listed {
			name = plugin.Name
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = fmt.Fprintf(w, connectedPage, html.EscapeString(name))
		return
	}
	// A server the config names by URL is named there already, under mcp_servers.
	if listed {
		if err := s.store.AddConfigPlugin(r.Context(), conn.CustomerID, conn.ConfigID, conn.PluginID); err != nil {
			writeFailure(w, r, err)
			return
		}
	}
	http.Redirect(w, r, auth.DashboardRedirect(conn.ConfigID, conn.PluginID), http.StatusFound)
}

// connectedPage is what an end user's browser shows once their login is stored.
const connectedPage = `<!doctype html><meta charset="utf-8"><title>Connected</title>` +
	`<p>%s is connected. You can close this tab and go back to the conversation.</p>`

func (s *Server) auth() *plugins.Auth {
	if s.oauth != nil {
		return s.oauth
	}
	return &plugins.Auth{
		PublicURL:    s.publicURL,
		DashboardURL: s.dashboardURL,
	}
}

func pluginOf(plugin plugins.Plugin, logoURL string) Plugin {
	rendered := Plugin{
		Id:          plugin.ID,
		Name:        plugin.Name,
		Category:    plugin.Category,
		Description: plugin.Description,
		LogoUrl:     logoURL,
	}
	if plugin.InstanceRequired {
		required := true
		rendered.InstanceRequired = &required
	}
	rendered.InstanceHint = optional(plugin.InstanceHint)
	if plugin.ReadonlyURL != "" {
		readonly := true
		rendered.Readonly = &readonly
	}
	if len(plugin.Toolsets) > 0 {
		toolsets := plugin.Toolsets
		rendered.Toolsets = &toolsets
	}
	if len(plugin.ScopesSupported) > 0 {
		scopes := plugin.ScopesSupported
		rendered.ScopesSupported = &scopes
	}
	if plugin.ClientRequired {
		required := true
		rendered.ClientRequired = &required
	}
	rendered.SetupUrl = optional(plugin.SetupURL)
	if len(plugin.SetupSteps) > 0 {
		steps := make([]PluginSetupStep, 0, len(plugin.SetupSteps))
		for _, step := range plugin.SetupSteps {
			steps = append(steps, PluginSetupStep{Title: step.Title, Description: step.Description})
		}
		rendered.SetupSteps = &steps
	}
	return rendered
}

func pluginConnectionOf(plugin plugins.Plugin, conn store.PluginConnection, clients map[string]store.PluginClient, logoURL string) PluginConnection {
	rendered := PluginConnection{
		PluginId: plugin.ID,
		Name:     plugin.Name,
		Status:   PluginConnectionStatus(conn.Status),
		LogoUrl:  logoURL,
	}
	rendered.Category = optional(plugin.Category)
	rendered.Description = optional(plugin.Description)
	rendered.InstanceHint = optional(plugin.InstanceHint)
	rendered.InstanceUrl = optional(conn.InstanceURL)
	if plugin.InstanceRequired {
		required := true
		rendered.InstanceRequired = &required
	}
	if plugin.ClientRequired {
		required := true
		rendered.ClientRequired = &required
	}
	if client, ok := clients[plugin.ID]; ok {
		set := pluginClientOf(client)
		rendered.Client = &set
	}
	return rendered
}

func pluginClientOf(client store.PluginClient) PluginClient {
	return PluginClient{ClientId: client.ClientID, HasSecret: len(client.SecretSealed) > 0}
}

// registerPlugins declares the operations served in plugins.go.
func (s *Server) registerPlugins(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listPlugins",
		Method:      http.MethodGet,
		Path:        "/v1/agents/plugins",
		Summary:     "The hosted MCP servers an agent may attach",
		Description: "A built-in catalog, not the customer's own rows. q filters by name, category or " +
			"description. Connecting one is a login on a config, not a change to this list.",
		Responses: map[string]*huma.Response{
			"200": {Description: "Matching plugins, in catalog order"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden},
	}, s.listPlugins)
	huma.Register(api, huma.Operation{
		OperationID: "listConfigPlugins",
		Method:      http.MethodGet,
		Path:        "/v1/agents/configs/{id}/plugins",
		Summary:     "The plugin logins this agent holds",
		Description: "The app's own logins, then every plugin the config names that has none yet, as " +
			"not_connected, then every MCP server it names by URL that needs a login and has no user, " +
			"which the app logs into the same way. An end user's logins, made for user_plugins or a server " +
			"with user, are never listed.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config's connections"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listConfigPlugins)
	huma.Register(api, huma.Operation{
		OperationID: "authorizePlugin",
		Method:      http.MethodPost,
		Path:        "/v1/agents/configs/{id}/plugins/{plugin_id}/authorize",
		Summary:     "Start a plugin login",
		Description: "Discovers the MCP server's OAuth endpoints and returns the URL the browser should open. " +
			"Shopify needs an instance url, because it has no single global host.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The URL the browser should open"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.authorizePlugin)
	huma.Register(api, huma.Operation{
		OperationID: "disconnectPlugin",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/configs/{id}/plugins/{plugin_id}",
		Summary:     "Drop a plugin login",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The login is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.disconnectPlugin)
	huma.Register(api, huma.Operation{
		OperationID: "setPluginClient",
		Method:      http.MethodPut,
		Path:        "/v1/agents/configs/{id}/plugins/{plugin_id}/client",
		Summary:     "Set the OAuth client an agent logs a plugin in with",
		Description: "The OAuth app the app registered with the provider, such as a Google Cloud client, " +
			"used for this config's logins to the plugin: the app's own and every end user's. A plugin " +
			"with client_required has no other way in. The secret is sealed and never returned. " +
			"Replaces the client set before; a login made with that one keeps working until it has to " +
			"be renewed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The client as stored, without its secret"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.setPluginClient)
	huma.Register(api, huma.Operation{
		OperationID: "deletePluginClient",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/configs/{id}/plugins/{plugin_id}/client",
		Summary:     "Drop the OAuth client an agent logs a plugin in with",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The client is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deletePluginClient)
}

type setPluginClientRequest struct {
	Id       string `path:"id" doc:"The resource, as returned when it was created."`
	PluginId string `path:"plugin_id" doc:"A built-in catalog id such as google_calendar."`
	Body     SetPluginClientRequest
}

type setPluginClientResponse struct {
	Body PluginClient
}

type deletePluginClientRequest struct {
	Id       string `path:"id" doc:"The resource, as returned when it was created."`
	PluginId string `path:"plugin_id" doc:"A built-in catalog id such as google_calendar."`
}

// SetPluginClientRequest is the OAuth client an app registered with a plugin's provider.
type SetPluginClientRequest struct {
	ClientId     string  `json:"client_id" minLength:"1" maxLength:"512" doc:"The client id the provider issued."`
	ClientSecret *string `json:"client_secret,omitempty" maxLength:"512" writeOnly:"true" doc:"The client secret the provider issued. Left out for a public client."`
	User         *bool   `json:"user,omitempty" doc:"Also name the plugin under the config's user_plugins, so that each end user connects their own account in the conversation, the first time the agent needs it. Left out names nothing: the app connects the plugin once with authorize, which names it under agent_plugins."`
}

func (*SetPluginClientRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The OAuth client an app registered with a plugin's provider, with the " +
		"redirect URI <public url>/v1/agents/plugins/callback."
	return schema
}

// PluginClient is the OAuth client a config logs a plugin in with, without its secret.
type PluginClient struct {
	ClientId  string `json:"client_id" doc:"The client id the provider issued."`
	HasSecret bool   `json:"has_secret" doc:"Whether a client secret is stored. It is never returned."`
}

func (*PluginClient) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The OAuth client a config logs a plugin in with. Its secret is sealed and never returned."
	return schema
}

type listPluginsRequest struct {
	Q optionalParam[string] `query:"q" doc:"Filter by name, category or description."`
}

type listPluginsResponse struct {
	Body []Plugin `nullable:"false"`
}

type listConfigPluginsRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type listConfigPluginsResponse struct {
	Body []PluginConnection `nullable:"false"`
}

type authorizePluginRequest struct {
	Id       string `path:"id" doc:"The resource, as returned when it was created."`
	PluginId string `path:"plugin_id" doc:"A built-in catalog id such as slack or calendly, or the name of an MCP server the config names by URL that the app logs into."`
	Body     *AuthorizePluginRequest
}

type authorizePluginResponse struct {
	Body PluginAuthorization
}

type disconnectPluginRequest struct {
	Id       string `path:"id" doc:"The resource, as returned when it was created."`
	PluginId string `path:"plugin_id" doc:"A built-in catalog id such as slack or calendly, or the name of an MCP server the config names by URL that the app logs into."`
}

// Plugin One hosted MCP server from the built-in catalog.
type Plugin struct {
	Category         string    `json:"category"`
	Description      string    `json:"description"`
	Id               string    `json:"id"`
	InstanceHint     *string   `json:"instance_hint,omitempty"`
	InstanceRequired *bool     `json:"instance_required,omitempty"`
	LogoUrl          string    `json:"logo_url" readOnly:"true" doc:"Where this deployment serves the plugin's logo, as an SVG needing no credential."`
	Name             string    `json:"name"`
	Readonly         *bool     `json:"readonly,omitempty" doc:"True when the plugin has a read-only endpoint an agent may pick on its entry."`
	Toolsets         *[]string `json:"toolsets,omitempty" doc:"The groups of tools an agent may limit the plugin to on its entry. Absent when it cannot be limited."`
	ScopesSupported  *[]string `json:"scopes_supported,omitempty" doc:"The OAuth scopes an agent may ask for on its entry, as the server advertises them. Absent when the server says nothing, and any scope is then passed through."`
	ClientRequired   *bool     `json:"client_required,omitempty" doc:"True when the provider registers no client on the fly, so a config needs one of the app's own, set with setPluginClient, before anybody can connect the plugin."`
	RedirectUri      *string   `json:"redirect_uri,omitempty" readOnly:"true" doc:"The redirect URI that client has to list, which is this deployment's. Only with client_required."`
	SetupUrl         *string   `json:"setup_url,omitempty" format:"uri" doc:"Where the app creates that client with the provider. Only with client_required."`
	SetupSteps       *[]PluginSetupStep `json:"setup_steps,omitempty" doc:"What to do there, in order, before pasting the client into setPluginClient. Absent when the catalog has no instructions for the plugin."`
}

func (*Plugin) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One hosted MCP server from the built-in catalog."
	return schema
}

// PluginSetupStep is one thing to do with a plugin's provider before its client is set.
type PluginSetupStep struct {
	Title       string `json:"title" doc:"What the step does, in a few words."`
	Description string `json:"description" doc:"How to do it with the provider."`
}

func (*PluginSetupStep) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One thing to do with a plugin's provider before its OAuth client can be set."
	return schema
}

// PluginConnection A catalog plugin as this agent has it, including whether it is logged in. A plugin the config names that nobody has logged into yet is not_connected, which is what a dashboard reminds the app to finish.
type PluginConnection struct {
	Category         *string                `json:"category,omitempty"`
	Description      *string                `json:"description,omitempty"`
	InstanceHint     *string                `json:"instance_hint,omitempty"`
	InstanceRequired *bool                  `json:"instance_required,omitempty"`
	InstanceUrl      *string                `json:"instance_url,omitempty"`
	LogoUrl          string                 `json:"logo_url" readOnly:"true" doc:"Where this deployment serves the plugin's logo, as an SVG needing no credential. Empty for an MCP server named by URL."`
	Name             string                 `json:"name"`
	PluginId         string                 `json:"plugin_id"`
	Status           PluginConnectionStatus `json:"status" enum:"pending,connected,failed,not_connected" doc:"The app's login. Always not_connected for a plugin with user, which the app does not log into."`
	User             *bool                  `json:"user,omitempty" doc:"True when the config names the plugin under user_plugins only: each end user connects their own account in the conversation."`
	ClientRequired   *bool                  `json:"client_required,omitempty" doc:"True when nobody can connect the plugin until the config has a client of the app's own, set with setPluginClient."`
	Client           *PluginClient          `json:"client,omitempty" doc:"The OAuth client the config set for the plugin. Absent when it set none."`
}

func (*PluginConnection) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A catalog plugin as this agent has it, including whether it is logged in. A plugin the config names that nobody has logged into yet is not_connected, which is what a dashboard reminds the app to finish, unless it has user, when each end user connects it in the conversation."
	return schema
}

// PluginConnectionStatus is the PluginConnectionStatus schema.
type PluginConnectionStatus string

// Defines values for PluginConnectionStatus.
const (
	PluginConnectionStatusConnected    PluginConnectionStatus = "connected"
	PluginConnectionStatusFailed       PluginConnectionStatus = "failed"
	PluginConnectionStatusNotConnected PluginConnectionStatus = "not_connected"
	PluginConnectionStatusPending      PluginConnectionStatus = "pending"
)

// Valid indicates whether the value is a known member of the PluginConnectionStatus enum.
func (e PluginConnectionStatus) Valid() bool {
	switch e {
	case PluginConnectionStatusConnected:
		return true
	case PluginConnectionStatusFailed:
		return true
	case PluginConnectionStatusNotConnected:
		return true
	case PluginConnectionStatusPending:
		return true
	default:
		return false
	}
}

// AuthorizePluginRequest is the AuthorizePluginRequest schema.
type AuthorizePluginRequest struct {
	InstanceUrl *string `json:"instance_url,omitempty" doc:"The shop hostname. Required for plugins that have no single global URL."`
}

// PluginEvent is one MCP event an agent subscribes to on a plugin it names.
type PluginEvent struct {
	Plugin       string          `json:"plugin" minLength:"1" doc:"A catalog plugin the config names under agent_plugins or user_plugins."`
	Event        string          `json:"event" minLength:"1" doc:"The event's name, as the server's events/list gives it, such as comment.created."`
	Arguments    *map[string]any `json:"arguments,omitempty" doc:"The event's filters, as its inputSchema describes them."`
	Instructions *string         `json:"instructions,omitempty" doc:"What the agent does with the event when it arrives, added to its instructions for that conversation."`
}

func (*PluginEvent) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One MCP event an agent subscribes to on a plugin it names. Each event " +
		"that arrives opens a text conversation from the config, as whoever's login it came " +
		"through, with the event's data as the first thing said to it."
	return schema
}

// PluginWithOptions names one catalog plugin an agent reaches, with how it is reached.
type PluginWithOptions struct {
	Name     string    `json:"name" minLength:"1" doc:"A catalog plugin id, such as linear."`
	Readonly *bool     `json:"readonly,omitempty" doc:"Reach the plugin's read-only MCP endpoint, which offers no tool that writes and asks for read access at consent. Only a plugin whose vendor runs one may set it, such as linear."`
	Scopes   *[]string `json:"scopes,omitempty" maxItems:"32" doc:"The OAuth scopes asked for at consent, in place of the catalog's. Left out asks for the catalog's, or the read-only endpoint's when readonly is set."`
	Toolsets *[]string `json:"toolsets,omitempty" maxItems:"32" doc:"Limit the server to these groups of tools, from the plugin's toolsets in the catalog, such as calcom's bookings and availability. Left out offers every tool. Changing them needs no new login."`
	Tools    *[]string `json:"tools,omitempty" maxItems:"128" doc:"Offer the model only the server's tools matching these names or path.Match patterns, such as search_files or read_*. A tool left out is neither listed nor callable. Left out offers every tool."`
}

func (*PluginWithOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One catalog plugin an agent names, with how it is reached and what " +
		"its login asks for. A login made before a change keeps what it was granted, so " +
		"connect it again for the change to take."
	return schema
}

// PluginEntry is one catalog plugin an agent names: its id alone, or a PluginWithOptions.
// It is a type of its own so that it answers in the shape it was given.
type PluginEntry PluginWithOptions

func (e PluginEntry) MarshalJSON() ([]byte, error) {
	if e.Readonly == nil && e.Scopes == nil && e.Toolsets == nil && e.Tools == nil {
		return json.Marshal(e.Name)
	}
	return json.Marshal(PluginWithOptions(e))
}

func (e *PluginEntry) UnmarshalJSON(raw []byte) error {
	var name string
	if json.Unmarshal(raw, &name) == nil {
		*e = PluginEntry{Name: name}
		return nil
	}
	var object PluginWithOptions
	if err := json.Unmarshal(raw, &object); err != nil {
		return err
	}
	*e = PluginEntry(object)
	return nil
}

func (PluginEntry) Schema(registry huma.Registry) *huma.Schema {
	one := 1
	schema := &huma.Schema{
		Description: "One catalog plugin an agent names: its id, such as sentry, or an object " +
			"naming it with how it is reached.",
		OneOf: []*huma.Schema{
			{Type: huma.TypeString, MinLength: &one},
			registry.Schema(reflect.TypeFor[PluginWithOptions](), true, "PluginWithOptions"),
		},
	}
	schema.PrecomputeMessages()
	registry.Map()["PluginEntry"] = schema
	return &huma.Schema{Ref: "#/components/schemas/PluginEntry"}
}

// McpServer is an MCP server outside the catalog that an agent reaches by its URL.
type McpServer struct {
	Name       string             `json:"name" minLength:"1" maxLength:"32" pattern:"^[a-z][a-z0-9_-]*$" doc:"What its tools are prefixed with, as <name>__<tool>. Lowercase, without __, and not a catalog plugin's id."`
	Url        string             `json:"url" minLength:"1" maxLength:"2048" doc:"Its Streamable HTTP endpoint, over https."`
	Tools      *[]string          `json:"tools,omitempty" maxItems:"128" doc:"Offer the model only the server's tools matching these names or path.Match patterns. A tool left out is neither listed nor callable. Left out offers every tool."`
	Scopes     *[]string          `json:"scopes,omitempty" maxItems:"32" doc:"The OAuth scopes its login asks for at consent. Left out, the login asks for the scopes_supported the server advertises. Only a server that needs a login may set it. A login made before a change keeps what it was granted."`
	User       *bool              `json:"user,omitempty" doc:"Each end user logs in with their own account, in the conversation, the first time the agent needs the server, as for user_plugins, rather than the app once, from the dashboard. Only a server that needs a login may set it."`
	Branding   *McpServerBranding `json:"branding,omitempty" readOnly:"true" doc:"How the server described itself when the config was saved. Absent when it did not answer."`
	NeedsLogin *bool              `json:"needs_login,omitempty" readOnly:"true" doc:"Whether the server requires an OAuth login, as it said when the config was saved: protected-resource metadata, or a 401 to a request without a token. Without user, the app logs in once, from the dashboard. Absent when it could not be asked, which a session starting asks again."`
}

// McpServerBranding is what an MCP server said about itself at initialize.
type McpServerBranding struct {
	Title       *string `json:"title,omitempty" doc:"Its display title, or its name when it gives none."`
	Description *string `json:"description,omitempty"`
	Version     *string `json:"version,omitempty"`
	IconUrl     *string `json:"icon_url,omitempty" doc:"Its first icon served over https, as the server links it."`
	WebsiteUrl  *string `json:"website_url,omitempty"`
}

func (*McpServerBranding) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The serverInfo an MCP server answers initialize with. Every field is " +
		"optional, and a server that sends only its name and version is titled by its name."
	return schema
}

func (*McpServer) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An MCP server the plugin catalog does not have. Every session opens " +
		"it at the start and offers its tools to the model; the instructions the server gives " +
		"are added to the agent's own. It is opened with no login unless it sets scopes or " +
		"user, when it logs in with OAuth as its protected-resource metadata says, registering " +
		"a client of its own, and saving it is refused when the server advertises no such login."
	return schema
}

// PluginAuthorization is the PluginAuthorization schema.
type PluginAuthorization struct {
	AuthorizeUrl string `json:"authorize_url" doc:"The URL the browser should open to finish the login."`
}
