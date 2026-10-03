package api

import (
	"context"
	"fmt"
	"html"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

const unknownPlugin = "no such plugin"

// ListPlugins returns the built-in catalog, optionally filtered.
func (s *Server) listPlugins(ctx context.Context, request *listPluginsRequest) (*listPluginsResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}

	found := plugins.Search(value(request.Q.ptr()))
	listed := make([]Plugin, 0, len(found))
	for _, plugin := range found {
		listed = append(listed, pluginOf(plugin))
	}
	return &listPluginsResponse{Body: listed}, nil
}

// ListConfigPlugins returns the catalog as this agent has it: the app's logins with their
// status, then every plugin the config names that has none yet, as not_connected, which is
// what a dashboard reminds the app to finish. The rest of the catalog is implied absent.
func (s *Server) listConfigPlugins(ctx context.Context, request *listConfigPluginsRequest) (*listConfigPluginsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	config, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}

	conns, err := s.store.PluginConnections(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}

	listed := make([]PluginConnection, 0, len(conns)+len(config.Plugins))
	held := map[string]bool{}
	for _, conn := range conns {
		plugin, ok := plugins.Lookup(conn.PluginID)
		if !ok {
			continue
		}
		held[plugin.ID] = true
		listed = append(listed, pluginConnectionOf(plugin, conn))
	}
	for _, id := range config.Plugins {
		plugin, ok := plugins.Lookup(id)
		if !ok || held[id] {
			continue
		}
		held[id] = true
		listed = append(listed, pluginConnectionOf(plugin, store.PluginConnection{Status: string(PluginConnectionStatusNotConnected)}))
	}
	return &listConfigPluginsResponse{Body: listed}, nil
}

// AuthorizePlugin starts a plugin login and returns the URL the browser should open.
func (s *Server) authorizePlugin(ctx context.Context, request *authorizePluginRequest) (*authorizePluginResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}

	plugin, ok := plugins.Lookup(string(request.PluginId))
	if !ok {
		return nil, huma.Error400BadRequest(unknownPlugin)
	}

	instance := ""
	if request.Body != nil {
		instance = value(request.Body.InstanceUrl)
	}
	if _, err := plugin.Endpoint(instance); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if _, err := s.configs.AgentConfig(ctx, customerID, request.Id); err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}

	pending, err := s.auth().StartAuthorize(ctx, plugin, instance)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
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
	if err := s.store.UpsertPluginConnection(ctx, &conn); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &authorizePluginResponse{Body: PluginAuthorization{AuthorizeUrl: pending.AuthorizeURL}}, nil
}

// DisconnectPlugin drops a login and unnames the plugin on the config.
func (s *Server) disconnectPlugin(ctx context.Context, request *disconnectPluginRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if _, ok := plugins.Lookup(string(request.PluginId)); !ok {
		return nil, huma.Error400BadRequest(unknownPlugin)
	}
	if _, err := s.configs.AgentConfig(ctx, customerID, request.Id); err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}
	if err := s.store.DeletePluginConnection(ctx, customerID, request.Id, string(request.PluginId)); err != nil {
		return nil, huma.Error404NotFound(unknownPlugin)
	}
	if err := s.store.RemoveConfigPlugin(ctx, customerID, request.Id, string(request.PluginId)); err != nil {
		return nil, err
	}
	return nil, nil
}

// finishPluginLogin is the unauthenticated callback the provider redirects to.
func (s *Server) finishPluginLogin(w http.ResponseWriter, r *http.Request) {
	auth := s.auth()
	query := r.URL.Query()
	if query.Get("error") != "" {
		http.Error(w, query.Get("error"), http.StatusBadRequest)
		return
	}
	state := query.Get("state")
	code := query.Get("code")
	if state == "" || code == "" {
		http.Error(w, "a code and a state are required", http.StatusBadRequest)
		return
	}
	if s.store == nil {
		http.Error(w, noConfigs, http.StatusBadRequest)
		return
	}

	conn, err := s.store.PluginConnectionByState(r.Context(), state)
	if err != nil {
		http.Error(w, "no such login", http.StatusNotFound)
		return
	}

	token, err := auth.Exchange(r.Context(), plugins.Pending{
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
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	conn.AccessToken = token.AccessToken
	conn.RefreshToken = token.RefreshToken
	conn.ExpiresAt = token.ExpiresAt
	conn.Status = store.PluginConnected
	conn.OAuthState = ""
	conn.CodeVerifier = ""
	if err := s.store.SavePluginConnection(r.Context(), &conn); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// An end user connected their own account from a conversation: there is no editor to
	// go back to, and the config already names the plugin for every user.
	if conn.UserID != "" {
		plugin, _ := plugins.Lookup(conn.PluginID)
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = fmt.Fprintf(w, connectedPage, html.EscapeString(plugin.Name))
		return
	}
	if err := s.store.AddConfigPlugin(r.Context(), conn.CustomerID, conn.ConfigID, conn.PluginID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	http.Redirect(w, r, auth.DashboardRedirect(conn.ConfigID), http.StatusFound)
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

func pluginOf(plugin plugins.Plugin) Plugin {
	rendered := Plugin{
		Id:          plugin.ID,
		Name:        plugin.Name,
		Category:    plugin.Category,
		Description: plugin.Description,
	}
	if plugin.InstanceRequired {
		required := true
		rendered.InstanceRequired = &required
	}
	rendered.InstanceHint = optional(plugin.InstanceHint)
	return rendered
}

func pluginConnectionOf(plugin plugins.Plugin, conn store.PluginConnection) PluginConnection {
	rendered := PluginConnection{
		PluginId: plugin.ID,
		Name:     plugin.Name,
		Status:   PluginConnectionStatus(conn.Status),
	}
	rendered.Category = optional(plugin.Category)
	rendered.Description = optional(plugin.Description)
	rendered.InstanceHint = optional(plugin.InstanceHint)
	rendered.InstanceUrl = optional(conn.InstanceURL)
	if plugin.InstanceRequired {
		required := true
		rendered.InstanceRequired = &required
	}
	return rendered
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
			"not_connected. An end user's logins, made for user_plugins, are never listed.",
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
			"Shopify and Salesforce need an instance url, because they have no single global host.\n" +
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
	PluginId string `path:"plugin_id" doc:"A built-in catalog id such as slack or calendly."`
	Body     *AuthorizePluginRequest
}

type authorizePluginResponse struct {
	Body PluginAuthorization
}

type disconnectPluginRequest struct {
	Id       string `path:"id" doc:"The resource, as returned when it was created."`
	PluginId string `path:"plugin_id" doc:"A built-in catalog id such as slack or calendly."`
}

// Plugin One hosted MCP server from the built-in catalog.
type Plugin struct {
	Category         string  `json:"category"`
	Description      string  `json:"description"`
	Id               string  `json:"id"`
	InstanceHint     *string `json:"instance_hint,omitempty"`
	InstanceRequired *bool   `json:"instance_required,omitempty"`
	Name             string  `json:"name"`
}

func (*Plugin) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One hosted MCP server from the built-in catalog."
	return schema
}

// PluginConnection A catalog plugin as this agent has it, including whether it is logged in. A plugin the config names that nobody has logged into yet is not_connected, which is what a dashboard reminds the app to finish.
type PluginConnection struct {
	Category         *string                `json:"category,omitempty"`
	Description      *string                `json:"description,omitempty"`
	InstanceHint     *string                `json:"instance_hint,omitempty"`
	InstanceRequired *bool                  `json:"instance_required,omitempty"`
	InstanceUrl      *string                `json:"instance_url,omitempty"`
	Name             string                 `json:"name"`
	PluginId         string                 `json:"plugin_id"`
	Status           PluginConnectionStatus `json:"status" enum:"pending,connected,failed,not_connected"`
}

func (*PluginConnection) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A catalog plugin as this agent has it, including whether it is logged in. A plugin the config names that nobody has logged into yet is not_connected, which is what a dashboard reminds the app to finish."
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
	InstanceUrl *string `json:"instance_url,omitempty" doc:"The shop hostname or Salesforce my-domain. Required for plugins that have no single global URL."`
}

// PluginAuthorization is the PluginAuthorization schema.
type PluginAuthorization struct {
	AuthorizeUrl string `json:"authorize_url" doc:"The URL the browser should open to finish the login."`
}
