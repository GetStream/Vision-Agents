package api

import (
	"context"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

const unknownPlugin = "no such plugin"

type Plugin struct {
	Id               string  `json:"id"`
	Name             string  `json:"name"`
	Category         string  `json:"category"`
	Description      string  `json:"description"`
	InstanceRequired *bool   `json:"instance_required,omitempty"`
	InstanceHint     *string `json:"instance_hint,omitempty"`
}

func (*Plugin) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One hosted MCP server from the built-in catalog."
	return schema
}

type PluginConnectionStatus string

const (
	PluginConnectionStatusConnected PluginConnectionStatus = "connected"
	PluginConnectionStatusFailed    PluginConnectionStatus = "failed"
	PluginConnectionStatusPending   PluginConnectionStatus = "pending"
)

// Valid indicates whether the value is a known member of the PluginConnectionStatus enum.
func (e PluginConnectionStatus) Valid() bool {
	switch e {
	case PluginConnectionStatusConnected:
		return true
	case PluginConnectionStatusFailed:
		return true
	case PluginConnectionStatusPending:
		return true
	default:
		return false
	}
}

type PluginConnection struct {
	PluginId         string                 `json:"plugin_id"`
	Name             string                 `json:"name"`
	Category         *string                `json:"category,omitempty"`
	Description      *string                `json:"description,omitempty"`
	InstanceRequired *bool                  `json:"instance_required,omitempty"`
	InstanceHint     *string                `json:"instance_hint,omitempty"`
	InstanceUrl      *string                `json:"instance_url,omitempty"`
	Status           PluginConnectionStatus `json:"status" enum:"pending,connected,failed"`
}

func (*PluginConnection) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A catalog plugin as this agent has it, including whether it is logged in."
	return schema
}

type AuthorizePluginRequest struct {
	InstanceUrl *string `json:"instance_url,omitempty" doc:"The shop hostname or Salesforce my-domain. Required for plugins that have no single global URL."`
}

type PluginAuthorization struct {
	AuthorizeUrl string `json:"authorize_url" doc:"The URL the browser should open to finish the login."`
}

type listPluginsRequest struct {
	Q optionalParam[string] `query:"q" doc:"Filter by name, category or description."`
}

type pluginListResponse struct {
	Body []Plugin
}

type listConfigPluginsRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type pluginConnectionListResponse struct {
	Body []PluginConnection
}

type authorizePluginRequest struct {
	ID       string `path:"id" doc:"The resource, as returned when it was created."`
	PluginID string `path:"plugin_id" doc:"A built-in catalog id such as slack or calendly."`
	Body     *AuthorizePluginRequest
}

type pluginAuthorizationResponse struct {
	Body PluginAuthorization
}

type disconnectPluginRequest struct {
	ID       string `path:"id" doc:"The resource, as returned when it was created."`
	PluginID string `path:"plugin_id" doc:"A built-in catalog id such as slack or calendly."`
}

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
		Description: "Discovers the MCP server's OAuth endpoints and returns the URL the browser " +
			"should open. Shopify and Salesforce need an instance url, because they have no " +
			"single global host.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
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
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"204": {Description: "The login is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.disconnectPlugin)
}

// listPlugins returns the built-in catalog, optionally filtered.
func (s *Server) listPlugins(ctx context.Context, request *listPluginsRequest) (*pluginListResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}

	found := plugins.Search(request.Q.Value)
	listed := make([]Plugin, 0, len(found))
	for _, plugin := range found {
		listed = append(listed, pluginOf(plugin))
	}
	return &pluginListResponse{Body: listed}, nil
}

// listConfigPlugins returns the catalog as this agent has it: connected ones carry a
// status, the rest are implied absent.
func (s *Server) listConfigPlugins(ctx context.Context, request *listConfigPluginsRequest) (*pluginConnectionListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if _, err := s.store.AgentConfig(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}

	conns, err := s.store.PluginConnections(ctx, customerID, request.ID)
	if err != nil {
		return nil, err
	}

	listed := make([]PluginConnection, 0, len(conns))
	for _, conn := range conns {
		plugin, ok := plugins.Lookup(conn.PluginID)
		if !ok {
			continue
		}
		listed = append(listed, pluginConnectionOf(plugin, conn))
	}
	return &pluginConnectionListResponse{Body: listed}, nil
}

// authorizePlugin starts a plugin login and returns the URL the browser should open.
func (s *Server) authorizePlugin(ctx context.Context, request *authorizePluginRequest) (*pluginAuthorizationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}

	plugin, ok := plugins.Lookup(string(request.PluginID))
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
	if _, err := s.store.AgentConfig(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}

	pending, err := s.auth().StartAuthorize(ctx, plugin, instance)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	conn := store.PluginConnection{
		CustomerID:    customerID,
		ConfigID:      request.ID,
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
	return &pluginAuthorizationResponse{Body: PluginAuthorization{AuthorizeUrl: pending.AuthorizeURL}}, nil
}

// disconnectPlugin drops a login and unnames the plugin on the config.
func (s *Server) disconnectPlugin(ctx context.Context, request *disconnectPluginRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if _, ok := plugins.Lookup(string(request.PluginID)); !ok {
		return nil, huma.Error400BadRequest(unknownPlugin)
	}
	if _, err := s.store.AgentConfig(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}
	if err := s.store.DeletePluginConnection(ctx, customerID, request.ID, string(request.PluginID)); err != nil {
		return nil, huma.Error404NotFound(unknownPlugin)
	}
	if err := s.store.RemoveConfigPlugin(ctx, customerID, request.ID, string(request.PluginID)); err != nil {
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
	if err := s.store.AddConfigPlugin(r.Context(), conn.CustomerID, conn.ConfigID, conn.PluginID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	http.Redirect(w, r, auth.DashboardRedirect(conn.ConfigID), http.StatusFound)
}

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
