package api

import (
	"context"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const unknownPlugin = "no such plugin"

// ListPlugins returns the built-in catalog, optionally filtered.
func (s *Server) ListPlugins(ctx context.Context, request ListPluginsRequestObject) (ListPluginsResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return ListPlugins401JSONResponse{missingCustomer()}, nil
	}

	found := plugins.Search(value(request.Params.Q))
	listed := make([]Plugin, 0, len(found))
	for _, plugin := range found {
		listed = append(listed, pluginOf(plugin))
	}
	return ListPlugins200JSONResponse(listed), nil
}

// ListConfigPlugins returns the catalog as this agent has it: connected ones carry a
// status, the rest are implied absent.
func (s *Server) ListConfigPlugins(ctx context.Context, request ListConfigPluginsRequestObject) (ListConfigPluginsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListConfigPlugins401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListConfigPlugins400JSONResponse{badRequest(noConfigs)}, nil
	}
	if _, err := s.store.AgentConfig(ctx, customerID, request.Id); err != nil {
		return ListConfigPlugins404JSONResponse{NotFoundJSONResponse{Error: unknownConfig}}, nil
	}

	conns, err := s.store.PluginConnections(ctx, customerID, request.Id)
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
	return ListConfigPlugins200JSONResponse(listed), nil
}

// AuthorizePlugin starts a plugin login and returns the URL the browser should open.
func (s *Server) AuthorizePlugin(ctx context.Context, request AuthorizePluginRequestObject) (AuthorizePluginResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return AuthorizePlugin401JSONResponse{missingCustomer()}, nil
	}

	plugin, ok := plugins.Lookup(string(request.PluginId))
	if !ok {
		return AuthorizePlugin400JSONResponse{badRequest(unknownPlugin)}, nil
	}

	instance := ""
	if request.Body != nil {
		instance = value(request.Body.InstanceUrl)
	}
	if _, err := plugin.Endpoint(instance); err != nil {
		return AuthorizePlugin400JSONResponse{badRequest(err.Error())}, nil
	}
	if s.store == nil {
		return AuthorizePlugin400JSONResponse{badRequest(noConfigs)}, nil
	}
	if _, err := s.store.AgentConfig(ctx, customerID, request.Id); err != nil {
		return AuthorizePlugin404JSONResponse{NotFoundJSONResponse{Error: unknownConfig}}, nil
	}

	pending, err := s.auth().StartAuthorize(ctx, plugin, instance)
	if err != nil {
		return AuthorizePlugin400JSONResponse{badRequest(err.Error())}, nil
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
		return AuthorizePlugin400JSONResponse{badRequest(err.Error())}, nil
	}
	return AuthorizePlugin200JSONResponse{AuthorizeUrl: pending.AuthorizeURL}, nil
}

// DisconnectPlugin drops a login and unnames the plugin on the config.
func (s *Server) DisconnectPlugin(ctx context.Context, request DisconnectPluginRequestObject) (DisconnectPluginResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DisconnectPlugin401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return DisconnectPlugin400JSONResponse{badRequest(noConfigs)}, nil
	}
	if _, ok := plugins.Lookup(string(request.PluginId)); !ok {
		return DisconnectPlugin400JSONResponse{badRequest(unknownPlugin)}, nil
	}
	if _, err := s.store.AgentConfig(ctx, customerID, request.Id); err != nil {
		return DisconnectPlugin404JSONResponse{NotFoundJSONResponse{Error: unknownConfig}}, nil
	}
	if err := s.store.DeletePluginConnection(ctx, customerID, request.Id, string(request.PluginId)); err != nil {
		return DisconnectPlugin404JSONResponse{NotFoundJSONResponse{Error: unknownPlugin}}, nil
	}
	if err := s.store.RemoveConfigPlugin(ctx, customerID, request.Id, string(request.PluginId)); err != nil {
		return nil, err
	}
	return DisconnectPlugin204Response{}, nil
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
