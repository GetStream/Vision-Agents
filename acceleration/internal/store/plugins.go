package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// UpsertPluginConnection writes a pending or connected login for one plugin on one config,
// the app's own or, with a UserID, one end user's. A second authorize of the same plugin by
// the same owner replaces the previous attempt rather than leaving two pending rows.
func (s *Store) UpsertPluginConnection(ctx context.Context, conn *PluginConnection) error {
	if conn.CustomerID == "" || conn.ConfigID == "" || conn.PluginID == "" {
		return stack.Wrap(errors.New("store: a customer, a config and a plugin are required"))
	}

	now := time.Now().UTC()
	conn.UpdatedAt = now
	conn.DeletedAt = nil
	if conn.Status == "" {
		conn.Status = PluginPending
	}

	existing, err := s.pluginConnection(ctx, conn.CustomerID, conn.ConfigID, conn.UserID, conn.PluginID)
	if err == nil {
		conn.ID = existing.ID
		conn.CreatedAt = existing.CreatedAt
		_, err := s.db.NewUpdate().Model(conn).
			Column("instance_url", "access_token", "refresh_token", "expires_at", "status",
				"oauth_state", "code_verifier", "client_id", "token_endpoint", "updated_at",
				"deleted_at").
			Where("id = ?", conn.ID).
			Exec(ctx)
		if err != nil {
			return stack.Wrap(fmt.Errorf("store: update plugin connection: %w", err))
		}
		return nil
	}
	if !isUnknownPlugin(err) {
		return err
	}

	conn.ID = newID()
	conn.CreatedAt = now
	if _, err := s.db.NewInsert().Model(conn).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: create plugin connection: %w", err))
	}
	return nil
}

// PluginConnectionByState finds the pending login the OAuth callback is finishing.
func (s *Store) PluginConnectionByState(ctx context.Context, state string) (PluginConnection, error) {
	if state == "" {
		return PluginConnection{}, errors.New("store: an oauth state is required")
	}

	var conn PluginConnection
	err := s.db.NewSelect().Model(&conn).
		Where("oauth_state = ?", state).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PluginConnection{}, unknownPluginConnection(state)
	}
	if err != nil {
		return PluginConnection{}, fmt.Errorf("store: plugin connection by state: %w", err)
	}
	return conn, nil
}

// PluginConnections returns every login the app holds on one config, newest first. An end
// user's own logins are not among them: every session of the config would use them.
func (s *Store) PluginConnections(ctx context.Context, customerID, configID string) ([]PluginConnection, error) {
	if customerID == "" || configID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a config are required"))
	}

	var conns []PluginConnection
	err := s.db.NewSelect().Model(&conns).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("user_id = ''").
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: plugin connections: %w", err))
	}
	return conns, nil
}

// ConnectedPlugins returns the logins a session may actually use.
func (s *Store) ConnectedPlugins(ctx context.Context, customerID, configID string) ([]PluginConnection, error) {
	conns, err := s.PluginConnections(ctx, customerID, configID)
	if err != nil {
		return nil, err
	}
	ready := make([]PluginConnection, 0, len(conns))
	for _, conn := range conns {
		if conn.Status == PluginConnected && conn.AccessToken != "" {
			ready = append(ready, conn)
		}
	}
	return ready, nil
}

// UserPluginConnection is the login one end user made to one plugin on one config.
func (s *Store) UserPluginConnection(ctx context.Context, customerID, configID, userID, pluginID string) (PluginConnection, error) {
	if customerID == "" || configID == "" || userID == "" || pluginID == "" {
		return PluginConnection{}, errors.New("store: a customer, a config, a user and a plugin are required")
	}
	return s.pluginConnection(ctx, customerID, configID, userID, pluginID)
}

// SavePluginConnection writes tokens and status after the callback, or after a refresh.
func (s *Store) SavePluginConnection(ctx context.Context, conn *PluginConnection) error {
	if conn.ID == "" {
		return stack.Wrap(errors.New("store: a plugin connection id is required"))
	}
	conn.UpdatedAt = time.Now().UTC()
	result, err := s.db.NewUpdate().Model(conn).
		Column("instance_url", "access_token", "refresh_token", "expires_at", "status",
			"oauth_state", "code_verifier", "client_id", "token_endpoint", "updated_at").
		Where("id = ?", conn.ID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save plugin connection: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save plugin connection: %w", err))
	}
	if affected == 0 {
		return unknownPluginConnection(conn.ID)
	}
	return nil
}

// DeletePluginConnection marks the app's login as gone.
func (s *Store) DeletePluginConnection(ctx context.Context, customerID, configID, pluginID string) error {
	if customerID == "" || configID == "" || pluginID == "" {
		return stack.Wrap(errors.New("store: a customer, a config and a plugin are required"))
	}

	result, err := s.db.NewUpdate().Model((*PluginConnection)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Set("access_token = ?", "").
		Set("refresh_token = ?", "").
		Set("oauth_state = ?", "").
		Set("code_verifier = ?", "").
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("plugin_id = ?", pluginID).
		Where("user_id = ''").
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete plugin connection: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete plugin connection: %w", err))
	}
	if affected == 0 {
		return unknownPluginConnection(pluginID)
	}
	return nil
}

// AddConfigPlugin names a plugin on a config if it is not already there.
func (s *Store) AddConfigPlugin(ctx context.Context, customerID, configID, pluginID string) error {
	config, err := s.AgentConfig(ctx, customerID, configID)
	if err != nil {
		return err
	}
	if NamesPlugin(config.AgentPlugins, pluginID) {
		return nil
	}
	config.AgentPlugins = append(config.AgentPlugins, PluginEntry{Name: pluginID})
	return s.UpdateAgentConfig(ctx, &config)
}

// ErrUnknownPluginClient is a plugin no OAuth client was set for on the config.
var ErrUnknownPluginClient = errors.New("store: no client is set for this plugin")

// SavePluginClient sets the OAuth client a config logs into a plugin with, replacing the
// one set before.
func (s *Store) SavePluginClient(ctx context.Context, client *PluginClient) error {
	if client.CustomerID == "" || client.ConfigID == "" || client.PluginID == "" || client.ClientID == "" {
		return stack.Wrap(errors.New("store: a customer, a config, a plugin and a client id are required"))
	}
	now := time.Now().UTC()
	client.CreatedAt = now
	client.UpdatedAt = now
	_, err := s.db.NewInsert().Model(client).
		On("CONFLICT (customer_id, config_id, plugin_id) DO UPDATE").
		Set("client_id = EXCLUDED.client_id").
		Set("secret_sealed = EXCLUDED.secret_sealed").
		Set("kek_version = EXCLUDED.kek_version").
		Set("updated_at = EXCLUDED.updated_at").
		Returning("created_at").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save plugin client: %w", err))
	}
	return nil
}

// PluginClient is the OAuth client a config logs into a plugin with.
func (s *Store) PluginClient(ctx context.Context, customerID, configID, pluginID string) (PluginClient, error) {
	var client PluginClient
	err := s.db.NewSelect().Model(&client).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("plugin_id = ?", pluginID).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PluginClient{}, stack.Wrap(ErrUnknownPluginClient)
	}
	if err != nil {
		return PluginClient{}, stack.Wrap(fmt.Errorf("store: plugin client: %w", err))
	}
	return client, nil
}

// PluginClients are the plugins a config has an OAuth client set for, by plugin id.
func (s *Store) PluginClients(ctx context.Context, customerID, configID string) (map[string]PluginClient, error) {
	var clients []PluginClient
	err := s.db.NewSelect().Model(&clients).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: plugin clients: %w", err))
	}
	byPlugin := make(map[string]PluginClient, len(clients))
	for _, client := range clients {
		byPlugin[client.PluginID] = client
	}
	return byPlugin, nil
}

// DeletePluginClient forgets the OAuth client a config logs into a plugin with.
func (s *Store) DeletePluginClient(ctx context.Context, customerID, configID, pluginID string) error {
	result, err := s.db.NewDelete().Model((*PluginClient)(nil)).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("plugin_id = ?", pluginID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete plugin client: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete plugin client: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(ErrUnknownPluginClient)
	}
	return nil
}

func (s *Store) pluginConnection(ctx context.Context, customerID, configID, userID, pluginID string) (PluginConnection, error) {
	var conn PluginConnection
	err := s.db.NewSelect().Model(&conn).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("user_id = ?", userID).
		Where("plugin_id = ?", pluginID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PluginConnection{}, unknownPluginConnection(pluginID)
	}
	if err != nil {
		return PluginConnection{}, stack.Wrap(fmt.Errorf("store: plugin connection: %w", err))
	}
	return conn, nil
}

func unknownPluginConnection(id string) error {
	return stack.Wrap(fmt.Errorf("store: there is no plugin connection %s", id))
}

func isUnknownPlugin(err error) bool {
	return err != nil && strings.Contains(err.Error(), "there is no plugin connection")
}
