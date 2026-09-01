package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"
)

// UpsertPluginConnection writes a pending or connected login for one plugin on one config.
// A second authorize of the same plugin replaces the previous attempt rather than leaving
// two pending rows.
func (s *Store) UpsertPluginConnection(ctx context.Context, conn *PluginConnection) error {
	if conn.CustomerID == "" || conn.ConfigID == "" || conn.PluginID == "" {
		return errors.New("store: a customer, a config and a plugin are required")
	}

	now := time.Now().UTC()
	conn.UpdatedAt = now
	conn.DeletedAt = nil
	if conn.Status == "" {
		conn.Status = PluginPending
	}

	existing, err := s.pluginConnection(ctx, conn.CustomerID, conn.ConfigID, conn.PluginID)
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
			return fmt.Errorf("store: update plugin connection: %w", err)
		}
		return nil
	}
	if !isUnknownPlugin(err) {
		return err
	}

	conn.ID = newID()
	conn.CreatedAt = now
	if _, err := s.db.NewInsert().Model(conn).Exec(ctx); err != nil {
		return fmt.Errorf("store: create plugin connection: %w", err)
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

// PluginConnections returns every login one config holds, newest first.
func (s *Store) PluginConnections(ctx context.Context, customerID, configID string) ([]PluginConnection, error) {
	if customerID == "" || configID == "" {
		return nil, errors.New("store: a customer and a config are required")
	}

	var conns []PluginConnection
	err := s.db.NewSelect().Model(&conns).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: plugin connections: %w", err)
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

// SavePluginConnection writes tokens and status after the callback, or after a refresh.
func (s *Store) SavePluginConnection(ctx context.Context, conn *PluginConnection) error {
	if conn.ID == "" {
		return errors.New("store: a plugin connection id is required")
	}
	conn.UpdatedAt = time.Now().UTC()
	result, err := s.db.NewUpdate().Model(conn).
		Column("instance_url", "access_token", "refresh_token", "expires_at", "status",
			"oauth_state", "code_verifier", "client_id", "token_endpoint", "updated_at").
		Where("id = ?", conn.ID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save plugin connection: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: save plugin connection: %w", err)
	}
	if affected == 0 {
		return unknownPluginConnection(conn.ID)
	}
	return nil
}

// DeletePluginConnection marks a login as gone.
func (s *Store) DeletePluginConnection(ctx context.Context, customerID, configID, pluginID string) error {
	if customerID == "" || configID == "" || pluginID == "" {
		return errors.New("store: a customer, a config and a plugin are required")
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
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete plugin connection: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete plugin connection: %w", err)
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
	for _, named := range config.Plugins {
		if named == pluginID {
			return nil
		}
	}
	config.Plugins = append(config.Plugins, pluginID)
	return s.UpdateAgentConfig(ctx, &config)
}

// RemoveConfigPlugin drops a plugin name from a config.
func (s *Store) RemoveConfigPlugin(ctx context.Context, customerID, configID, pluginID string) error {
	config, err := s.AgentConfig(ctx, customerID, configID)
	if err != nil {
		return err
	}
	kept := make([]string, 0, len(config.Plugins))
	for _, named := range config.Plugins {
		if named != pluginID {
			kept = append(kept, named)
		}
	}
	config.Plugins = kept
	return s.UpdateAgentConfig(ctx, &config)
}

func (s *Store) pluginConnection(ctx context.Context, customerID, configID, pluginID string) (PluginConnection, error) {
	var conn PluginConnection
	err := s.db.NewSelect().Model(&conn).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("plugin_id = ?", pluginID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PluginConnection{}, unknownPluginConnection(pluginID)
	}
	if err != nil {
		return PluginConnection{}, fmt.Errorf("store: plugin connection: %w", err)
	}
	return conn, nil
}

func unknownPluginConnection(id string) error {
	return fmt.Errorf("store: there is no plugin connection %s", id)
}

func isUnknownPlugin(err error) bool {
	return err != nil && strings.Contains(err.Error(), "there is no plugin connection")
}
