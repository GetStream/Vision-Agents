package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// ErrUnknownPluginEventSubscription is a callback token nobody subscribed with, or one whose
// subscription was dropped.
var ErrUnknownPluginEventSubscription = errors.New("store: there is no such plugin event subscription")

// PluginLogins returns every connected login to one plugin on one config: the app's own and
// each end user's.
func (s *Store) PluginLogins(ctx context.Context, customerID, configID, pluginID string) ([]PluginConnection, error) {
	if customerID == "" || configID == "" || pluginID == "" {
		return nil, errors.New("store: a customer, a config and a plugin are required")
	}

	var conns []PluginConnection
	err := s.db.NewSelect().Model(&conns).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("plugin_id = ?", pluginID).
		Where("status = ?", PluginConnected).
		Where("access_token <> ''").
		Where("deleted_at IS NULL").
		Order("created_at").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: plugin logins: %w", err)
	}
	return conns, nil
}

// PluginEventConfigs returns every config that declares plugin events or still holds a
// subscription, deleted ones included, which is what has to be kept in step with its servers.
func (s *Store) PluginEventConfigs(ctx context.Context) ([]AgentConfig, error) {
	var configs []AgentConfig
	err := s.db.NewSelect().Model(&configs).
		WhereOr("(plugin_events <> '[]' AND deleted_at IS NULL)").
		WhereOr("id IN (SELECT config_id FROM agent_plugin_event_subscriptions WHERE deleted_at IS NULL)").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: plugin event configs: %w", err)
	}
	return configs, nil
}

// PluginEventSubscriptions returns the subscriptions one config holds.
func (s *Store) PluginEventSubscriptions(ctx context.Context, customerID, configID string) ([]PluginEventSubscription, error) {
	if customerID == "" || configID == "" {
		return nil, errors.New("store: a customer and a config are required")
	}

	var subs []PluginEventSubscription
	err := s.db.NewSelect().Model(&subs).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("deleted_at IS NULL").
		Order("created_at").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: plugin event subscriptions: %w", err)
	}
	return subs, nil
}

// PluginEventSubscriptionByToken finds the subscription a delivery is addressed to.
func (s *Store) PluginEventSubscriptionByToken(ctx context.Context, token string) (PluginEventSubscription, error) {
	if token == "" {
		return PluginEventSubscription{}, ErrUnknownPluginEventSubscription
	}

	var sub PluginEventSubscription
	err := s.db.NewSelect().Model(&sub).
		Where("token = ?", token).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PluginEventSubscription{}, ErrUnknownPluginEventSubscription
	}
	if err != nil {
		return PluginEventSubscription{}, fmt.Errorf("store: plugin event subscription: %w", err)
	}
	return sub, nil
}

// SavePluginEventSubscription creates a subscription without an id and writes one with.
func (s *Store) SavePluginEventSubscription(ctx context.Context, sub *PluginEventSubscription) error {
	if sub.CustomerID == "" || sub.ConfigID == "" || sub.PluginID == "" || sub.Event == "" {
		return errors.New("store: a customer, a config, a plugin and an event are required")
	}
	now := time.Now().UTC()
	sub.UpdatedAt = now
	if sub.Arguments == nil {
		sub.Arguments = map[string]any{}
	}

	if sub.ID == "" {
		sub.ID = newID()
		sub.CreatedAt = now
		if _, err := s.db.NewInsert().Model(sub).Exec(ctx); err != nil {
			return fmt.Errorf("store: create plugin event subscription: %w", err)
		}
		return nil
	}
	_, err := s.db.NewUpdate().Model(sub).
		Column("secret", "remote_id", "refresh_before", "status", "error", "updated_at").
		Where("id = ?", sub.ID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save plugin event subscription: %w", err)
	}
	return nil
}

// DeletePluginEventSubscription drops a subscription, so deliveries to it are refused.
func (s *Store) DeletePluginEventSubscription(ctx context.Context, id string) error {
	_, err := s.db.NewUpdate().Model((*PluginEventSubscription)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Set("secret = ?", "").
		Where("id = ?", id).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete plugin event subscription: %w", err)
	}
	return nil
}

// ClaimPluginEvent records an event as delivered. False is one already delivered, which a
// retry of the same event is.
func (s *Store) ClaimPluginEvent(ctx context.Context, subscriptionID, eventID string) (bool, error) {
	result, err := s.db.NewRaw(
		"INSERT INTO agent_plugin_event_deliveries (subscription_id, event_id, received_at) "+
			"VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
		subscriptionID, eventID, time.Now().UTC()).Exec(ctx)
	if err != nil {
		return false, fmt.Errorf("store: claim plugin event: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("store: claim plugin event: %w", err)
	}
	return affected == 1, nil
}
