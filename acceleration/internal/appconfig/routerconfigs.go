package appconfig

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// RouterConfig returns one config a customer holds.
func (s *Store) RouterConfig(ctx context.Context, customerID, id string) (store.RouterConfig, error) {
	if customerID == "" || id == "" {
		return s.db.RouterConfig(ctx, customerID, id)
	}
	return read(ctx, s, key("router", customerID, id), func(ctx context.Context) (store.RouterConfig, error) {
		return s.db.RouterConfig(ctx, customerID, id)
	})
}

// RouterConfigByName returns the config a customer holds under this name.
func (s *Store) RouterConfigByName(ctx context.Context, customerID, name string) (store.RouterConfig, bool, error) {
	if customerID == "" || name == "" {
		return s.db.RouterConfigByName(ctx, customerID, name)
	}
	answer, err := read(ctx, s, key("router-name", customerID, name),
		func(ctx context.Context) (found[store.RouterConfig], error) {
			config, exists, err := s.db.RouterConfigByName(ctx, customerID, name)
			return found[store.RouterConfig]{Value: config, Found: exists}, err
		})
	return answer.Value, answer.Found, err
}

// CreateRouterConfig stores a new preset.
func (s *Store) CreateRouterConfig(ctx context.Context, config *store.RouterConfig) error {
	if err := s.db.CreateRouterConfig(ctx, config); err != nil {
		return err
	}
	s.forget(ctx, key("router-name", config.CustomerID, config.Name))
	return nil
}

// UpdateRouterConfig replaces a preset, dropping the name it had as well as the one it has.
func (s *Store) UpdateRouterConfig(ctx context.Context, config *store.RouterConfig) error {
	was, err := s.db.RouterConfig(ctx, config.CustomerID, config.ID)
	if err != nil {
		return err
	}
	if err := s.db.UpdateRouterConfig(ctx, config); err != nil {
		return err
	}
	s.forget(ctx,
		key("router", config.CustomerID, config.ID),
		key("router-name", config.CustomerID, config.Name),
		key("router-name", config.CustomerID, was.Name))
	return nil
}

// DeleteRouterConfig marks a preset as gone.
func (s *Store) DeleteRouterConfig(ctx context.Context, customerID, id string) error {
	was, err := s.db.RouterConfig(ctx, customerID, id)
	if err != nil {
		return err
	}
	if err := s.db.DeleteRouterConfig(ctx, customerID, id); err != nil {
		return err
	}
	s.forget(ctx, key("router", customerID, id), key("router-name", customerID, was.Name))
	return nil
}
