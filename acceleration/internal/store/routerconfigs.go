package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// CreateRouterConfig stores a new config and fills in its id and timestamps.
func (s *Store) CreateRouterConfig(ctx context.Context, config *RouterConfig) error {
	if config.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if config.Name == "" {
		return errors.New("store: a router config needs a name")
	}

	config.ID = newID()
	now := time.Now().UTC()
	config.CreatedAt = now
	config.UpdatedAt = now
	config.DeletedAt = nil
	normalizeRouterConfig(config)

	if _, err := s.db.NewInsert().Model(config).Exec(ctx); err != nil {
		return fmt.Errorf("store: create router config: %w", err)
	}
	return nil
}

// UpdateRouterConfig replaces a config a customer holds. Every field is written, so an
// update is what the config now is rather than what changed about it.
func (s *Store) UpdateRouterConfig(ctx context.Context, config *RouterConfig) error {
	if config.CustomerID == "" || config.ID == "" {
		return errors.New("store: a customer and a config id are required")
	}
	if config.Name == "" {
		return errors.New("store: a router config needs a name")
	}

	config.UpdatedAt = time.Now().UTC()
	normalizeRouterConfig(config)

	result, err := s.db.NewUpdate().Model(config).
		Column("name", "stt", "tts", "llm", "search", "tags", "updated_at").
		Where("id = ?", config.ID).
		Where("customer_id = ?", config.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: update router config: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: update router config: %w", err)
	}
	if affected == 0 {
		return unknownRouterConfig(config.ID)
	}
	return nil
}

// DeleteRouterConfig marks a config as gone. The row stays, because the requests that ran
// under it still name it.
func (s *Store) DeleteRouterConfig(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a config id are required")
	}

	result, err := s.db.NewUpdate().Model((*RouterConfig)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete router config: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete router config: %w", err)
	}
	if affected == 0 {
		return unknownRouterConfig(id)
	}
	return nil
}

// RouterConfig returns one config a customer holds.
func (s *Store) RouterConfig(ctx context.Context, customerID, id string) (RouterConfig, error) {
	if customerID == "" || id == "" {
		return RouterConfig{}, errors.New("store: a customer and a config id are required")
	}

	var config RouterConfig
	err := s.db.NewSelect().Model(&config).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return RouterConfig{}, unknownRouterConfig(id)
	}
	if err != nil {
		return RouterConfig{}, fmt.Errorf("store: router config: %w", err)
	}
	return config, nil
}

// RouterConfigByName returns the config a customer holds under this name, which is how a
// caller that names a config rather than an id reaches one.
func (s *Store) RouterConfigByName(ctx context.Context, customerID, name string) (RouterConfig, bool, error) {
	if customerID == "" || name == "" {
		return RouterConfig{}, false, errors.New("store: a customer and a config name are required")
	}

	var config RouterConfig
	err := s.db.NewSelect().Model(&config).
		Where("customer_id = ?", customerID).
		Where("name = ?", name).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return RouterConfig{}, false, nil
	}
	if err != nil {
		return RouterConfig{}, false, fmt.Errorf("store: router config by name: %w", err)
	}
	return config, true, nil
}

// CustomerRouterConfigs returns the configs a customer holds, newest first.
func (s *Store) CustomerRouterConfigs(ctx context.Context, customerID string) ([]RouterConfig, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var configs []RouterConfig
	err := s.db.NewSelect().Model(&configs).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: customer router configs: %w", err)
	}
	return configs, nil
}

// normalizeRouterConfig fills in the map a nil would write as null, which the column is
// not.
func normalizeRouterConfig(config *RouterConfig) {
	if config.Tags == nil {
		config.Tags = map[string]string{}
	}
}

func unknownRouterConfig(id string) error {
	return fmt.Errorf("store: there is no router config %s", id)
}
