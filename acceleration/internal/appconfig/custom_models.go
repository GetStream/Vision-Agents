package appconfig

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// CustomModelNamed returns the model a customer calls this. It is read every time a session
// opens on one, and its key stays sealed in the cache as it is in Postgres.
func (s *Store) CustomModelNamed(ctx context.Context, customerID, name string) (store.CustomModel, error) {
	if customerID == "" || name == "" {
		return s.db.CustomModelNamed(ctx, customerID, name)
	}
	return read(ctx, s, key("model-name", customerID, name), func(ctx context.Context) (store.CustomModel, error) {
		return s.db.CustomModelNamed(ctx, customerID, name)
	})
}

// CreateCustomModel stores a model a customer serves themselves.
func (s *Store) CreateCustomModel(ctx context.Context, model *store.CustomModel) error {
	if err := s.db.CreateCustomModel(ctx, model); err != nil {
		return err
	}
	s.forget(ctx, key("model-name", model.CustomerID, model.Name))
	return nil
}

// UpdateCustomModel replaces a model, dropping the name it had as well as the one it has.
func (s *Store) UpdateCustomModel(ctx context.Context, model *store.CustomModel) error {
	was, err := s.db.CustomModel(ctx, model.CustomerID, model.ID)
	if err != nil {
		return err
	}
	if err := s.db.UpdateCustomModel(ctx, model); err != nil {
		return err
	}
	s.forget(ctx,
		key("model-name", model.CustomerID, model.Name),
		key("model-name", model.CustomerID, was.Name))
	return nil
}

// DeleteCustomModel removes a model, so a session naming it is refused from then on.
func (s *Store) DeleteCustomModel(ctx context.Context, customerID, id string) error {
	was, err := s.db.CustomModel(ctx, customerID, id)
	if err != nil {
		return err
	}
	if err := s.db.DeleteCustomModel(ctx, customerID, id); err != nil {
		return err
	}
	s.forget(ctx, key("model-name", customerID, was.Name))
	return nil
}
