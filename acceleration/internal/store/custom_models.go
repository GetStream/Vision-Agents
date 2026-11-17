package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNoCustomModel says a customer holds no model by that id or name.
var ErrNoCustomModel = errors.New("store: no such model")

const (
	defaultCustomModelLimit = 25
	maxCustomModelLimit     = 200
)

// CustomModelLimit is the page size a model list uses for the limit asked for. CustomModels
// returns one row more than this, the same as the session queries.
func CustomModelLimit(asked int) int {
	return clampLimit(asked, defaultCustomModelLimit, maxCustomModelLimit)
}

// CustomModel is a language model a customer serves themselves behind an OpenAI-compatible
// endpoint, which a session names as custom/<name>.
type CustomModel struct {
	bun.BaseModel `bun:"table:custom_models,alias:cm"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Name       string `bun:"name,notnull"`
	// BaseURL is the endpoint root, up to and including /v1.
	BaseURL string `bun:"base_url,notnull"`
	// Model is the id the endpoint serves the weights under.
	Model string `bun:"model,notnull"`
	// APIKeySealed is the key the endpoint wants, sealed under KEKVersion and bound to the
	// customer. Nil for an endpoint that takes none.
	APIKeySealed    []byte   `bun:"api_key_sealed"`
	KEKVersion      int      `bun:"kek_version,notnull"`
	ContextWindow   int64    `bun:"context_window,notnull"`
	InputModalities []string `bun:"input_modalities,array"`
	// The prices are what the customer is billed by whoever serves the model, so usage can
	// say what a conversation cost. Zero records it as free.
	PerMillionInputTokens  float64   `bun:"per_million_input_tokens,notnull"`
	PerMillionOutputTokens float64   `bun:"per_million_output_tokens,notnull"`
	TrainsOnData           string    `bun:"trains_on_data,notnull"`
	Retention              string    `bun:"retention,notnull"`
	CreatedAt              time.Time `bun:"created_at,notnull"`
	UpdatedAt              time.Time `bun:"updated_at,notnull"`
}

// CustomModelPosition is where a page of models ends.
type CustomModelPosition struct {
	CreatedAt time.Time `json:"c"`
	ID        string    `json:"id"`
}

// CreateCustomModel stores a new model and fills in its id and timestamps.
func (s *Store) CreateCustomModel(ctx context.Context, model *CustomModel) error {
	if model.CustomerID == "" || model.Name == "" {
		return stack.Wrap(errors.New("store: a model needs a customer and a name"))
	}

	model.ID = newID()
	if model.InputModalities == nil {
		model.InputModalities = []string{}
	}
	now := time.Now().UTC()
	model.CreatedAt = now
	model.UpdatedAt = now

	if _, err := s.db.NewInsert().Model(model).Exec(ctx); err != nil {
		if constraint(err) == "custom_models_name_idx" {
			return stack.Wrap(ErrNameTaken)
		}
		return stack.Wrap(fmt.Errorf("store: create model: %w", err))
	}
	return nil
}

// UpdateCustomModel replaces everything about a model but its id, owner and creation.
func (s *Store) UpdateCustomModel(ctx context.Context, model *CustomModel) error {
	if model.CustomerID == "" || model.ID == "" || model.Name == "" {
		return stack.Wrap(errors.New("store: a customer, a model id and a name are required"))
	}

	if model.InputModalities == nil {
		model.InputModalities = []string{}
	}
	model.UpdatedAt = time.Now().UTC()
	result, err := s.db.NewUpdate().Model(model).
		ExcludeColumn("id", "customer_id", "created_at").
		Where("id = ?", model.ID).
		Where("customer_id = ?", model.CustomerID).
		Exec(ctx)
	if constraint(err) == "custom_models_name_idx" {
		return stack.Wrap(ErrNameTaken)
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: update model: %w", err))
	}
	return affectedModel(result, model.ID)
}

// DeleteCustomModel removes a model. A session that names it afterwards is refused.
func (s *Store) DeleteCustomModel(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a model id are required"))
	}

	result, err := s.db.NewDelete().Model((*CustomModel)(nil)).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete model: %w", err))
	}
	return affectedModel(result, id)
}

// CustomModel returns one of a customer's models by id.
func (s *Store) CustomModel(ctx context.Context, customerID, id string) (CustomModel, error) {
	return s.customModelWhere(ctx, customerID, "id", id)
}

// CustomModelNamed returns the customer's model of that name, which is what a session names.
func (s *Store) CustomModelNamed(ctx context.Context, customerID, name string) (CustomModel, error) {
	return s.customModelWhere(ctx, customerID, "name", name)
}

// CustomModels returns a page of a customer's models, newest first.
func (s *Store) CustomModels(ctx context.Context, customerID string, limit int, after *CustomModelPosition) ([]CustomModel, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	var models []CustomModel
	query := s.db.NewSelect().Model(&models).Where("customer_id = ?", customerID)
	if after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	err := query.Order("created_at DESC", "id DESC").Limit(CustomModelLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: custom models: %w", err))
	}
	return models, nil
}

func (s *Store) customModelWhere(ctx context.Context, customerID, column, value string) (CustomModel, error) {
	if customerID == "" || value == "" {
		return CustomModel{}, stack.Wrap(errors.New("store: a customer and a model are required"))
	}

	var model CustomModel
	err := s.db.NewSelect().Model(&model).
		Where("customer_id = ?", customerID).
		Where("? = ?", bun.Ident(column), value).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return CustomModel{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoCustomModel, value))
	}
	if err != nil {
		return CustomModel{}, stack.Wrap(fmt.Errorf("store: custom model: %w", err))
	}
	return model, nil
}

func affectedModel(result sql.Result, id string) error {
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: model: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrNoCustomModel, id))
	}
	return nil
}
