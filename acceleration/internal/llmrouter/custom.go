package llmrouter

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/custom"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// CustomModels resolves the models customers serve themselves, which a session names as
// custom/<name>.
type CustomModels struct {
	store   *appconfig.Store
	secrets *auth.Sealer
	client  *http.Client
}

// NewCustomModels reads models from the configuration store. secrets opens their keys and
// may be nil on a deployment that stores none. private lets an endpoint sit on a private
// address, which only a router nobody else's tenants share should allow.
func NewCustomModels(configs *appconfig.Store, secrets *auth.Sealer, private bool) *CustomModels {
	client := egress.NewClient(0, nil)
	if private {
		client = &http.Client{}
	}
	return &CustomModels{store: configs, secrets: secrets, client: client}
}

// CustomModel returns how to route to the customer's model of that name.
func (m *CustomModels) CustomModel(ctx context.Context, customerID, name string) (routing.ProviderConfig, routing.Endpoint, error) {
	stored, err := m.store.CustomModelNamed(ctx, customerID, name)
	if errors.Is(err, store.ErrNoCustomModel) {
		return routing.ProviderConfig{}, routing.Endpoint{}, stack.Wrap(
			fmt.Errorf("routing: %q is not one of this customer's models", routing.CustomProvider+"/"+name))
	}
	if err != nil {
		return routing.ProviderConfig{}, routing.Endpoint{}, err
	}

	endpoint := routing.Endpoint{BaseURL: stored.BaseURL, Model: stored.Model, Client: m.client}
	if len(stored.APIKeySealed) > 0 {
		if m.secrets == nil {
			return routing.ProviderConfig{}, routing.Endpoint{}, stack.Wrap(
				errors.New("routing: this deployment has no key to open a model's api key with"))
		}
		endpoint.APIKey, err = m.secrets.OpenWithAADVersion(stored.APIKeySealed, []byte(customerID), stored.KEKVersion)
		if err != nil {
			return routing.ProviderConfig{}, routing.Endpoint{}, err
		}
	}
	return CustomModelConfig(stored), endpoint, nil
}

// CustomModelConfig is what routing knows about a customer's model: everything but where
// it is and how to get in.
func CustomModelConfig(stored store.CustomModel) routing.ProviderConfig {
	return routing.ProviderConfig{
		Provider:        custom.ProviderName,
		Model:           stored.Name,
		Realtime:        true,
		InputModalities: stored.InputModalities,
		ContextWindow:   stored.ContextWindow,
		DataPolicy: options.DataHandling{
			TrainsOnData: options.Claim(stored.TrainsOnData),
			Retention:    options.Retention(stored.Retention),
		},
		Price: routing.Price{
			PerMillionInputTokens:  stored.PerMillionInputTokens,
			PerMillionOutputTokens: stored.PerMillionOutputTokens,
		},
	}
}
