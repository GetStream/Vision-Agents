package stream

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// DefineModel stores a language model you serve yourself behind an OpenAI-compatible
// endpoint, which a router config or session then names as custom/<name>.
//
// It is written by name, so calling this twice edits what is stored rather than storing a
// second copy of it. An APIKey left nil keeps the key already stored.
//
//	key := os.Getenv("BASETEN_API_KEY")
//	_, err := client.DefineModel(ctx, acceleration.CustomModelRequest{
//	    Name:    "support-qwen",
//	    BaseUrl: "https://model-abc123.api.baseten.co/environments/production/sync/v1",
//	    Model:   "Qwen/Qwen3.8-27B",
//	    ApiKey:  &key,
//	})
func (c *Client) DefineModel(
	ctx context.Context,
	wanted acceleration.CustomModelRequest,
) (*acceleration.CustomModel, error) {
	if strings.TrimSpace(wanted.Name) == "" {
		return nil, errors.New("stream: a model needs a name")
	}

	client, err := c.backend.Client()
	if err != nil {
		return nil, err
	}

	stored, err := namedModel(ctx, client, wanted.Name)
	if err != nil {
		return nil, err
	}
	if stored != nil {
		updated, err := client.UpdateCustomModelWithResponse(ctx, stored.Id, wanted)
		if err != nil {
			return nil, fmt.Errorf("stream: updating model %s: %w", stored.Id, err)
		}
		if updated.JSON200 == nil {
			return nil, refusal(updated.HTTPResponse, updated.Body)
		}
		return updated.JSON200, nil
	}

	created, err := client.CreateCustomModelWithResponse(ctx, wanted)
	if err != nil {
		return nil, fmt.Errorf("stream: creating model %s: %w", wanted.Name, err)
	}
	if created.JSON201 == nil {
		return nil, refusal(created.HTTPResponse, created.Body)
	}
	return created.JSON201, nil
}

// DeleteModel removes the model of that name. Sessions naming it are refused from then on.
// A name nothing is stored under is not an error.
func (c *Client) DeleteModel(ctx context.Context, name string) error {
	client, err := c.backend.Client()
	if err != nil {
		return err
	}

	stored, err := namedModel(ctx, client, name)
	if err != nil || stored == nil {
		return err
	}
	deleted, err := client.DeleteCustomModelWithResponse(ctx, stored.Id)
	if err != nil {
		return fmt.Errorf("stream: deleting model %s: %w", stored.Id, err)
	}
	if deleted.StatusCode() != http.StatusNoContent {
		return refusal(deleted.HTTPResponse, deleted.Body)
	}
	return nil
}

// namedModel finds the customer's model of that name, or nil.
func namedModel(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	name string,
) (*acceleration.CustomModel, error) {
	params := &acceleration.ListCustomModelsParams{}
	for {
		listed, err := client.ListCustomModelsWithResponse(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("stream: listing models: %w", err)
		}
		if listed.JSON200 == nil {
			return nil, refusal(listed.HTTPResponse, listed.Body)
		}
		for _, model := range listed.JSON200.Items {
			if model.Name == name {
				return &model, nil
			}
		}
		if !listed.JSON200.HasMore {
			return nil, nil
		}
		params.Cursor = listed.JSON200.NextCursor
	}
}
