package client

import (
	"context"
	"fmt"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// Routes lists the capability shortcuts a modality offers as a choice, default first, each with
// the models it resolves to now: image-fast and image-quality for acceleration.ModalityImage. Only a
// backend may ask.
func (c *Client) Routes(ctx context.Context, modality acceleration.Modality) ([]acceleration.Route, error) {
	api, err := c.api()
	if err != nil {
		return nil, err
	}

	listed, err := api.ListRoutesWithResponse(ctx, modality)
	if err != nil {
		return nil, fmt.Errorf("client: listing routes: %w", err)
	}
	if listed.JSON200 == nil {
		return nil, failure("listing routes", listed.HTTPResponse, listed.Body)
	}
	return *listed.JSON200, nil
}
