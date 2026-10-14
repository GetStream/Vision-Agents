package client

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// Detect finds things in one image and returns their boxes, as fractions of the image.
//
// image is a JPEG or PNG. options picks the model and what to keep, and may be nil for
// detection-fast's defaults. A detection that reached a provider and failed is returned
// rather than raised: its Status is failed, with an ErrorCode saying why. Only a backend may
// ask.
func (c *Client) Detect(ctx context.Context, image []byte, options *acceleration.DetectionOptions) (*acceleration.DetectionResult, error) {
	if len(image) == 0 {
		return nil, errors.New("client: detecting needs an image")
	}
	api, err := c.api()
	if err != nil {
		return nil, err
	}

	detected, err := api.DetectWithResponse(ctx, acceleration.DetectionRequest{Image: image, Options: options})
	if err != nil {
		return nil, fmt.Errorf("client: detecting: %w", err)
	}
	if detected.JSON200 == nil {
		return nil, failure("detecting", detected.HTTPResponse, detected.Body)
	}
	return detected.JSON200, nil
}
