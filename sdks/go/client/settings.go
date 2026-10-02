package client

import (
	"context"
	"fmt"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// AppSettings is what the router does for this app.
type AppSettings = acceleration.AppSettings

// Settings is what the router does for this app, as far as the app can see it.
type Settings struct {
	client *Client
}

// Settings is what the router does for this app.
func (c *Client) Settings() *Settings { return &Settings{client: c} }

// App reads this app's settings: which Stream app its conversations, transcripts, calls
// and phone lines are written into, and whether that app holds the agent channel and call
// types they need. Only a backend may ask.
func (s *Settings) App(ctx context.Context) (*AppSettings, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	read, err := api.GetAppSettingsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("client: reading the app's settings: %w", err)
	}
	if read.JSON200 == nil {
		return nil, failure("reading the app's settings", read.Status(), read.JSON401, read.JSON403, read.JSON503)
	}
	return read.JSON200, nil
}
