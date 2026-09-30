package client

import (
	"context"
	"errors"
	"fmt"
	"net/http"
)

// Memories is what agents remember about the app's users between conversations.
type Memories struct {
	client *Client
}

// Memories is what this app's agents remember about its users.
func (c *Client) Memories() *Memories { return &Memories{client: c} }

// Truncate deletes everything remembered about one user: every session's and every agent's,
// whatever memory filter it was written under. userID is the "user_id" of the memory filter
// the sessions were opened with. Only a backend may ask.
func (m *Memories) Truncate(ctx context.Context, userID string) error {
	if userID == "" {
		return errors.New("client: truncating memories needs a user id")
	}
	api, err := m.client.api()
	if err != nil {
		return err
	}

	truncated, err := api.TruncateMemoriesWithResponse(ctx, userID)
	if err != nil {
		return fmt.Errorf("client: truncating the memories of %s: %w", userID, err)
	}
	if truncated.StatusCode() != http.StatusNoContent {
		return failure("truncating the memories of "+userID, truncated.Status(),
			truncated.JSON400, truncated.JSON401, truncated.JSON403)
	}
	return nil
}
