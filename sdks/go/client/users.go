package client

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// Users is what the router knows about the app's users.
type Users struct {
	client *Client
}

// Users is what the router knows about this app's users.
func (c *Client) Users() *Users { return &Users{client: c} }

// LinkPhone says that the app's user userID and the phone number phone, in E.164, are one
// person, so their app chat and their texts and calls from that number share one omni-channel
// with each agent. It returns the number as the router keeps it. A number another user of the
// app holds is refused (409 phone_taken). Only a backend may ask.
func (u *Users) LinkPhone(ctx context.Context, userID, phone string) (string, error) {
	if phone == "" {
		return "", errors.New("client: linking a user needs a phone number; UnlinkPhone removes one")
	}
	user, err := u.put(ctx, userID, &phone)
	if err != nil {
		return "", err
	}
	if user.Phone == nil {
		return "", fmt.Errorf("client: linking %s answered no number", userID)
	}
	return *user.Phone, nil
}

// UnlinkPhone removes the number linked to the app's user userID, if any. What was joined
// before stays joined. Only a backend may ask.
func (u *Users) UnlinkPhone(ctx context.Context, userID string) error {
	_, err := u.put(ctx, userID, nil)
	return err
}

func (u *Users) put(ctx context.Context, userID string, phone *string) (acceleration.AgentUser, error) {
	if userID == "" {
		return acceleration.AgentUser{}, errors.New("client: linking a user needs a user id")
	}
	api, err := u.client.api()
	if err != nil {
		return acceleration.AgentUser{}, err
	}
	updated, err := api.UpdateUserWithResponse(ctx, userID, acceleration.UpdateUserJSONRequestBody{Phone: phone})
	if err != nil {
		return acceleration.AgentUser{}, fmt.Errorf("client: linking %s: %w", userID, err)
	}
	if updated.JSON200 == nil {
		return acceleration.AgentUser{}, failure("linking "+userID, updated.HTTPResponse, updated.Body)
	}
	return *updated.JSON200, nil
}
