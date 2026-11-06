package client

import (
	"context"
	"errors"
	"fmt"
	"time"

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

// PhoneHistory is how much of what a number said before its link the user reads: all of it, or
// what it said from Since on. The zero value is none of it, from the request on.
type PhoneHistory struct {
	All   bool
	Since time.Time
}

// LinkPhoneWithHistory is LinkPhone with how much of the number's past the user reads, in their
// cards and in the sessions on the number's threads. It returns the user as the router keeps them.
func (u *Users) LinkPhoneWithHistory(ctx context.Context, userID, phone string, history PhoneHistory) (acceleration.AgentUser, error) {
	if phone == "" {
		return acceleration.AgentUser{}, errors.New("client: linking a user needs a phone number; UnlinkPhone removes one")
	}
	body := acceleration.UpdateUserJSONRequestBody{Phone: &phone}
	if history.Since.IsZero() {
		body.IncludePhoneHistory = &history.All
	} else {
		body.PhoneHistorySince = &history.Since
	}
	return u.update(ctx, userID, body)
}

// Get is the app's user userID as the router keeps their link: the number, and how much of its
// past they read. A user with no number has a nil Phone. Only a backend may ask.
func (u *Users) Get(ctx context.Context, userID string) (acceleration.AgentUser, error) {
	if userID == "" {
		return acceleration.AgentUser{}, errors.New("client: reading a user needs a user id")
	}
	api, err := u.client.api()
	if err != nil {
		return acceleration.AgentUser{}, err
	}
	got, err := api.GetUserWithResponse(ctx, userID)
	if err != nil {
		return acceleration.AgentUser{}, fmt.Errorf("client: reading %s: %w", userID, err)
	}
	if got.JSON200 == nil {
		return acceleration.AgentUser{}, failure("reading "+userID, got.HTTPResponse, got.Body)
	}
	return *got.JSON200, nil
}

// UnlinkPhone removes the number linked to the app's user userID, if any. What was joined
// before stays joined. Only a backend may ask.
func (u *Users) UnlinkPhone(ctx context.Context, userID string) error {
	_, err := u.put(ctx, userID, nil)
	return err
}

func (u *Users) put(ctx context.Context, userID string, phone *string) (acceleration.AgentUser, error) {
	return u.update(ctx, userID, acceleration.UpdateUserJSONRequestBody{Phone: phone})
}

func (u *Users) update(ctx context.Context, userID string, body acceleration.UpdateUserJSONRequestBody) (acceleration.AgentUser, error) {
	if userID == "" {
		return acceleration.AgentUser{}, errors.New("client: linking a user needs a user id")
	}
	api, err := u.client.api()
	if err != nil {
		return acceleration.AgentUser{}, err
	}
	updated, err := api.UpdateUserWithResponse(ctx, userID, body)
	if err != nil {
		return acceleration.AgentUser{}, fmt.Errorf("client: linking %s: %w", userID, err)
	}
	if updated.JSON200 == nil {
		return acceleration.AgentUser{}, failure("linking "+userID, updated.HTTPResponse, updated.Body)
	}
	return *updated.JSON200, nil
}
