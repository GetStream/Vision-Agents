package client

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// Guest is a guest as it was minted: the id, and the token that proves it.
type Guest = acceleration.GuestUser

// GuestOptions is who a guest is, as far as anybody knows yet.
type GuestOptions struct {
	// ID reuses a guest, for somebody coming back. Empty mints a new one.
	ID string
	// Name is what to call them, for a transcript a person reads later. Empty is the
	// backend's own default, which is honest about not knowing.
	Name string
	// Custom is anything of the caller's own to keep against them.
	Custom map[string]any
}

// GuestUser mints a guest so somebody can talk to an agent before they sign up.
//
// Nothing is remembered here. A Go process is a backend, and a backend handling two visitors
// that remembered one guest would hand them each other's conversations; remembering which
// visitor is which is the caller's job, because only the caller knows what a visitor is. The
// browser SDKs do keep one, in a cookie, because there a page is one person.
//
// Asking for an ID that is already a guest returns that guest with a fresh token rather than
// failing, because coming back is the same person.
func (c *Client) GuestUser(ctx context.Context, options GuestOptions) (*Guest, error) {
	api, err := c.api()
	if err != nil {
		return nil, err
	}

	request := acceleration.GuestUserRequest{
		Id:   pointer(options.ID),
		Name: pointer(options.Name),
	}
	if len(options.Custom) > 0 {
		custom := options.Custom
		request.Custom = &custom
	}

	minted, err := api.CreateGuestUserWithResponse(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("client: minting a guest: %w", err)
	}
	if minted.JSON201 == nil {
		return nil, failure("minting a guest", minted.Status(), minted.JSON400, minted.JSON401)
	}
	return minted.JSON201, nil
}

// AsGuest is a client acting for a guest, which is what its conversations belong to.
//
// The same thing SetUser does, said the way the guest paths say it: a guest is a user with a
// token, and everything after this point treats them as one.
func (c *Client) AsGuest(guest *Guest) (*Client, error) {
	if guest == nil {
		return nil, errors.New("client: there is no guest to act for")
	}
	return c.SetUser(guest.Id, guest.Token)
}

// ClaimGuestUser moves a guest's conversations onto the account they turned out to be.
//
// Server side only, and the one thing here that most needs to be: only the app's own backend
// knows that a given guest is a given account, because it is the thing that just
// authenticated them. A page able to ask this could claim anybody's conversations by guessing
// a guest id, so the router refuses it from one.
func (c *Client) ClaimGuestUser(ctx context.Context, guestID, userID string) (*acceleration.ClaimGuestResult, error) {
	if guestID == "" || userID == "" {
		return nil, errors.New("client: claiming a guest needs the guest and the account")
	}

	api, err := c.api()
	if err != nil {
		return nil, err
	}

	claimed, err := api.ClaimGuestUserWithResponse(ctx,
		acceleration.ClaimGuestRequest{GuestId: guestID, UserId: userID})
	if err != nil {
		return nil, fmt.Errorf("client: claiming the guest %s: %w", guestID, err)
	}
	if claimed.JSON200 == nil {
		return nil, failure("claiming the guest "+guestID, claimed.Status(),
			claimed.JSON400, claimed.JSON401, claimed.JSON403, claimed.JSON404, claimed.JSON409)
	}
	return claimed.JSON200, nil
}
