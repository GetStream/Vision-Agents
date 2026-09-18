package api

import (
	"context"
	"errors"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// guestTokenValidity is how long a guest's token lasts.
//
// Longer than a listener's, because a guest is a person having a conversation rather than a
// process reading one: somebody who asked a question, read the answer, and came back after
// lunch should not find themselves logged out of a conversation they can see on the screen.
// Short enough that a token left in a stale browser tab stops working eventually.
const guestTokenValidity = 24 * time.Hour

// guestPrefix marks the ids this mints, so a guest is recognisable as one in a transcript
// and in the Stream dashboard without looking the row up.
const guestPrefix = "guest-"

// guestRole is what Stream is told these users are. It is the role an app turns off when it
// does not admit guests, so naming it here is what makes that switch mean anything.
const guestRole = "guest"

// CreateGuestUser mints a guest so somebody can talk to an agent before they sign up.
//
// Open to a page on purpose. A guest a backend had to mint is a guest that costs every
// anonymous visitor a round trip through the customer's own servers, which is exactly the
// integration this is meant to remove. What keeps it safe is that a guest id is not evidence
// of who anybody is: their sessions are their own, and the kind recorded beside the name is
// what stops one guest reaching another's by naming their id.
func (s *Server) CreateGuestUser(ctx context.Context, request CreateGuestUserRequestObject) (CreateGuestUserResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateGuestUser401JSONResponse{missingCustomer()}, nil
	}
	if s.streamKey == "" || s.streamSecret == "" {
		return CreateGuestUser400JSONResponse{badRequest(noStreamKeys)}, nil
	}

	body := GuestUserRequest{}
	if request.Body != nil {
		body = *request.Body
	}

	// Asked before the guest is minted rather than after. A token handed out by an app that
	// does not admit guests is one the next request rejects, which reads as a broken
	// deployment rather than as a setting somebody chose.
	if s.store != nil {
		settings, err := s.store.AppSettingsFor(ctx, customerID)
		if err != nil {
			return nil, err
		}
		if !settings.GuestAllowed() {
			return CreateGuestUser403Response{}, nil
		}
	}

	userID := strings.TrimSpace(value(body.Id))
	if userID == "" {
		userID = guestPrefix + store.NewID()
	}
	name := value(body.Name)
	if name == "" {
		name = "Guest"
	}

	// A guest asking for an id that is already somebody else's claimed account would be a
	// way to be handed their conversations, so only an id that is a guest of this customer,
	// or one nobody has used, is reusable.
	if s.store != nil && value(body.Id) != "" {
		held, err := s.store.Guest(ctx, customerID, userID)
		switch {
		case err != nil:
			// Not a guest of this customer. It may be nobody at all, which is fine, or it
			// may be a real user, which is not something this can tell apart -- so the
			// name is refused rather than guessed at.
			return CreateGuestUser400JSONResponse{badRequest(
				"that id is not a guest of this app, so a token cannot be minted for it")}, nil
		case held.ClaimedBy != "":
			return CreateGuestUser400JSONResponse{badRequest(
				"that guest has been claimed, so they have an account to sign in with")}, nil
		}
	}

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return nil, err
	}

	role := guestRole
	if _, err := client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{
		Users: map[string]getstream.UserRequest{
			userID: {ID: userID, Name: &name, Role: &role, Custom: value(body.Custom)},
		},
	}); err != nil {
		return nil, err
	}

	expiresAt := time.Now().UTC().Add(guestTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(guestTokenValidity))
	if err != nil {
		return nil, err
	}

	// Recorded after the token is minted, so a guest that could not be given one leaves no
	// row behind. Without a store guests still work; they simply cannot be claimed later,
	// because there is nothing that says which ids were ever guests of this app.
	if s.store != nil {
		guest := &store.GuestUser{
			ID: userID, CustomerID: customerID, Name: name, Custom: value(body.Custom),
		}
		if err := s.store.RecordGuest(ctx, guest); err != nil {
			return nil, err
		}
	}

	rendered := GuestUser{Id: userID, Token: token, Name: &name, ExpiresAt: &expiresAt}
	if body.Custom != nil {
		rendered.Custom = body.Custom
	}
	return CreateGuestUser201JSONResponse(rendered), nil
}

// ClaimGuestUser moves a guest's conversations onto the account they turned out to be.
//
// Server-side only, and the one operation here that most needs to be: only the customer's
// own backend knows that a given guest is a given account, because it is the thing that just
// authenticated them. A page allowed to ask this could claim anybody's conversations by
// guessing a guest id, which is the whole attack.
func (s *Server) ClaimGuestUser(ctx context.Context, request ClaimGuestUserRequestObject) (ClaimGuestUserResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ClaimGuestUser401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ClaimGuestUser404JSONResponse{NotFoundJSONResponse{
			Error: "this deployment records no guests, so there is nothing to claim"}}, nil
	}
	if request.Body == nil {
		return ClaimGuestUser400JSONResponse{badRequest("a request body is required")}, nil
	}

	guestID := strings.TrimSpace(request.Body.GuestId)
	userID := strings.TrimSpace(request.Body.UserId)
	if guestID == "" || userID == "" {
		return ClaimGuestUser400JSONResponse{badRequest("a guest id and a user id are required")}, nil
	}

	guest, err := s.store.Guest(ctx, customerID, guestID)
	if err != nil {
		return ClaimGuestUser404JSONResponse{NotFoundJSONResponse{
			Error: "no such guest"}}, nil
	}
	if guest.ClaimedBy != "" {
		// A second claim naming the same account is the same claim arriving twice, which is
		// not a conflict: a retried request must not read as somebody else's.
		if guest.ClaimedBy == userID {
			return ClaimGuestUser200JSONResponse{
				GuestId: guestID, UserId: userID, SessionsMoved: 0,
			}, nil
		}
		return ClaimGuestUser409JSONResponse{Error: "that guest was already claimed"}, nil
	}

	moved, err := s.store.ClaimGuest(ctx, customerID, guestID, userID)
	if err != nil {
		return ClaimGuestUser400JSONResponse{badRequest(err.Error())}, nil
	}

	// The channels come after the rows, and a failure here is logged rather than returned.
	// The conversations have already moved, which is the part that matters; a membership that
	// could not be added leaves the person able to find the conversation and unable to read
	// it, and answering with an error would have them try again against rows already moved.
	if err := s.addToGuestChannels(ctx, customerID, guestID, userID); err != nil {
		s.logger.Error("claimed a guest but could not add the account to their channels",
			"guest", guestID, "user", userID, "error", err)
	}

	return ClaimGuestUser200JSONResponse{
		GuestId: guestID, UserId: userID, SessionsMoved: int(moved),
	}, nil
}

// addToGuestChannels puts the real account into the transcripts the guest was talking in, so
// the conversations a claim just moved are readable by the person they moved to.
func (s *Server) addToGuestChannels(ctx context.Context, customerID, guestID, userID string) error {
	if s.streamKey == "" || s.streamSecret == "" {
		return nil
	}

	// The sessions have already been rewritten, so they are found by the account rather than
	// by the guest. Only the ones that kept a transcript have a channel to join.
	moved, err := s.store.QuerySessions(ctx, customerID, store.SessionFilter{
		UserID: userID, Limit: 200,
	})
	if err != nil {
		return err
	}

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return err
	}

	var failures []error
	for _, one := range moved {
		channel := transcriptChannel(one)
		if channel == "" {
			continue
		}
		_, err := client.Chat().UpdateChannel(ctx, chatlog.ChannelType, channel,
			&getstream.UpdateChannelRequest{
				AddMembers: []getstream.ChannelMemberRequest{{UserID: userID}},
			})
		if err != nil {
			failures = append(failures, err)
		}
	}
	if len(failures) > 0 {
		return errors.Join(failures...)
	}
	return nil
}

// transcriptChannel is the Stream Chat channel a session's words are in, empty for one that
// kept none.
//
// Two kinds of session write into the same channel type under different ids: a conversation
// held in writing is keyed by the id of the channel it opened, and a call's transcript is
// keyed by the agent taking it. Picking the wrong one here would add somebody to a channel
// that does not exist, which reports as success and reads as a claim that did nothing.
func transcriptChannel(one store.AgentSession) string {
	if one.ConversationID != "" {
		return strings.TrimPrefix(one.ConversationID, chatlog.ChannelType+":")
	}
	return one.AgentID
}
