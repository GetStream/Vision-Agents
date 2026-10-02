package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/danielgtaylor/huma/v2"

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

// createGuestUser mints a guest so somebody can talk to an agent before they sign up.
type GuestUserRequest struct {
	Id     *string                 `json:"id,omitempty" doc:"A guest id to reuse, for somebody coming back. Omitted mints a new one. Asking for an id that is already a guest of this customer returns that guest with a fresh token rather than failing, because coming back is the same person."`
	Name   *string                 `json:"name,omitempty" doc:"What to call them, for a transcript a person reads later."`
	Custom *map[string]interface{} `json:"custom,omitempty"`
}

type GuestUser struct {
	Id        string                  `json:"id"`
	Token     string                  `json:"token" doc:"A Stream user token for this guest, which is what the chat and video SDKs connect with. It carries role guest, so an app that has turned guests off refuses it."`
	Name      *string                 `json:"name,omitempty"`
	Custom    *map[string]interface{} `json:"custom,omitempty"`
	ExpiresAt *time.Time              `json:"expires_at,omitempty" doc:"When the token stops working. A guest coming back after it asks for another."`
}

type ClaimGuestRequest struct {
	GuestId string `json:"guest_id"`
	UserId  string `json:"user_id" doc:"The account the guest turned out to be."`
}

type ClaimGuestResult struct {
	GuestId       string `json:"guest_id"`
	UserId        string `json:"user_id"`
	SessionsMoved int    `json:"sessions_moved" doc:"How many conversations moved onto the account."`
}

type createGuestUserRequest struct {
	Body *GuestUserRequest
}

type guestUserResponse struct {
	Body GuestUser
}

type claimGuestUserRequest struct {
	Body ClaimGuestRequest
}

type claimGuestResultResponse struct {
	Body ClaimGuestResult
}

func (s *Server) registerGuests(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "createGuestUser",
		Method:      http.MethodPost,
		Path:        "/v1/agents/guests",
		Summary:     "Mint a guest so somebody can talk to an agent before signing up",
		Description: "Returns a user id and a Stream token with role guest, which is what the chat " +
			"and video SDKs connect with. From the router's point of view a guest is an " +
			"ordinary end user whose name is worth less: their sessions are their own and " +
			"nobody else's, but a guest id is not evidence of who anybody is, so a guest " +
			"cannot be handed another guest's conversations by naming their id.\n" +
			"Open to a page on purpose. A guest that a backend had to mint is a guest every " +
			"anonymous visitor costs a round trip through the customer's own servers, which " +
			"is exactly the integration this is meant to remove. An app that has turned " +
			"guests off refuses it.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The guest and a token for them"},
			"403": errorResponse("This app does not admit guests"),
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.createGuestUser)
	huma.Register(api, huma.Operation{
		OperationID: "claimGuestUser",
		Method:      http.MethodPost,
		Path:        "/v1/agents/guests/claim",
		Summary:     "Move a guest's conversations onto the account they turned out to be",
		Description: "For somebody who talked to an agent and then signed up. Their sessions are " +
			"rewritten to the real user and the guest is marked claimed, in one transaction: " +
			"a guest marked claimed whose sessions still say the guest owns them is a person " +
			"who signed up and lost their history, and sessions moved without the guest " +
			"being marked is a guest that can be claimed again, by somebody else.\n" +
			"Server-side only, and the one operation here that most needs to be. Only the " +
			"customer's own backend knows that a given guest is a given account -- it is the " +
			"thing that just authenticated them. A page allowed to ask this could claim " +
			"anybody's conversations by guessing a guest id, which is the whole attack.\n" +
			"A guest already claimed is refused rather than moved again.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The conversations were moved"},
			"409": errorResponse("That guest was already claimed"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.claimGuestUser)
}

// Open to a page on purpose. A guest a backend had to mint is a guest that costs every
// anonymous visitor a round trip through the customer's own servers, which is exactly the
// integration this is meant to remove. What keeps it safe is that a guest id is not evidence
// of who anybody is: their sessions are their own, and the kind recorded beside the name is
// what stops one guest reaching another's by naming their id.
func (s *Server) createGuestUser(ctx context.Context, request *createGuestUserRequest) (*guestUserResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.streamKey == "" || s.streamSecret == "" {
		return nil, huma.Error400BadRequest(noStreamKeys)
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
			return nil, huma.Error403Forbidden("this app does not admit guests")
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
			return nil, huma.Error400BadRequest(
				"that id is not a guest of this app, so a token cannot be minted for it")
		case held.ClaimedBy != "":
			return nil, huma.Error400BadRequest(
				"that guest has been claimed, so they have an account to sign in with")
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
	return &guestUserResponse{Body: rendered}, nil
}

// claimGuestUser moves a guest's conversations onto the account they turned out to be.
//
// Server-side only, and the one operation here that most needs to be: only the customer's
// own backend knows that a given guest is a given account, because it is the thing that just
// authenticated them. A page allowed to ask this could claim anybody's conversations by
// guessing a guest id, which is the whole attack.
func (s *Server) claimGuestUser(ctx context.Context, request *claimGuestUserRequest) (*claimGuestResultResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error404NotFound("this deployment records no guests, so there is nothing to claim")
	}

	guestID := strings.TrimSpace(request.Body.GuestId)
	userID := strings.TrimSpace(request.Body.UserId)
	if guestID == "" || userID == "" {
		return nil, huma.Error400BadRequest("a guest id and a user id are required")
	}

	guest, err := s.store.Guest(ctx, customerID, guestID)
	if err != nil {
		return nil, huma.Error404NotFound("no such guest")
	}
	if guest.ClaimedBy != "" {
		// A second claim naming the same account is the same claim arriving twice, which is
		// not a conflict: a retried request must not read as somebody else's.
		if guest.ClaimedBy == userID {
			return &claimGuestResultResponse{Body: ClaimGuestResult{
				GuestId: guestID, UserId: userID, SessionsMoved: 0,
			}}, nil
		}
		return nil, huma.Error409Conflict("that guest was already claimed")
	}

	moved, err := s.store.ClaimGuest(ctx, customerID, guestID, userID)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	// The channels come after the rows, and a failure here is logged rather than returned.
	// The conversations have already moved, which is the part that matters; a membership that
	// could not be added leaves the person able to find the conversation and unable to read
	// it, and answering with an error would have them try again against rows already moved.
	if err := s.addToGuestChannels(ctx, customerID, guestID, userID); err != nil {
		s.logger.Error("claimed a guest but could not add the account to their channels",
			"guest", guestID, "user", userID, "error", err)
	}

	return &claimGuestResultResponse{Body: ClaimGuestResult{
		GuestId: guestID, UserId: userID, SessionsMoved: int(moved),
	}}, nil
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
