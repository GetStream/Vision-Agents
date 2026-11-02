package api

import (
	"cmp"
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
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

// createGuestUser mints a guest so somebody can talk to an agent before they sign up, or
// gives a guest coming back a fresh token.
//
// Open to a page on purpose. A guest a backend had to mint is a guest that costs every
// anonymous visitor a round trip through the customer's own servers, which is exactly the
// integration this is meant to remove. What keeps it safe is that a guest id is not evidence
// of who anybody is: it is the author of the guest's messages and a member of their channels,
// so whoever shared a channel with them knows it. A token for an existing guest is only handed
// to a caller that proves it is that guest (returningGuest); any other caller naming an id is
// minted a new guest, exactly as if it had named none.
func (s *Server) createGuestUser(ctx context.Context, request *createGuestUserRequest) (*createGuestUserResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	bound, mintable, err := s.streamFor(ctx, customerID)
	if elsewhere(err) {
		return nil, invalidRequest("this app's Stream app is disconnected, so no guest can be made in it")
	}
	if err != nil {
		return nil, err
	}
	if !mintable {
		return nil, errNoStreamKeys
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
			return nil, forbidden("this app does not admit guests")
		}
	}

	if requested := strings.TrimSpace(value(body.Id)); requested != "" {
		returning, err := s.returningGuest(ctx, customerID, bound, requested, body)
		if err != nil || returning != nil {
			return returning, err
		}
	}
	return s.newGuest(ctx, customerID, bound, body)
}

// newGuest mints a guest nobody has been before, and a refresh secret to come back with.
func (s *Server) newGuest(ctx context.Context, customerID string, bound streamapp.Bound, body GuestUserRequest) (*createGuestUserResponse, error) {
	userID := guestPrefix + store.NewID()
	name := value(body.Name)
	if name == "" {
		name = "Guest"
	}

	if err := s.updateGuestProfile(ctx, bound, userID, name, value(body.Custom)); err != nil {
		return nil, err
	}
	token, expiresAt, err := s.mintGuestToken(ctx, bound, userID)
	if err != nil {
		return nil, err
	}

	// Recorded after the token is minted, so a guest that could not be given one leaves no
	// row behind. Without a store guests still work; they simply cannot be claimed later or
	// come back once their token expires, because there is nothing that says which ids were
	// ever guests of this app.
	rendered := GuestUser{Id: userID, Token: token, Name: &name, ExpiresAt: &expiresAt, Custom: body.Custom}
	if s.store != nil {
		secret, hash, err := store.NewGuestSecret()
		if err != nil {
			return nil, err
		}
		guest := &store.User{
			ID: userID, CustomerID: customerID, Name: name, Custom: value(body.Custom), RefreshSecretHash: hash,
		}
		if err := s.store.RecordGuest(ctx, guest); err != nil {
			return nil, err
		}
		rendered.RefreshSecret = &secret
	}
	return &createGuestUserResponse{Body: rendered}, nil
}

// returningGuest gives an existing guest a fresh token, when the caller proves it may have
// one. It answers nil, with no error, for a caller that proves nothing, which is then minted a
// new guest instead: refusing would tell whoever is guessing which ids are guests and which
// have signed up, and would strand a device holding a guest from before secrets existed.
//
// Three things prove it:
//   - the app's backend, which can mint a token for anyone anyway;
//   - the guest's own token, still valid, naming this id. The token the router mints carries
//     no role claim, so it reads as KindAuthenticated rather than KindGuest; a caller that
//     only named an id in a header is KindAnonymous and proves nothing;
//   - the refresh secret the guest was last handed, which is spent by being used.
func (s *Server) returningGuest(ctx context.Context, customerID string, bound streamapp.Bound, requested string, body GuestUserRequest) (*createGuestUserResponse, error) {
	kind := KindFrom(ctx)
	server := kind == auth.KindServer
	self := (kind == auth.KindAuthenticated || kind == auth.KindGuest) && CallerFrom(ctx).UserID == requested
	if s.store == nil {
		// Nothing records which ids are guests, so nothing can say this one is not
		// somebody's real account.
		if server {
			return nil, invalidRequest("this deployment records no guests, so a guest id cannot be reused")
		}
		return nil, nil
	}

	if !server && !self {
		return s.redeemGuestSecret(ctx, customerID, bound, requested, strings.TrimSpace(value(body.RefreshSecret)))
	}

	held, err := s.store.Guest(ctx, customerID, requested)
	switch {
	case err != nil || (!held.Minted && !s.ownApp(customerID, bound.Identity.StreamApp)):
		// Not a guest this customer was handed. It may be nobody at all, or a real user, which
		// is not something this can tell apart -- so the name is refused rather than guessed
		// at. In an app shared with other customers, a guest only recorded from a token the
		// app signed itself is no proof either: the app may sign any id. In its own app every
		// id is the customer's anyway.
		return nil, invalidRequest("that id is not a guest of this app, so a token cannot be minted for it")
	case held.ClaimedBy != "":
		return nil, invalidRequest("that guest has been claimed, so they have an account to sign in with")
	}

	// Only the backend may rename a guest here: a guest's own token is proof of who they are,
	// not a profile editor, and the name is what a transcript shows to the people reading it.
	name, custom := held.Name, held.Custom
	if server {
		name = cmp.Or(value(body.Name), held.Name, "Guest")
		if body.Custom != nil {
			custom = *body.Custom
		}
		if err := s.updateGuestProfile(ctx, bound, requested, name, custom); err != nil {
			return nil, err
		}
		if err := s.store.SetGuestProfile(ctx, customerID, requested, name, custom); err != nil {
			return nil, err
		}
	}

	token, expiresAt, err := s.mintGuestToken(ctx, bound, requested)
	if err != nil {
		return nil, err
	}
	secret, hash, err := store.NewGuestSecret()
	if err != nil {
		return nil, err
	}

	// The backend hands out a new secret, replacing the old: it is what an app does for a
	// device it has decided is this guest. The guest's own token only gives one to a guest
	// holding none, such as one minted before secrets existed; one holding a secret keeps it.
	issued := true
	if server {
		err = s.store.SetGuestSecret(ctx, customerID, requested, hash)
	} else {
		issued, err = s.store.GiveGuestASecret(ctx, customerID, requested, hash)
	}
	if err != nil {
		return nil, err
	}
	if !issued {
		secret = ""
	}
	return returned(requested, token, name, custom, expiresAt, secret), nil
}

// redeemGuestSecret gives a fresh token to the guest whose refresh secret this is, and the
// next secret in its place. A secret that is wrong, spent or missing answers nil, so the caller
// is minted a new guest instead.
//
// The token is minted before the secret is spent, so a failure to mint, which can be a call to
// Stream's gateway, leaves the device holding a secret that still works. Spending it is one
// conditional update, so of two requests replaying it only one is answered; the other's token is
// never handed out. Every request carrying a secret costs a mint, right or wrong, which says
// nothing about whether the id is a guest's.
func (s *Server) redeemGuestSecret(ctx context.Context, customerID string, bound streamapp.Bound, requested, secret string) (*createGuestUserResponse, error) {
	if secret == "" {
		return nil, nil
	}
	next, hash, err := store.NewGuestSecret()
	if err != nil {
		return nil, err
	}
	token, expiresAt, err := s.mintGuestToken(ctx, bound, requested)
	if err != nil {
		return nil, err
	}
	held, err := s.store.RotateGuestSecret(ctx, customerID, requested, store.GuestSecretHash(secret), hash)
	if errors.Is(err, store.ErrGuestSecretRefused) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return returned(requested, token, held.Name, held.Custom, expiresAt, next), nil
}

// updateGuestProfile tells Stream who a guest is, with the role an app turns off when it does
// not admit them.
func (s *Server) updateGuestProfile(ctx context.Context, bound streamapp.Bound, userID, name string, custom map[string]any) error {
	role := guestRole
	_, err := bound.Client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{
		Users: map[string]getstream.UserRequest{
			userID: {ID: userID, Name: &name, Role: &role, Custom: custom},
		},
	})
	return stack.Wrap(err)
}

// mintGuestToken mints a guest's Stream token and says when it stops working.
func (s *Server) mintGuestToken(ctx context.Context, bound streamapp.Bound, userID string) (string, time.Time, error) {
	expiresAt := time.Now().UTC().Add(guestTokenValidity)
	token, _, err := s.stream.MintUserToken(ctx, bound, requestKeyFrom(ctx), userID, guestTokenValidity)
	if err != nil {
		return "", time.Time{}, stack.Wrap(err)
	}
	return token, expiresAt, nil
}

// returned renders a guest coming back, with no refresh_secret when none was issued.
func returned(userID, token, name string, custom map[string]any, expiresAt time.Time, secret string) *createGuestUserResponse {
	rendered := GuestUser{Id: userID, Token: token, Name: &name, ExpiresAt: &expiresAt}
	if secret != "" {
		rendered.RefreshSecret = &secret
	}
	if len(custom) > 0 {
		rendered.Custom = &custom
	}
	return &createGuestUserResponse{Body: rendered}
}

// claimGuestUser moves a guest's conversations onto the account they turned out to be.
//
// Server-side only, and the one operation here that most needs to be: only the customer's
// own backend knows that a given guest is a given account, because it is the thing that just
// authenticated them. A page allowed to ask this could claim anybody's conversations by
// guessing a guest id, which is the whole attack.
func (s *Server) claimGuestUser(ctx context.Context, request *claimGuestUserRequest) (*claimGuestUserResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, notFound("this deployment records no guests, so there is nothing to claim")
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	guestID := strings.TrimSpace(request.Body.GuestId)
	userID := strings.TrimSpace(request.Body.UserId)
	if guestID == "" || userID == "" {
		return nil, invalidRequest("a guest id and a user id are required")
	}

	guest, err := s.store.Guest(ctx, customerID, guestID)
	if err != nil {
		return nil, notFound("no such guest")
	}
	if guest.ClaimedBy != "" {
		// A second claim naming the same account is the same claim arriving twice, which is
		// not a conflict: a retried request must not read as somebody else's.
		if guest.ClaimedBy == userID {
			return &claimGuestUserResponse{Body: ClaimGuestResult{GuestId: guestID, UserId: userID, SessionsMoved: 0}}, nil
		}
		return nil, conflict("that guest was already claimed")
	}

	moved, err := s.store.ClaimGuest(ctx, customerID, guestID, userID)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	// The guest's row has changed under every replica holding it, so the caches are told.
	// The account they turned out to be is not recorded here: the next request they make
	// carries their own token, and that is what says which kind of user they are.
	if s.users != nil {
		s.users.Forget(ctx, customerID, guestID)
	}

	// The channels come after the rows, and a failure here is logged rather than returned.
	// The conversations have already moved, which is the part that matters; a membership that
	// could not be added leaves the person able to find the conversation and unable to read
	// it, and answering with an error would have them try again against rows already moved.
	if err := s.addToGuestChannels(ctx, customerID, guestID, userID); err != nil {
		s.logger.Error("claimed a guest but could not add the account to their channels",
			"guest", guestID, "user", userID, "error", err)
	}

	return &claimGuestUserResponse{Body: ClaimGuestResult{GuestId: guestID, UserId: userID, SessionsMoved: int(moved)}}, nil
}

// addToGuestChannels puts the real account into the transcripts the guest was talking in, so
// the conversations a claim just moved are readable by the person they moved to.
func (s *Server) addToGuestChannels(ctx context.Context, customerID, guestID, userID string) error {
	if s.stream == nil {
		return nil
	}

	// The sessions have already been rewritten, so they are found by the account rather than
	// by the guest. Only the ones that kept a transcript have a channel to join, and only the
	// ones the account reaches: an anonymous caller that went by its id is somebody else.
	moved, err := s.store.QuerySessions(ctx, customerID, store.SessionFilter{
		UserID: userID, CallerKinds: []string{string(auth.KindAuthenticated), string(auth.KindServer)}, Limit: 200,
	})
	if err != nil {
		return err
	}

	var failures []error
	for _, one := range moved {
		channel := transcriptChannel(one)
		if channel == "" {
			continue
		}
		// Each conversation is joined in the app it was held in, which for a guest who
		// talked before their app had an identity of its own is the deployment's.
		bound, err := s.streamForApp(ctx, customerID, one.StreamAppPK, false)
		if _, refused := refusal(err, "", ""); refused {
			continue
		}
		if err != nil {
			failures = append(failures, err)
			continue
		}
		if err := refuseChannelOutside(ctx, bound.Client, customerID, channel); err != nil {
			failures = append(failures, err)
			continue
		}
		_, err = bound.Client.Chat().UpdateChannel(ctx, chatlog.ChannelType, channel,
			&getstream.UpdateChannelRequest{
				AddMembers: []getstream.ChannelMemberRequest{{UserID: userID}},
			})
		if err != nil {
			failures = append(failures, err)
		}
	}
	if len(failures) > 0 {
		return stack.Wrap(errors.Join(failures...))
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

// registerGuests declares the operations served in guests.go.
func (s *Server) registerGuests(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "createGuestUser",
		Method:      http.MethodPost,
		Path:        "/v1/agents/guests",
		Summary:     "Mint a guest so somebody can talk to an agent before signing up",
		Description: "Returns a user id and a Stream token with role guest, which is what the chat and video " +
			"SDKs connect with, and a refresh_secret to come back with once it has expired. From the " +
			"router's point of view a guest is an ordinary end user whose name is worth less: their " +
			"sessions are their own and nobody else's, but a guest id is not evidence of who anybody " +
			"is, so a guest cannot be handed another guest's conversations by naming their id. A token " +
			"for an existing guest goes only to the app's backend, to the guest's own still-valid " +
			"token, or to the refresh_secret the guest was last handed; any other caller naming an id " +
			"is minted a new guest.\n" +
			"Open to a page on purpose. A guest that a backend had to mint is a guest every " +
			"anonymous visitor costs a round trip through the customer's own servers, which is " +
			"exactly the integration this is meant to remove. An app that has turned guests off " +
			"refuses it.",
		Extensions:    map[string]any{clientAccessibleExtension: true},
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The guest and a token for them"},
			"403": errorResponse("This app does not admit guests"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized},
	}, s.createGuestUser)
	huma.Register(api, huma.Operation{
		OperationID: "claimGuestUser",
		Method:      http.MethodPost,
		Path:        "/v1/agents/guests/claim",
		Summary:     "Move a guest's conversations onto the account they turned out to be",
		Description: "For somebody who talked to an agent and then signed up. Their sessions are rewritten to " +
			"the real user and the guest is marked claimed, in one transaction: a guest marked " +
			"claimed whose sessions still say the guest owns them is a person who signed up and lost " +
			"their history, and sessions moved without the guest being marked is a guest that can be " +
			"claimed again, by somebody else.\n" +
			"Server-side only, and the one operation here that most needs to be. Only the customer's " +
			"own backend knows that a given guest is a given account -- it is the thing that just " +
			"authenticated them. A page allowed to ask this could claim anybody's conversations by " +
			"guessing a guest id, which is the whole attack.\n" +
			"A guest already claimed is refused rather than moved again.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The conversations were moved"},
			"409": errorResponse("That guest was already claimed"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.claimGuestUser)
}

type createGuestUserRequest struct {
	Body *GuestUserRequest
}

type createGuestUserResponse struct {
	Body GuestUser
}

type claimGuestUserRequest struct {
	Body *ClaimGuestRequest `required:"true"`
}

type claimGuestUserResponse struct {
	Body ClaimGuestResult
}

// ClaimGuestRequest is the ClaimGuestRequest schema.
type ClaimGuestRequest struct {
	GuestId string `json:"guest_id"`
	UserId  string `json:"user_id" doc:"The account the guest turned out to be."`
}

// ClaimGuestResult is the ClaimGuestResult schema.
type ClaimGuestResult struct {
	GuestId       string `json:"guest_id"`
	SessionsMoved int    `json:"sessions_moved" doc:"How many conversations moved onto the account."`
	UserId        string `json:"user_id"`
}

func (*ClaimGuestResult) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["sessions_moved"].Format = ""
	return schema
}

// GuestUser is the GuestUser schema.
type GuestUser struct {
	Custom    *map[string]interface{} `json:"custom,omitempty"`
	ExpiresAt *time.Time              `json:"expires_at,omitempty" doc:"When the token stops working. A guest coming back after it asks for another."`
	Id        string                  `json:"id"`
	Name      *string                 `json:"name,omitempty"`
	// Sent only to whoever proved they are this guest; never readable afterwards.
	RefreshSecret *string `json:"refresh_secret,omitempty" doc:"What this guest sends back, with their id, to be given a fresh token once this one has expired. Keep it where the token is kept. It works once: every answer to it carries the next one, so store it each time. Absent where the deployment records no guests, and from a refresh made with the guest's own token while they already hold one, which keeps working."`
	Token         string  `json:"token" doc:"A Stream user token for this guest, which is what the chat and video SDKs connect with. It carries role guest, so an app that has turned guests off refuses it."`
}

// GuestUserRequest is the GuestUserRequest schema.
type GuestUserRequest struct {
	Custom        *map[string]interface{} `json:"custom,omitempty"`
	Id            *string                 `json:"id,omitempty" doc:"A guest coming back, to be given a fresh token as the same guest. Only honoured for a caller that proves it is that guest: the app's backend, the guest's own still-valid token, or the refresh_secret they were last handed. Any other caller is minted a new guest, as if no id had been sent, because a guest id is no secret: it is the author of their messages. Omitted mints a new guest."`
	Name          *string                 `json:"name,omitempty" doc:"What to call a new guest, for a transcript a person reads later. A guest coming back keeps their name unless the app's backend renames them."`
	RefreshSecret *string                 `json:"refresh_secret,omitempty" doc:"The refresh_secret this guest was last handed, sent with their id once their token has expired. Spent by being used. One that is wrong or already spent mints a new guest instead."`
}
