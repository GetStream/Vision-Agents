package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

var (
	errNoUserLinks = notConfigured("linking a user to a phone number is not available: no database configured")
	// errPhoneTaken names neither the number nor the other user: the caller sent the one and
	// may not learn the other.
	errPhoneTaken = APIError{Type: ErrorTypeConflict, Code: codePhoneTaken,
		Message: "the phone number is linked to another user of this app; unlink it from that user first"}
)

// AgentUserRequest is what the customer's backend knows about one of its users.
type AgentUserRequest struct {
	Phone               *string    `json:"phone" nullable:"true" example:"+15551234567" doc:"The user's phone number in E.164, with its plus. From then on, their SMS, WhatsApp messages, iMessages and calls from it share one omni-channel with their app chat, under each agent of the app: the agent's episode cards of one are read by the other. What the number said before the link is not the user's, unless include_phone_history or phone_history_since says otherwise: the user reads none of its earlier cards, whether the agent had episode_cards on at the link or turns them on later, and a session on the number's SMS, WhatsApp or iMessage thread is handed none of the thread's earlier messages. null unlinks the number the user had, and that number then reads none of the user's cards. A new number takes the place of the old one, which reads none of the user's cards from then on. A number linked to another user of the app is refused (409 phone_taken)."`
	IncludePhoneHistory *bool      `json:"include_phone_history,omitempty" doc:"How much of what the number said before this request the user reads. true: all of it, ever. false: none of it, from this request on. Left out with phone_history_since left out too: from the link on for a number new to the user, and what was stored for the number the user has. true with phone_history_since is refused (400); false with it, phone_history_since applies."`
	PhoneHistorySince   *time.Time `json:"phone_history_since,omitempty" doc:"The user reads what the number said from this time on, in RFC 3339, such as 2026-11-02T00:00:00Z: a conversation that started before it is left out whole. A time in the future is refused (400)."`
}

func (*AgentUserRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who one of the app's users is, as the app's backend knows it. phone is required: " +
		"a number, or null for none."
	return schema
}

// AgentUser is what the router holds about one of the app's users.
type AgentUser struct {
	UserID              string     `json:"user_id" doc:"The Stream user id, as the app's own token signs it."`
	Phone               *string    `json:"phone" nullable:"true" example:"+15551234567" doc:"The number linked to the user, in E.164; null when none is."`
	IncludePhoneHistory *bool      `json:"include_phone_history,omitempty" doc:"true when the user reads all of what the number ever said. Left out otherwise."`
	PhoneHistorySince   *time.Time `json:"phone_history_since,omitempty" doc:"From when on the user reads what the number said: the link's time, the time of a request with include_phone_history false, or the phone_history_since sent. Left out with include_phone_history true, or with no number."`
}

type getUserRequest struct {
	UserID string `path:"user_id" minLength:"1" doc:"The Stream user id the app chat knows the user by: the id the app's own token signs."`
}

type putUserRequest struct {
	UserID string            `path:"user_id" minLength:"1" doc:"The Stream user id the app chat knows the user by: the id the app's own token signs."`
	Body   *AgentUserRequest `required:"true"`
}

type putUserResponse struct {
	Body AgentUser
}

func (s *Server) registerUserPhones(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "updateUser",
		Method:      http.MethodPut,
		Path:        "/v1/agents/users/{user_id}",
		Summary:     "Link a user to a phone number",
		Description: "Says that a user of the app and a phone number are one person, so the person's " +
			"app chat and what they text and call from that number from then on share one omni-channel " +
			"with each agent of the app. What the number said before the link stays apart: under an " +
			"agent with episode_cards record or on when the link is made, the number's earlier conversations, and " +
			"one still going on, keep the omni-channel they had, and the user's app chat reads none of " +
			"their cards. Otherwise, where the person already had one of each, both point at the older " +
			"one, and the user still reads none of the number's cards from before the link once the agent " +
			"turns episode_cards on. include_phone_history and phone_history_since say how much of the number's " +
			"past the user reads instead: all of it, or what it said from a time on, in its cards and in the " +
			"sessions on its threads. No message moves. One number for each user, and one user for each " +
			"number, within the app: the same user id in another Stream app is another person. Sending the same number " +
			"again with neither field changes nothing; null unlinks.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The user as the router now holds it"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusConflict},
	}, s.putUser)
	huma.Register(api, huma.Operation{
		OperationID: "getUser",
		Method:      http.MethodGet,
		Path:        "/v1/agents/users/{user_id}",
		Summary:     "Get a user's phone link",
		Description: "The number linked to a user of the app, and how much of what it said before the link " +
			"the user reads. A user with no number linked has a null phone.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The user as the router holds it"}},
		Errors:    []int{http.StatusUnauthorized, http.StatusForbidden},
	}, s.getUser)
}

// getUser is one of the caller's users as their link is stored.
func (s *Server) getUser(ctx context.Context, request *getUserRequest) (*putUserResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoUserLinks
	}
	app, err := s.userApp(ctx, customerID)
	if err != nil {
		return nil, err
	}
	return s.storedUser(ctx, customerID, app, request.UserID)
}

// storedUser is a user as their link is stored: the number, and the boundary of its past as
// one of the two fields.
func (s *Server) storedUser(ctx context.Context, customerID string, app int64, userID string) (*putUserResponse, error) {
	answer := &putUserResponse{Body: AgentUser{UserID: userID}}
	phone, since, linked, err := s.store.LinkedPhone(ctx, customerID, app, userID)
	if err != nil || !linked {
		return answer, err
	}
	answer.Body.Phone = &phone
	if since == nil {
		all := true
		answer.Body.IncludePhoneHistory = &all
	} else {
		at := since.UTC()
		answer.Body.PhoneHistorySince = &at
	}
	return answer, nil
}

// userApp is the app a request about a user acts in: its pin, 0 for the deployment's own.
func (s *Server) userApp(ctx context.Context, customerID string) (int64, error) {
	if s.stream == nil {
		return 0, nil
	}
	return s.stream.Pin(ctx, customerID)
}

// phoneHistoryOf is how much of the number's past a PUT gives the user, nil for no word on it.
func phoneHistoryOf(request *AgentUserRequest) (*store.PhoneHistory, error) {
	include, since := request.IncludePhoneHistory, request.PhoneHistorySince
	if request.Phone == nil && (include != nil || since != nil) {
		return nil, invalidRequest("include_phone_history and phone_history_since need a phone")
	}
	if include != nil && *include && since != nil {
		return nil, invalidRequest("include_phone_history true and phone_history_since say two things; send one")
	}
	if since != nil && since.After(time.Now()) {
		return nil, invalidRequest("phone_history_since is in the future")
	}
	switch {
	case since != nil:
		return &store.PhoneHistory{Since: *since}, nil
	case include != nil:
		return &store.PhoneHistory{All: *include}, nil
	}
	return nil, nil
}

// putUser links one of the caller's users to a number, or unlinks them.
func (s *Server) putUser(ctx context.Context, request *putUserRequest) (*putUserResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoUserLinks
	}
	history, err := phoneHistoryOf(request.Body)
	if err != nil {
		return nil, err
	}
	app, err := s.userApp(ctx, customerID)
	if err != nil {
		return nil, err
	}
	if request.Body.Phone == nil {
		return &putUserResponse{Body: AgentUser{UserID: request.UserID}}, s.store.UnlinkUserPhone(ctx, customerID, app, request.UserID)
	}
	person, err := omnichannel.Phone(*request.Body.Phone)
	if err != nil {
		return nil, invalidRequest("phone is not a number in E.164, such as +15551234567")
	}
	err = s.store.LinkUserPhone(ctx, &store.UserPhone{CustomerID: customerID, StreamAppPK: app,
		UserID: request.UserID, Phone: person.Address, History: history})
	if errors.Is(err, store.ErrPhoneTaken) {
		return nil, errPhoneTaken
	}
	if err != nil {
		return nil, err
	}
	return s.storedUser(ctx, customerID, app, request.UserID)
}
