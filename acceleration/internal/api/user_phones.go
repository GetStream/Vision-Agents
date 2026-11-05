package api

import (
	"context"
	"errors"
	"net/http"

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
	Phone *string `json:"phone" nullable:"true" example:"+15551234567" doc:"The user's phone number in E.164, with its plus. From then on, their SMS, WhatsApp messages, iMessages and calls from it share one omni-channel with their app chat, under each agent of the app: the agent's episode cards of one are read by the other. What the number said before the link is not the user's: the user reads none of its earlier cards, whether the agent had episode_cards on at the link or turns them on later. null unlinks the number the user had, and that number then reads none of the user's cards. A new number takes the place of the old one, which reads none of the user's cards from then on. A number linked to another user of the app is refused (409 phone_taken)."`
}

func (*AgentUserRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who one of the app's users is, as the app's backend knows it. phone is required: " +
		"a number, or null for none."
	return schema
}

// AgentUser is what the router holds about one of the app's users.
type AgentUser struct {
	UserID string  `json:"user_id" doc:"The Stream user id, as the app's own token signs it."`
	Phone  *string `json:"phone" nullable:"true" example:"+15551234567" doc:"The number linked to the user, in E.164; null when none is."`
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
			"agent with episode_cards on when the link is made, the number's earlier conversations, and " +
			"one still going on, keep the omni-channel they had, and the user's app chat reads none of " +
			"their cards. Otherwise, where the person already had one of each, both point at the older " +
			"one, and the user still reads none of the number's cards from before the link once the agent " +
			"turns episode_cards on. No message moves. One number for each user, and one user for each " +
			"number, within the app: the same user id in another Stream app is another person. Sending the same number " +
			"again changes nothing; null unlinks.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The user as the router now holds it"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusConflict},
	}, s.putUser)
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
	var app int64
	if s.stream != nil {
		var err error
		if app, err = s.stream.Pin(ctx, customerID); err != nil {
			return nil, err
		}
	}
	answer := &putUserResponse{Body: AgentUser{UserID: request.UserID}}
	if request.Body.Phone == nil {
		return answer, s.store.UnlinkUserPhone(ctx, customerID, app, request.UserID)
	}
	person, err := omnichannel.Phone(*request.Body.Phone)
	if err != nil {
		return nil, invalidRequest("phone is not a number in E.164, such as +15551234567")
	}
	err = s.store.LinkUserPhone(ctx, &store.UserPhone{CustomerID: customerID, StreamAppPK: app,
		UserID: request.UserID, Phone: person.Address})
	if errors.Is(err, store.ErrPhoneTaken) {
		return nil, errPhoneTaken
	}
	if err != nil {
		return nil, err
	}
	answer.Body.Phone = &person.Address
	return answer, nil
}
