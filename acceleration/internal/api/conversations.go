package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

func (s *Server) getConversationMessages(ctx context.Context, req *getConversationMessagesRequest) (*getConversationMessagesResponse, error) {
	owner, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.sessions == nil {
		return nil, errNoSessions
	}
	service, err := s.sessions.Conversations()
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	page, err := service.HistoryForCaller(ctx, owner, req.AgentId, req.Cid, value(req.Before.ptr()), CallerFrom(ctx).UserID)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	b, _ := json.Marshal(page)
	var result map[string]any
	_ = json.Unmarshal(b, &result)
	return &getConversationMessagesResponse{Body: result}, nil
}

// getConversationCommand answers for a command whose session is gone, without opening one.
func (s *Server) getConversationCommand(ctx context.Context, req *getConversationCommandRequest) (*getConversationCommandResponse, error) {
	owner, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.sessions == nil {
		return nil, errNoSessions
	}
	service, err := s.sessions.Conversations()
	if err != nil {
		return nil, errUnknownCommand
	}
	receipt, err := service.CommandForCaller(ctx, owner, req.AgentId, req.Cid, CallerFrom(ctx).UserID, req.RequestId)
	if err != nil {
		return nil, errUnknownCommand
	}
	return &getConversationCommandResponse{Body: receiptOf(receipt)}, nil
}

// errNoApprovalHeld answers an approval no call of a running session waits for.
var errNoApprovalHeld = notFound("no call in this conversation waits for this approval: it was answered or " +
	"stopped already, its tool_call_id or request_id names none, or the channel is not one the caller is in")

// errApprovalAskedOfAnother answers a person answering a call held for another to allow.
var errApprovalAskedOfAnother = forbidden("this call waits for another person to allow it")

// errApprovalNeedsPerson answers an approval from a caller whose user id nobody vouched for.
var errApprovalNeedsPerson = forbidden("only a signed-in person or the app's backend may answer an approval")

// answerToolApproval hands a person's answer to a connector call held for their approval in
// the conversation cid, as a tool_approval on the session's socket does
// (Session.AnswerConnectorApproval). It is how the writer of a message in a channel of a type
// an agent config answers in (chat_channels) answers the call their message asked for: the
// Router's own session there is no socket of theirs. A person must still be a member of the
// channel; the customer's backend answers any.
func (s *Server) answerToolApproval(ctx context.Context, req *answerToolApprovalRequest) (*struct{}, error) {
	customer, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.sessions == nil {
		return nil, errNoSessions
	}
	by := OwnerFrom(ctx)
	backend := by.Kind == auth.KindServer
	if !backend && (!by.Kind.Verified() || by.UserID == "") {
		return nil, errApprovalNeedsPerson
	}
	found, running := s.sessions.ByConversationWhere(req.Cid, func(owner string, _ int64) bool { return owner == customer })
	if !running {
		return nil, errNoApprovalHeld
	}
	if !backend {
		service, err := s.sessions.Conversations()
		if err != nil {
			return nil, err
		}
		member, err := service.Member(ctx, customer, req.Cid, by.UserID)
		if err != nil {
			return nil, err
		}
		if !member {
			return nil, errNoApprovalHeld
		}
	}
	answer := req.Body
	err := found.AnswerConnectorApproval(answer.ToolCallId, answer.RequestId, answer.Allowed, answer.Summary, by)
	switch {
	case errors.Is(err, session.ErrApprovalAskedOfAnother):
		return nil, errApprovalAskedOfAnother
	case err != nil:
		return nil, errNoApprovalHeld
	}
	return nil, nil
}

// registerConversations declares the operations served in conversations.go.
func (s *Server) registerConversations(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getConversationMessages",
		Method:      http.MethodGet,
		Path:        "/v1/agents/conversations/{cid}/messages",
		Summary:     "Read a persistent text conversation",
		Responses: map[string]*huma.Response{
			"200": {Description: "Conversation messages, oldest first, with an older-page cursor"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getConversationMessages)
	huma.Register(api, huma.Operation{
		OperationID: "getConversationCommand",
		Method:      http.MethodGet,
		Path:        "/v1/agents/conversations/{cid}/commands/{request_id}",
		Summary:     "What a command in this conversation ended as",
		Description: "Reads one command's receipt from the conversation's own durable record. It opens " +
			"nothing and starts nothing, so a client whose stop found no session left to reach " +
			"reconciles that command here rather than reopening a session to ask about it.\n" +
			"A command still running is reported as it stands; the session holding it is where it " +
			"can be stopped.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The command's current receipt"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
	}, s.getConversationCommand)
	huma.Register(api, huma.Operation{
		OperationID: "answerConversationToolApproval",
		Method:      http.MethodPost,
		Path:        "/v1/agents/conversations/{cid}/tool_approvals",
		Summary:     "Answer a call waiting for a person's approval",
		Description: "Allows or declines a connector call whose binding sets `require_approval`, held on its " +
			"`ai_tool_call` step (`awaiting_approval`) in the conversation's channel, as `tool_approval` on " +
			"the session's socket does. The step's `target_user_id` is who may answer: the session's user, or " +
			"in a channel of a type an agent config answers in (`chat_channels`), the person whose message " +
			"the reply answers, with their own user token while they are a member of the channel. The app's " +
			"backend may answer any. Allowed, the call is sent; declined, it is not, and the model is told so.\n" +
			"A call held for another person to allow is a 403. A call nobody waits on, answered or stopped " +
			"already, or in a channel the caller is not a member of, is a 404.",
		DefaultStatus: http.StatusNoContent,
		Extensions:    map[string]any{clientAccessibleExtension: true},
		Responses: map[string]*huma.Response{
			"204": {Description: "The answer reached the call"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.answerToolApproval)
}

// ToolApprovalAnswer is a person's answer to a connector call waiting for their approval.
type ToolApprovalAnswer struct {
	ToolCallId string `json:"tool_call_id" minLength:"1" doc:"The call's id, the id of its ai_tool_call step."`
	RequestId  string `json:"request_id" minLength:"1" doc:"The command the reply answers: support_runtime.command_id on the reply's message."`
	Allowed    bool   `json:"allowed" doc:"Whether the person allows the call."`
	Summary    string `json:"summary,omitempty" maxLength:"120" doc:"Shown on the declined call, such as \"Not now\"."`
}

func (*ToolApprovalAnswer) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A person's answer to a connector call waiting for their approval. An unknown field is " +
		"refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

type answerToolApprovalRequest struct {
	Cid  string `path:"cid" doc:"The conversation's channel, type:id."`
	Body ToolApprovalAnswer
}

type getConversationMessagesRequest struct {
	Cid     string                `path:"cid"`
	AgentId string                `query:"agent_id" required:"true"`
	Before  optionalParam[string] `query:"before"`
}

type getConversationCommandRequest struct {
	Cid       string `path:"cid"`
	RequestId string `path:"request_id" doc:"The request id the question was sent with."`
	AgentId   string `query:"agent_id" required:"true"`
}

type getConversationCommandResponse struct {
	Body CommandReceipt
}

type getConversationMessagesResponse struct {
	Body map[string]any
}
