package api

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
)

type CommandReceipt struct {
	CommandId          string `json:"command_id"`
	UserMessageId      string `json:"user_message_id"`
	AssistantMessageId string `json:"assistant_message_id"`
	State              string `json:"state" doc:"Latest locally recorded response state; an interrupted command is never automatically rerun."`
	Duplicate          bool   `json:"duplicate" doc:"True when this command already exists and no new inference was started."`
}

type getConversationMessagesRequest struct {
	CID     string                `path:"cid"`
	AgentID string                `query:"agent_id" required:"true"`
	Before  optionalParam[string] `query:"before"`
}

type objectResponse struct {
	Body map[string]any
}

type getConversationCommandRequest struct {
	CID       string `path:"cid"`
	CommandID string `path:"command_id" doc:"The client's own command id, as sent when the command was submitted."`
	AgentID   string `query:"agent_id" required:"true"`
}

type commandReceiptResponse struct {
	Body CommandReceipt
}

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
		Path:        "/v1/agents/conversations/{cid}/commands/{command_id}",
		Summary:     "What a command in this conversation ended as",
		Description: "Reads one command's receipt from the conversation's own durable record. It " +
			"opens nothing and starts nothing, so a client whose stop found no session left " +
			"to reach reconciles that command here rather than reopening a session to ask " +
			"about it.\n" +
			"A command still running is reported as it stands; the session holding it is " +
			"where it can be stopped.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The command's current receipt"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusNotFound},
	}, s.getConversationCommand)
}

func (s *Server) getConversationMessages(ctx context.Context, request *getConversationMessagesRequest) (*objectResponse, error) {
	owner, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.sessions == nil {
		return nil, huma.Error400BadRequest(noSessions)
	}
	service, err := s.sessions.Conversations()
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	page, err := service.HistoryForCaller(ctx, owner, request.AgentID, request.CID, request.Before.Value, CallerFrom(ctx).UserID)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	b, _ := json.Marshal(page)
	var result map[string]any
	_ = json.Unmarshal(b, &result)
	return &objectResponse{Body: result}, nil
}

// getConversationCommand answers for a command whose session is gone, without opening one.
func (s *Server) getConversationCommand(ctx context.Context, request *getConversationCommandRequest) (*commandReceiptResponse, error) {
	owner, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.sessions == nil {
		return nil, huma.Error404NotFound(noSessions)
	}
	service, err := s.sessions.Conversations()
	if err != nil {
		return nil, huma.Error404NotFound(unknownCommand)
	}
	receipt, err := service.CommandForCaller(ctx, owner, request.AgentID, request.CID, CallerFrom(ctx).UserID, request.CommandID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCommand)
	}
	return &commandReceiptResponse{Body: receiptOf(receipt)}, nil
}
