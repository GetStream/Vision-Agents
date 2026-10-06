package api

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
)

func (s *Server) getConversationMessages(ctx context.Context, req *getConversationMessagesRequest) (*getConversationMessagesResponse, error) {
	owner, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer()
	}
	if s.sessions == nil {
		return nil, invalidRequest(noSessions)
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
		return nil, missingCustomer()
	}
	if s.sessions == nil {
		return nil, notFound(noSessions)
	}
	service, err := s.sessions.Conversations()
	if err != nil {
		return nil, notFound(unknownCommand)
	}
	receipt, err := service.CommandForCaller(ctx, owner, req.AgentId, req.Cid, CallerFrom(ctx).UserID, req.CommandId)
	if err != nil {
		return nil, notFound(unknownCommand)
	}
	return &getConversationCommandResponse{Body: receiptOf(receipt)}, nil
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
		Path:        "/v1/agents/conversations/{cid}/commands/{command_id}",
		Summary:     "What a command in this conversation ended as",
		Description: "Reads one command's receipt from the conversation's own durable record. It opens " +
			"nothing and starts nothing, so a client whose stop found no session left to reach " +
			"reconciles that command here rather than reopening a session to ask about it.\n" +
			"A command still running is reported as it stands; the session holding it is where it " +
			"can be stopped.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The command's current receipt"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusNotFound},
	}, s.getConversationCommand)
}

type getConversationMessagesRequest struct {
	Cid     string                `path:"cid"`
	AgentId string                `query:"agent_id" required:"true"`
	Before  optionalParam[string] `query:"before"`
}

type getConversationCommandRequest struct {
	Cid       string `path:"cid"`
	CommandId string `path:"command_id" doc:"The client's own command id, as sent when the command was submitted."`
	AgentId   string `query:"agent_id" required:"true"`
}

type getConversationCommandResponse struct {
	Body CommandReceipt
}

type getConversationMessagesResponse struct {
	Body map[string]any
}
