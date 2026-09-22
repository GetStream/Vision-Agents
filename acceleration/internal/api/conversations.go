package api

import (
	"context"
	"encoding/json"
)

func (s *Server) GetConversationMessages(ctx context.Context, req GetConversationMessagesRequestObject) (GetConversationMessagesResponseObject, error) {
	owner, ok := CustomerFrom(ctx)
	if !ok {
		return GetConversationMessages401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return GetConversationMessages400JSONResponse{badRequest(noSessions)}, nil
	}
	service, err := s.sessions.Conversations()
	if err != nil {
		return GetConversationMessages400JSONResponse{badRequest(err.Error())}, nil
	}
	page, err := service.HistoryForCaller(ctx, owner, req.Params.AgentId, req.Cid, value(req.Params.Before), CallerFrom(ctx).UserID)
	if err != nil {
		return GetConversationMessages400JSONResponse{badRequest(err.Error())}, nil
	}
	b, _ := json.Marshal(page)
	var result GetConversationMessages200JSONResponse
	_ = json.Unmarshal(b, &result)
	return result, nil
}

// GetConversationCommand answers for a command whose session is gone, without opening one.
func (s *Server) GetConversationCommand(ctx context.Context, req GetConversationCommandRequestObject) (GetConversationCommandResponseObject, error) {
	owner, ok := CustomerFrom(ctx)
	if !ok {
		return GetConversationCommand401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return GetConversationCommand404JSONResponse{NotFoundJSONResponse{Error: noSessions}}, nil
	}
	service, err := s.sessions.Conversations()
	if err != nil {
		return GetConversationCommand404JSONResponse{NotFoundJSONResponse{Error: unknownCommand}}, nil
	}
	receipt, err := service.CommandForCaller(ctx, owner, req.Params.AgentId, req.Cid, CallerFrom(ctx).UserID, req.CommandId)
	if err != nil {
		return GetConversationCommand404JSONResponse{NotFoundJSONResponse{Error: unknownCommand}}, nil
	}
	return GetConversationCommand200JSONResponse(receiptOf(receipt)), nil
}
