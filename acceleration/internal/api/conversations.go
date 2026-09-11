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
	page, err := service.History(ctx, owner, req.Params.AgentId, req.Cid, value(req.Params.Before))
	if err != nil {
		return GetConversationMessages400JSONResponse{badRequest(err.Error())}, nil
	}
	b, _ := json.Marshal(page)
	var result GetConversationMessages200JSONResponse
	_ = json.Unmarshal(b, &result)
	return result, nil
}
