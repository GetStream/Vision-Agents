package api

import (
	"context"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
)

type deleteSessionRequest struct {
	ID string `path:"id" doc:"The session, as returned when it was created."`
}

func (s *Server) registerSessionDelete(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID:   "deleteSession",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/sessions/{id}",
		Summary:       "Delete a session",
		DefaultStatus: http.StatusNoContent,
		Description: "Deletes the session, running or stopped: it is stopped first if it is " +
			"running, then its turns and their items are deleted, and so is everything it " +
			"taught the memory store. Memories other sessions learned about the same user are " +
			"kept. The transcript a conversation in writing kept in Stream Chat is not deleted.\n\n" +
			"If the memory store fails, the answer is a 503 memory_unavailable and the " +
			"session is kept, stopped: deleting it again later deletes the rest.\n\n" +
			"To end a call and keep the conversation, stop the session instead.",
		Responses:  map[string]*huma.Response{"204": {Description: "The session is deleted"}},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusServiceUnavailable},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.deleteSession)
}

// deleteSession deletes one of the caller's sessions and what it remembered.
func (s *Server) deleteSession(ctx context.Context, request *deleteSessionRequest) (*struct{}, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if _, failure := s.storedOrLiveSession(ctx, request.ID); failure != nil {
		return nil, failure
	}
	if err := s.sessions.Delete(ctx, request.ID, OwnerFrom(ctx)); err != nil {
		return nil, memoryFailure(ctx, err, errSessionKept)
	}
	// A deleted conversation takes its tasks with it: there is nowhere left
	// for a follow-up to land. Their runs go with them (the store cascades).
	if s.store != nil {
		customerID, _ := CustomerFrom(ctx)
		if err := s.store.DeleteScheduledTasksForSession(ctx, customerID, request.ID); err != nil {
			s.logger.Error("could not delete a deleted session's scheduled tasks",
				"session", request.ID, "error", err)
		}
	}
	return nil, nil
}
