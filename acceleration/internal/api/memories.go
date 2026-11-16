package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

var (
	errNoMemory = notConfigured("memory is not available: no memory provider configured")
	// errMemoryUnavailable names no vendor: which store a deployment uses, and what it said,
	// are for the log.
	errMemoryUnavailable = APIError{Type: ErrorTypeUnavailable, Code: codeMemoryUnavailable,
		Message: "What was remembered could not be removed right now. Try again later."}
	errSessionKept = APIError{Type: ErrorTypeUnavailable, Code: codeMemoryUnavailable,
		Message: "The session was not deleted: what it remembered could not be removed right now. Try again later."}
)

type truncateMemoriesRequest struct {
	UserID string `path:"user_id" minLength:"1" doc:"The memory user id sessions were opened with, memory.user_id on a session."`
}

type deleteSessionMemoriesRequest struct {
	ID string `path:"id" doc:"The session whose memories to delete."`
}

func (s *Server) registerMemories(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID:   "truncateMemories",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/users/{user_id}/memories",
		Summary:       "Delete everything remembered about one user",
		DefaultStatus: http.StatusNoContent,
		Description: "Deletes every memory about the user, whichever session and agent learned " +
			"it and whatever memory filter it was written under. Only the calling app's " +
			"memories are deleted, and a user nothing is known about is not an error.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The memories are deleted"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusServiceUnavailable},
	}, s.truncateMemories)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteSessionMemories",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/sessions/{id}/memories",
		Summary:       "Delete what one session remembered",
		DefaultStatus: http.StatusNoContent,
		Description: "Deletes every memory learned in the session, running or ended, and " +
			"leaves the rest of the user's memories alone. Ending a session keeps its " +
			"memories, so the next conversation knows what this one established; this is " +
			"how to take them back.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The memories are deleted"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusServiceUnavailable},
	}, s.deleteSessionMemories)
}

// truncateMemories deletes everything the calling app remembers about one user.
func (s *Server) truncateMemories(ctx context.Context, request *truncateMemoriesRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.sessions == nil {
		return nil, errNoMemory
	}
	if err := s.sessions.TruncateMemories(ctx, customerID, request.UserID); err != nil {
		return nil, memoryFailure(ctx, err, errMemoryUnavailable)
	}
	return nil, nil
}

// deleteSessionMemories deletes what one of the caller's sessions learned.
func (s *Server) deleteSessionMemories(ctx context.Context, request *deleteSessionMemoriesRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if _, failure := s.storedOrLiveSession(ctx, request.ID); failure != nil {
		return nil, failure
	}
	if err := s.sessions.ForgetSession(ctx, customerID, request.ID); err != nil {
		return nil, memoryFailure(ctx, err, errMemoryUnavailable)
	}
	return nil, nil
}

// memoryFailure is a 400 for a deployment with no memory, unavailable for a memory store
// that failed, and the error itself otherwise.
//
// The store's failure is answered without its words and left for the request log, which
// logs only the error an operation's answer was made from, not one it returned.
func memoryFailure(ctx context.Context, err error, unavailable APIError) error {
	switch {
	case errors.Is(err, session.ErrNoMemory):
		return errNoMemory
	case errors.Is(err, session.ErrMemoryFailed):
		recordFailure(ctx, err, stack.Trace(err))
		return unavailable
	}
	return stack.Wrap(err)
}
