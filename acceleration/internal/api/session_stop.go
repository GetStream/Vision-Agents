package api

import (
	"context"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
)

type stopSessionRequest struct {
	ID string `path:"id" doc:"The session, as returned when it was created."`
}

func (s *Server) registerSessionStop(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID:   "stopSession",
		Method:        http.MethodPost,
		Path:          "/v1/agents/sessions/{id}/stop",
		Summary:       "Stop a running session",
		DefaultStatus: http.StatusNoContent,
		Description: "The agent leaves the call and the session stops running. Everything it " +
			"recorded is kept: it can still be read back, renamed and forked, and what it " +
			"remembered carries into the next conversation. Deleting a session is what takes " +
			"those away.\n\n" +
			"A conversation in writing has nothing to hang up, so it is usually left running " +
			"rather than stopped. Stopping is for a call, where the agent is holding a line open.",
		Responses:  map[string]*huma.Response{"204": {Description: "The agent has left"}},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.stopSession)
}

// stopSession ends one of the caller's running sessions, keeping everything it recorded.
func (s *Server) stopSession(ctx context.Context, request *stopSessionRequest) (*struct{}, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if _, failure := s.session(ctx, request.ID); failure != nil {
		return nil, failure
	}
	stopped, err := s.sessions.Close(request.ID, OwnerFrom(ctx))
	if err != nil {
		return nil, err
	}
	if !stopped {
		return nil, errUnknownSession
	}
	return nil, nil
}
