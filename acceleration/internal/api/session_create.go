package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/danielgtaylor/huma/v2"
)

// createSession joins a call and returns the session running it.
func (s *Server) createSession(ctx context.Context, request *createSessionRequest) (*createSessionResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.sessions == nil {
		return nil, huma.Error404NotFound(noSessions)
	}
	if request.Body == nil {
		return nil, huma.Error400BadRequest("a request body is required")
	}

	// A config is read before the session is created rather than inside it, so a caller
	// naming one that is not theirs is told so instead of getting a session that quietly
	// ignored it.
	config, failure := s.configFor(ctx, customerID, request.Body.ConfigId, request.Body.Agent)
	if failure != nil {
		if failure.status == notFound {
			return nil, huma.Error404NotFound(failure.message)
		}
		return nil, huma.Error400BadRequest(failure.message)
	}

	spec := specOf(*request.Body, customerID, config)
	// Who asked comes from the credential rather than from specOf, which merges the request
	// with the config and so only ever sees what the caller was willing to say about
	// themselves. Both halves are recorded, because the name is only worth what the kind
	// says it is: this pair is what the session is owned by and what every later request
	// for it is matched against.
	spec.Caller = CallerFrom(ctx)
	spec.CallerKind = KindFrom(ctx)
	created, err := s.sessions.Create(ctx, spec)
	if errors.Is(err, session.ErrSessionExists) {
		return nil, huma.Error409Conflict(err.Error())
	}
	if err != nil {
		// Everything that can go wrong here is the caller's spec or a provider that would
		// not start, and both are worth reading rather than a 500 with the detail in a
		// log the caller cannot see.
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &createSessionResponse{Body: sessionOf(created)}, nil
}

// registerSessionCreate declares the operations served in session_create.go.
func (s *Server) registerSessionCreate(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "createSession",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions",
		Summary:     "Join a call as a voice agent",
		Description: "The whole conversation runs here: the agent joins the call, transcribes what it hears, " +
			"answers it and speaks back, all through the routers. The caller keeps the session id " +
			"and watches the conversation over the events socket.\n" +
			"It returns once the agent is in the call, so a session that comes back is one that is " +
			"already listening. Tools declared here are the caller's own: the model asks for them " +
			"over the events socket and waits for the caller to answer.",
		Extensions:    map[string]any{clientAccessibleExtension: true},
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The agent is in the call"},
			"409": errorResponse("A session with that id already exists"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
	}, s.createSession)
}

type createSessionRequest struct {
	Body *CreateSessionRequest `required:"true"`
}

type createSessionResponse struct {
	Body Session
}
