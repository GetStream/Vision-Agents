package api

import (
	"context"
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// CreateSession joins a call and returns the session running it.
func (s *Server) CreateSession(ctx context.Context, request CreateSessionRequestObject) (CreateSessionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateSession401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return CreateSession404JSONResponse{NotFoundJSONResponse{Error: noSessions}}, nil
	}
	if request.Body == nil {
		return CreateSession400JSONResponse{badRequest("a request body is required")}, nil
	}

	// A config is read before the session is created rather than inside it, so a caller
	// naming one that is not theirs is told so instead of getting a session that quietly
	// ignored it.
	config, failure := s.configFor(ctx, customerID, request.Body.ConfigId, request.Body.Agent)
	if failure != nil {
		if failure.status == notFound {
			return CreateSession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
		}
		return CreateSession400JSONResponse{badRequest(failure.message)}, nil
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
		return CreateSession409JSONResponse{Error: err.Error()}, nil
	}
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		return nil, err
	}
	if err != nil {
		// Everything that can go wrong here is the caller's spec or a provider that would
		// not start, and both are worth reading rather than a 500 with the detail in a
		// log the caller cannot see.
		return CreateSession400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateSession201JSONResponse(sessionOf(created)), nil
}
