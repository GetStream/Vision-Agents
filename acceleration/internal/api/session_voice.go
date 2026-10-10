package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/danielgtaylor/huma/v2"
)

type sessionVoiceRequest struct {
	ID string `path:"id" doc:"The session, as returned when it was created."`
}

type sessionVoiceResponse struct {
	Body Session
}

func (s *Server) registerSessionVoice(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "startSessionVoice",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/voice",
		Summary:     "Start voice on a session",
		Description: "The agent joins the call agent:<session id>, which joining creates, with the " +
			"conversation so far, and returns once it is there. What is said on the call and what " +
			"is typed into the session are one conversation, kept in the same Stream Chat channel: " +
			"a typed question is answered aloud. The models are the ones the session was opened " +
			"with, or the defaults, and a native config speaks with its speech-to-speech model.\n\n" +
			"Starting voice on a session that is already on its call changes nothing. A " +
			"conversation in writing that ended is carried on under the same id, as a message to " +
			"it is.",
		Responses:  map[string]*huma.Response{"200": {Description: "The agent is on the call"}},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.startSessionVoice)
	huma.Register(api, huma.Operation{
		OperationID: "stopSessionVoice",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/sessions/{id}/voice",
		Summary:     "Stop voice on a session",
		Description: "The agent finishes what it is saying, leaves the call and carries the " +
			"conversation on in writing, with everything said on the call. Stopping voice on a " +
			"session held in writing changes nothing. Stopping the session is what ends it.",
		Responses:  map[string]*huma.Response{"200": {Description: "The agent has left the call"}},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.stopSessionVoice)
}

// startSessionVoice puts the agent on the session's call.
func (s *Server) startSessionVoice(ctx context.Context, request *sessionVoiceRequest) (*sessionVoiceResponse, error) {
	found, sent, err := s.sessionToAnswer(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	defer sent()
	if err := found.StartVoice(ctx); err != nil {
		if errors.Is(err, session.ErrVoiceUnavailable) {
			return nil, errVoiceUnavailable
		}
		return nil, invalidRequest(err.Error())
	}
	return &sessionVoiceResponse{Body: sessionOf(found)}, nil
}

// stopSessionVoice takes the agent off the session's call.
func (s *Server) stopSessionVoice(ctx context.Context, request *sessionVoiceRequest) (*sessionVoiceResponse, error) {
	found, failure := s.session(ctx, request.ID)
	if failure != nil {
		return nil, failure
	}
	if err := found.StopVoice(ctx); err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &sessionVoiceResponse{Body: sessionOf(found)}, nil
}

// errVoiceUnavailable refuses voice on a session that cannot hold a call.
var errVoiceUnavailable = notConfigured("this session cannot start voice")
