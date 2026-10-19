package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/danielgtaylor/huma/v2"
)

type sessionVoiceRequest struct {
	ID string `path:"id" doc:"The session, as returned when it was created."`
}

type startSessionVoiceRequest struct {
	ID   string                    `path:"id" doc:"The session, as returned when it was created."`
	Body *StartSessionVoiceRequest `required:"false"`
}

// StartSessionVoiceRequest is what voice is started with.
type StartSessionVoiceRequest struct {
	Call *SessionCall `json:"call,omitempty" doc:"An existing Stream call for the agent to join, such as the default call the people are already on. Left out, the agent goes back to the call voice was last on, else joins agent:<session id>, which joining creates. Naming a call other than the one voice is on while it is started is refused with a 409: stop voice first to move. The session has to act in the customer's own Stream app: in a Stream app shared with other customers (deployment tenancy, or app tenancy falling back to the deployment's app) a call is refused with a 403. The call has to be in the Stream app the session acts in, and the router does not ask Stream whether it exists. A phone or socket session takes no call. The call runs on its own type's audio settings: the built-in agent type turns DTX and noise cancellation off so turn detection hears every word, and the default type has DTX on and noise cancellation on. Server-side only: a device sending it is refused with a 403."`
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
		Description: "The agent joins the call named in call, else the one voice was last on, else " +
			"agent:<session id>, which joining creates, with the conversation so far, and returns " +
			"once it is there. What is said on the call and what " +
			"is typed into the session are one conversation, kept in the same Stream Chat channel: " +
			"a typed question is answered aloud. The models are the ones the session was opened " +
			"with, or the defaults, and a native config speaks with its speech-to-speech model.\n\n" +
			"Starting voice on a session that is already on its call changes nothing, and naming " +
			"another call while it is on is refused. A " +
			"conversation in writing that ended is carried on under the same id, as a message to " +
			"it is.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The agent is on the call"},
			"409": errorResponse("Voice is on another call"),
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
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
func (s *Server) startSessionVoice(ctx context.Context, request *startSessionVoiceRequest) (*sessionVoiceResponse, error) {
	var named *SessionCall
	if request.Body != nil {
		named = request.Body.Call
	}
	if named != nil && !ServerSideFrom(ctx) {
		return nil, errDeviceCall
	}
	found, sent, err := s.sessionToAnswer(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	defer sent()
	var call session.Call
	if named != nil {
		if found.Spec().Phone != nil {
			return nil, errCallWithPhone
		}
		if found.Spec().Edge != nil {
			return nil, errSocketCall
		}
		customerID, _ := CustomerFrom(ctx)
		if err := s.refuseCallOutside(ctx, customerID, *named, found.Spec().StreamApp); err != nil {
			return nil, err
		}
		call = session.Call{Type: named.Type, ID: named.Id}
	}
	if err := found.StartVoice(ctx, call); err != nil {
		if errors.Is(err, session.ErrOnAnotherCall) {
			return nil, errVoiceOnAnotherCall
		}
		if errors.Is(err, session.ErrVoiceUnavailable) {
			return nil, errVoiceUnavailable
		}
		if errors.Is(err, session.ErrNoAgentCallType) {
			return nil, errNoAgentCallType
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

// errVoiceOnAnotherCall refuses naming a call while voice is on a different one.
var errVoiceOnAnotherCall = conflict("voice is on another call: stop voice first to move it")

// errVoiceUnavailable refuses voice on a session that cannot hold a call.
var errVoiceUnavailable = notConfigured("this session cannot start voice")

// errNoAgentCallType refuses a call in a Stream app of the customer's own that has no agent
// call type, which Stream would refuse the agent's call for.
var errNoAgentCallType = notConfigured("this app's Stream app has no " + streamapp.AgentCallType +
	" call type, so the agent has no call to join: create it with " +
	`getstream api CreateCallType --request '{"name":"` + streamapp.AgentCallType + `"}'`)
