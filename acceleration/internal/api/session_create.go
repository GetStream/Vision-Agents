package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/danielgtaylor/huma/v2"
)

// createSession joins a call and returns the session running it.
func (s *Server) createSession(ctx context.Context, request *createSessionRequest) (*createSessionResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.sessions == nil {
		return nil, errNoSessions
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if failure, refused := refuseServerSideFields(ctx, *request.Body); refused {
		return nil, failure
	}
	if failure := s.refuseCall(ctx, customerID, *request.Body); failure != nil {
		return nil, failure
	}

	// A config is read before the session is created rather than inside it, so a caller
	// naming one that is not theirs is told so instead of getting a session that quietly
	// ignored it.
	config, failure := s.configFor(ctx, customerID, request.Body.ConfigId, request.Body.Agent)
	if failure != nil {
		return nil, failure
	}

	spec := specOf(*request.Body, customerID, config)
	ctx = s.threadConversation(ctx, customerID, &spec)
	// Who asked comes from the credential rather than from specOf, which merges the request
	// with the config and so only ever sees what the caller was willing to say about
	// themselves. Both halves are recorded, because the name is only worth what the kind
	// says it is: this pair is what the session is owned by and what every later request
	// for it is matched against.
	spec.Caller = CallerFrom(ctx)
	spec.CallerKind = KindFrom(ctx)
	created, err := s.sessions.Create(ctx, spec)
	if errors.Is(err, session.ErrSessionExists) {
		return nil, conflict(err.Error())
	}
	if errors.Is(err, session.ErrNoAgentCallType) {
		return nil, errNoAgentCallType
	}
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		return nil, err
	}
	if err != nil {
		// Everything that can go wrong here is the caller's spec or a provider that would
		// not start, and both are worth reading rather than a 500 with the detail in a
		// log the caller cannot see.
		return nil, invalidRequest(err.Error())
	}
	return &createSessionResponse{Body: sessionOf(created)}, nil
}

// errDeviceHistory refuses a history sent by an end user's device.
var errDeviceHistory = forbidden("history is sent server-side: an assistant message in it is " +
	"what the agent said, which a device cannot vouch for")

// errDeviceInstructions refuses instructions sent by an end user's device.
var errDeviceInstructions = forbidden("instructions are sent server-side: what the agent is " +
	"told decides who it is, which a device may not")

// errDeviceCall refuses a call named by an end user's device.
var errDeviceCall = forbidden("call is sent server-side: which call the agent joins puts it " +
	"among the people on that call, which a device may not decide")

// errCallWithoutVoice refuses a call named for a session held in writing.
var errCallWithoutVoice = invalidRequest("call needs start_voice: a session held in writing is on no call")

// errCallWithPhone refuses a call named for a phone session, which is on its own call.
var errCallWithPhone = invalidRequest("call cannot be given with phone: a phone session is on its own call, agent:<session id>")

// errCallInAnotherApp refuses a call another Stream app holds than the one the session acts
// in: the agent would join a call of the same name in its own app, where nobody is.
var errCallInAnotherApp = invalidRequest("the call is in another Stream app than the one this session acts in")

// refuseCall is the failure a new session naming a call is refused with, or nil. The session
// acts in the app the call is pinned to (callPinApp, as the manager resolves it), so the call
// is refused only when another session runs it in a different app.
func (s *Server) refuseCall(ctx context.Context, customerID string, request CreateSessionRequest) error {
	if request.Call == nil {
		return nil
	}
	if request.Phone != nil {
		return errCallWithPhone
	}
	if !value(request.StartVoice) {
		return errCallWithoutVoice
	}
	acting, err := s.callPinApp(ctx, customerID, request.Call.Type, request.Call.Id)
	if err != nil {
		return err
	}
	return s.refuseCallOutside(ctx, customerID, *request.Call, acting)
}

// refuseCallOutside refuses a call that callApp places in another Stream app than acting.
func (s *Server) refuseCallOutside(ctx context.Context, customerID string, call SessionCall, acting int64) error {
	app, err := s.callApp(ctx, customerID, call.Type, call.Id)
	if err != nil {
		return err
	}
	if app != acting {
		return errCallInAnotherApp
	}
	return nil
}

// refuseServerSideFields reports the failure a new session is refused with when an end user's
// device sent a field only the backend may send. createSession and the session socket both
// call it, so a field one of them refuses from a device the other refuses too, with the same
// error.
//
// What the model is told is the backend's to decide: an assistant message in a device's
// history would be the agent having said it, and a device's instructions could make the
// agent another one.
func refuseServerSideFields(ctx context.Context, request CreateSessionRequest) (APIError, bool) {
	if request.History != nil && !ServerSideFrom(ctx) {
		return errDeviceHistory, true
	}
	if request.Instructions != nil && !ServerSideFrom(ctx) {
		return errDeviceInstructions, true
	}
	if request.Call != nil && !ServerSideFrom(ctx) {
		return errDeviceCall, true
	}
	return APIError{}, false
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
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.createSession)
}

type createSessionRequest struct {
	Body *CreateSessionRequest `required:"true"`
}

type createSessionResponse struct {
	Body Session
}
