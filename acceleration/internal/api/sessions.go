package api

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noSessions is what every session path says on a deployment that only inspects routing.
// It is a 404 rather than a 501 because the resource genuinely is not there: this router
// runs no conversations, so it holds no sessions to find.
const noSessions = "this deployment does not run sessions"

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
	var config *store.AgentConfig
	if id := value(request.Body.ConfigId); id != "" {
		if s.store == nil {
			return CreateSession400JSONResponse{badRequest(noConfigs)}, nil
		}
		found, err := s.store.AgentConfig(ctx, customerID, id)
		if err != nil {
			return CreateSession404JSONResponse{NotFoundJSONResponse{Error: unknownConfig}}, nil
		}
		config = &found
	}

	spec := specOf(*request.Body, customerID, config)
	created, err := s.sessions.Create(ctx, spec)
	if err != nil {
		// Everything that can go wrong here is the caller's spec or a provider that would
		// not start, and both are worth reading rather than a 500 with the detail in a
		// log the caller cannot see.
		return CreateSession400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateSession201JSONResponse(sessionOf(created)), nil
}

// ListSessions returns the calling customer's sessions, newest first.
func (s *Server) ListSessions(ctx context.Context, _ ListSessionsRequestObject) (ListSessionsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListSessions401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return ListSessions200JSONResponse{}, nil
	}

	running := s.sessions.List(customerID)
	listed := make([]Session, 0, len(running))
	for _, found := range running {
		listed = append(listed, sessionOf(found))
	}
	return ListSessions200JSONResponse(listed), nil
}

// GetSession returns one session.
func (s *Server) GetSession(ctx context.Context, request GetSessionRequestObject) (GetSessionResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		switch failure.status {
		case unauthorized:
			return GetSession401JSONResponse{missingCustomer()}, nil
		default:
			return GetSession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
		}
	}
	return GetSession200JSONResponse(sessionOf(found)), nil
}

// CloseSession ends a session, which is how the agent leaves the call.
func (s *Server) CloseSession(ctx context.Context, request CloseSessionRequestObject) (CloseSessionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CloseSession401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return CloseSession404JSONResponse{NotFoundJSONResponse{Error: noSessions}}, nil
	}

	closed, err := s.sessions.Close(request.Id, customerID)
	if err != nil {
		return nil, err
	}
	if !closed {
		return CloseSession404JSONResponse{NotFoundJSONResponse{Error: unknownSession}}, nil
	}
	return CloseSession204Response{}, nil
}

// SaySession speaks a piece of text without going through the model.
func (s *Server) SaySession(ctx context.Context, request SaySessionRequestObject) (SaySessionResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return SaySession401JSONResponse{missingCustomer()}, nil
		}
		return SaySession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if request.Body == nil || request.Body.Text == "" {
		return SaySession400JSONResponse{badRequest("there is nothing to say")}, nil
	}

	if err := found.Say(ctx, request.Body.Text); err != nil {
		return SaySession400JSONResponse{badRequest(err.Error())}, nil
	}
	return SaySession204Response{}, nil
}

// RespondSession answers a piece of text through the model.
func (s *Server) RespondSession(ctx context.Context, request RespondSessionRequestObject) (RespondSessionResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return RespondSession401JSONResponse{missingCustomer()}, nil
		}
		return RespondSession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if request.Body == nil || request.Body.Text == "" {
		return RespondSession400JSONResponse{badRequest("there is nothing to answer")}, nil
	}

	if err := found.Respond(ctx, request.Body.Text); err != nil {
		return RespondSession400JSONResponse{badRequest(err.Error())}, nil
	}
	return RespondSession204Response{}, nil
}

// InterruptSession abandons the reply being spoken.
func (s *Server) InterruptSession(ctx context.Context, request InterruptSessionRequestObject) (InterruptSessionResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return InterruptSession401JSONResponse{missingCustomer()}, nil
		}
		return InterruptSession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}

	found.Interrupt()
	return InterruptSession204Response{}, nil
}

// SetSessionInstructions changes what the agent is told to be.
func (s *Server) SetSessionInstructions(ctx context.Context, request SetSessionInstructionsRequestObject) (SetSessionInstructionsResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return SetSessionInstructions401JSONResponse{missingCustomer()}, nil
		}
		return SetSessionInstructions404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if request.Body == nil {
		return SetSessionInstructions400JSONResponse{badRequest("a request body is required")}, nil
	}

	found.SetInstructions(request.Body.Instructions)
	return SetSessionInstructions204Response{}, nil
}

// lookupStatus says which of the two ways finding a session can fail happened.
type lookupStatus int

const (
	unauthorized lookupStatus = iota
	notFound
)

// unknownSession is what a caller is told about a session that is not theirs, which is the
// same thing they are told about one that never existed.
const unknownSession = "no such session"

type lookupFailure struct {
	status  lookupStatus
	message string
}

// session finds a session belonging to the calling customer.
func (s *Server) session(ctx context.Context, id string) (*session.Session, *lookupFailure) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, &lookupFailure{status: unauthorized}
	}
	if s.sessions == nil {
		return nil, &lookupFailure{status: notFound, message: noSessions}
	}
	found, ok := s.sessions.Get(id, customerID)
	if !ok {
		return nil, &lookupFailure{status: notFound, message: unknownSession}
	}
	return found, nil
}

// specOf turns a request into what the session package needs, over whatever config it
// named. The customer comes from the trusted header rather than the body: a caller naming
// its own would be billing somebody else.
//
// The request wins wherever it says anything, so a caller can reuse a configuration and
// still change one thing about this call. A field the request omits is one the config
// decides, and a field neither mentions falls to the session defaults.
func specOf(request CreateSessionRequest, customerID string, config *store.AgentConfig) session.Spec {
	var spec session.Spec
	if config != nil {
		spec = session.FromConfig(*config)
	}
	spec.CallID = value(request.CallId)
	spec.CustomerID = customerID
	spec.Text = value(request.Text)

	spec.CallType = override(spec.CallType, request.CallType)
	spec.UserID = override(spec.UserID, request.UserId)
	spec.UserName = override(spec.UserName, request.UserName)
	spec.AgentID = override(spec.AgentID, request.AgentId)
	spec.Instructions = override(spec.Instructions, request.Instructions)
	spec.Greeting = override(spec.Greeting, request.Greeting)
	spec.Navigating = override(spec.Navigating, request.Navigating)
	spec.LLMTarget = override(spec.LLMTarget, request.Llm)
	spec.STTTarget = override(spec.STTTarget, request.Stt)
	spec.TTSTarget = override(spec.TTSTarget, request.Tts)
	spec.SubagentTarget = override(spec.SubagentTarget, request.Subagent)
	spec.Voice = override(spec.Voice, request.Voice)
	spec.MaxTokens = override(spec.MaxTokens, request.MaxTokens)
	spec.Tasks = override(spec.Tasks, request.Tasks)
	spec.ToolTimeoutMs = override(spec.ToolTimeoutMs, request.ToolTimeoutMs)
	spec.Backchannel = override(spec.Backchannel, request.Backchannel)
	spec.MinConfidence = override(spec.MinConfidence, request.MinConfidence)

	if request.Sandbox != nil {
		spec.Sandbox = string(*request.Sandbox)
	}

	if request.Languages != nil {
		spec.LanguageHints = *request.Languages
	}
	if request.Keyterms != nil {
		spec.Keyterms = *request.Keyterms
	}
	// Cost labels are merged rather than replaced: a config labels which agent the spend
	// belongs to and a call labels which conversation, and both are worth billing on.
	if request.Tags != nil {
		if spec.Tags == nil {
			spec.Tags = routing.Tags{}
		}
		for key, tag := range *request.Tags {
			spec.Tags[key] = tag
		}
	}
	if request.Memory != nil {
		spec.Memory = session.MemorySpec{
			UserID: value(request.Memory.UserId),
			AppID:  value(request.Memory.AppId),
		}
		if request.Memory.Filter != nil {
			spec.Memory.Filter = *request.Memory.Filter
		}
	}
	if request.Phone != nil {
		spec.Phone = &session.PhoneSpec{
			Number:       request.Phone.Number,
			Vendor:       value(request.Phone.Vendor),
			VendorCallID: value(request.Phone.VendorCallId),
		}
	}
	if request.Skills != nil {
		skills := harness.Skills{Skills: make([]harness.Skill, 0, len(*request.Skills))}
		for _, skill := range *request.Skills {
			skills.Skills = append(skills.Skills, harness.Skill{
				Name:         skill.Name,
				Description:  skill.Description,
				Instructions: skill.Instructions,
				Deadline:     time.Duration(value(skill.DeadlineMs)) * time.Millisecond,
			})
		}
		spec.Skills = &skills
	}
	if request.SkillNames != nil {
		spec.SkillNames = *request.SkillNames
	}
	if request.Tools != nil {
		for _, tool := range *request.Tools {
			declared := harness.Tool{Name: tool.Name, Description: tool.Description}
			if tool.Parameters != nil {
				declared.Parameters = *tool.Parameters
			}
			spec.Tools = append(spec.Tools, declared)
		}
	}
	return spec
}

// sessionOf renders a session for the wire.
func sessionOf(found *session.Session) Session {
	spec := found.Spec()
	model, voice := found.Provider()
	instructions := spec.Instructions

	rendered := Session{
		Id:        found.ID(),
		CallId:    spec.CallID,
		CallType:  spec.CallType,
		UserId:    spec.UserID,
		AgentId:   spec.AgentID,
		State:     SessionState(found.State()),
		CreatedAt: found.CreatedAt(),
	}
	if spec.Text {
		rendered.Text = &spec.Text
	}
	if model != "" {
		rendered.Llm = &model
	}
	if voice != "" {
		rendered.Tts = &voice
	}
	if instructions != "" {
		rendered.Instructions = &instructions
	}
	return rendered
}

// value reads an optional field, which the generated types carry as pointers.
func value[T any](pointer *T) T {
	if pointer == nil {
		var zero T
		return zero
	}
	return *pointer
}

// override prefers what the request said over what the config did. An omitted field is
// the config's to decide, which is the whole point of naming one.
func override[T any](base T, requested *T) T {
	if requested == nil {
		return base
	}
	return *requested
}
