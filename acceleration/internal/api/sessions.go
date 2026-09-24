package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
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
	if err != nil {
		// Everything that can go wrong here is the caller's spec or a provider that would
		// not start, and both are worth reading rather than a 500 with the detail in a
		// log the caller cannot see.
		return CreateSession400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateSession201JSONResponse(sessionOf(created)), nil
}

// configFor resolves whichever way the caller addressed the agent.
//
// A name is the one a person actually knows the agent by, so it is worth supporting even
// though it costs a lookup. Both at once is refused rather than picking one: there is no
// sensible answer when they disagree, and quietly preferring the id would leave a caller
// wondering why the name they wrote had no effect.
func (s *Server) configFor(ctx context.Context, customerID string, configID, name *string) (*store.AgentConfig, *lookupFailure) {
	id, named := value(configID), value(name)
	switch {
	case id == "" && named == "":
		return nil, nil
	case id != "" && named != "":
		return nil, &lookupFailure{status: badInput,
			message: "name an agent by config_id or by agent, not both"}
	case s.store == nil:
		return nil, &lookupFailure{status: badInput, message: noConfigs}
	}

	if id != "" {
		found, err := s.store.AgentConfig(ctx, customerID, id)
		if err != nil {
			return nil, &lookupFailure{status: notFound, message: unknownConfig}
		}
		return &found, nil
	}

	found, exists, err := s.store.AgentConfigByName(ctx, customerID, named)
	if err != nil {
		return nil, &lookupFailure{status: badInput, message: err.Error()}
	}
	if !exists {
		// Refused rather than started unconfigured. A typo in a name would otherwise get a
		// working session with default instructions, which is far harder to notice than an
		// error: the agent answers, just not as the agent that was asked for.
		return nil, &lookupFailure{status: notFound,
			message: "there is no agent called " + named}
	}
	return &found, nil
}

// ListSessions returns the calling customer's sessions, newest first.
//
// Without filters it is the live sessions, as it always was. With any of them it is a query
// over what has happened too, so a caller asking for their conversations gets the ones that
// ended as well as the one they are having.
func (s *Server) ListSessions(ctx context.Context, request ListSessionsRequestObject) (ListSessionsResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return ListSessions401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return ListSessions200JSONResponse{}, nil
	}

	filter, err := sessionFilter(ctx, sessionQuery{
		Agent: request.Params.Agent, ConfigID: request.Params.ConfigId,
		UserID: request.Params.UserId, Project: request.Params.Project,
		State: string(value(request.Params.State)), Custom: request.Params.Custom,
		After: request.Params.CreatedAfter, Before: request.Params.CreatedBefore,
		Limit: request.Params.Limit, Offset: request.Params.Offset,
	})
	if err != nil {
		return ListSessions400JSONResponse{badRequest(err.Error())}, nil
	}

	found, err := s.sessions.Query(ctx, OwnerFrom(ctx), filter)
	if err != nil {
		return nil, err
	}
	return ListSessions200JSONResponse(sessionsOf(found)), nil
}

// SearchSessions finds a conversation by what the caller named it.
func (s *Server) SearchSessions(ctx context.Context, request SearchSessionsRequestObject) (SearchSessionsResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return SearchSessions401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return SearchSessions200JSONResponse{}, nil
	}

	filter, err := sessionFilter(ctx, sessionQuery{
		Agent: request.Params.Agent, ConfigID: request.Params.ConfigId,
		UserID: request.Params.UserId, Project: request.Params.Project,
		State: string(value(request.Params.State)), Custom: request.Params.Custom,
		After: request.Params.CreatedAfter, Before: request.Params.CreatedBefore,
		Limit: request.Params.Limit, Offset: request.Params.Offset,
	})
	if err != nil {
		return SearchSessions400JSONResponse{badRequest(err.Error())}, nil
	}

	found, err := s.sessions.Search(ctx, OwnerFrom(ctx), value(request.Params.Q), filter)
	if err != nil {
		return nil, err
	}
	return SearchSessions200JSONResponse(sessionsOf(found)), nil
}

// ForkSession continues a conversation as a new one.
func (s *Server) ForkSession(ctx context.Context, request ForkSessionRequestObject) (ForkSessionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ForkSession401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return ForkSession404JSONResponse{NotFoundJSONResponse{Error: noSessions}}, nil
	}

	parent, failure := s.storedOrLiveSession(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return ForkSession401JSONResponse{missingCustomer()}, nil
		}
		return ForkSession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}

	body := ForkSessionRequest{}
	if request.Body != nil {
		body = *request.Body
	}

	// A named agent on the fork replaces the parent's config wholesale, which is the point:
	// asking the same question of a different agent is the reason to fork.
	config, failure := s.configFor(ctx, customerID, body.ConfigId, body.Agent)
	if failure != nil {
		if failure.status == notFound {
			return ForkSession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
		}
		return ForkSession400JSONResponse{badRequest(failure.message)}, nil
	}

	spec, err := forkSpec(parent, body, config)
	if err != nil {
		return ForkSession400JSONResponse{badRequest(err.Error())}, nil
	}
	recalled, err := s.recordedHistory(ctx, parent, body, spec.Recall)
	switch {
	case errors.Is(err, store.ErrUnknownResponse):
		return ForkSession404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	case errors.Is(err, errForkNeedsHistory), errors.Is(err, errNoRecords):
		return ForkSession400JSONResponse{badRequest(err.Error())}, nil
	case err != nil:
		return nil, err
	}
	if recalled != nil {
		spec.Recall = &session.Recall{Messages: recalled}
	}
	spec.CustomerID = customerID
	spec.Caller = CallerFrom(ctx)
	spec.CallerKind = KindFrom(ctx)

	created, err := s.sessions.Create(ctx, spec)
	if err != nil {
		return ForkSession400JSONResponse{badRequest(err.Error())}, nil
	}
	return ForkSession201JSONResponse(sessionOf(created)), nil
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
	if _, ok := CustomerFrom(ctx); !ok {
		return CloseSession401JSONResponse{missingCustomer()}, nil
	}
	if s.sessions == nil {
		return CloseSession404JSONResponse{NotFoundJSONResponse{Error: noSessions}}, nil
	}

	if _, failure := s.session(ctx, request.Id); failure != nil {
		return CloseSession404JSONResponse{NotFoundJSONResponse{Error: unknownSession}}, nil
	}
	closed, err := s.sessions.Close(request.Id, OwnerFrom(ctx))
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

	if id := value(request.Body.CommandId); id != "" {
		receipt, err := found.RespondCommand(ctx, id, request.Body.Text)
		if errors.Is(err, conversation.ErrCommandConflict) {
			return RespondSession409JSONResponse{Error: err.Error()}, nil
		}
		if err != nil {
			return RespondSession400JSONResponse{badRequest(err.Error())}, nil
		}
		return RespondSession200JSONResponse{CommandId: receipt.CommandID, UserMessageId: receipt.UserMessageID,
			AssistantMessageId: receipt.AssistantMessageID, State: receipt.State, Duplicate: receipt.Duplicate}, nil
	}
	if _, err := found.Respond(ctx, request.Body.Text, nil); err != nil {
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

// RewindSession carries a conversation on from the end of one of its responses.
func (s *Server) RewindSession(ctx context.Context, request RewindSessionRequestObject) (RewindSessionResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return RewindSession401JSONResponse{missingCustomer()}, nil
		}
		return RewindSession404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if request.Body == nil || request.Body.ResponseId == "" {
		return RewindSession400JSONResponse{badRequest("name the response to carry on from")}, nil
	}
	if s.store == nil {
		return RewindSession400JSONResponse{badRequest(noStore)}, nil
	}

	err := found.Rewind(ctx, s.store, request.Body.ResponseId)
	switch {
	case errors.Is(err, store.ErrUnknownResponse):
		return RewindSession404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	case errors.Is(err, session.ErrCannotRewind):
		return RewindSession400JSONResponse{badRequest(err.Error())}, nil
	case err != nil:
		return nil, err
	}
	return RewindSession204Response{}, nil
}

// GetSessionCommand reports what one durable command ended as, without running anything.
func (s *Server) GetSessionCommand(ctx context.Context, request GetSessionCommandRequestObject) (GetSessionCommandResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return GetSessionCommand401JSONResponse{missingCustomer()}, nil
		}
		return GetSessionCommand404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}

	receipt, err := found.Command(request.CommandId)
	if err != nil {
		return GetSessionCommand404JSONResponse{NotFoundJSONResponse{Error: unknownCommand}}, nil
	}
	return GetSessionCommand200JSONResponse(receiptOf(receipt)), nil
}

// InterruptSessionCommand stops the named command and leaves every other one alone.
func (s *Server) InterruptSessionCommand(ctx context.Context, request InterruptSessionCommandRequestObject) (InterruptSessionCommandResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return InterruptSessionCommand401JSONResponse{missingCustomer()}, nil
		}
		return InterruptSessionCommand404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}

	receipt, err := found.InterruptCommand(request.CommandId)
	if errors.Is(err, conversation.ErrCommandNotFound) {
		return InterruptSessionCommand404JSONResponse{NotFoundJSONResponse{Error: unknownCommand}}, nil
	}
	if err != nil {
		// The stop was taken but its durable outcome is not known, so the caller is told
		// to keep the intent and retry this command id rather than that it stopped.
		return InterruptSessionCommand503JSONResponse{Error: err.Error()}, nil
	}
	return InterruptSessionCommand200JSONResponse(receiptOf(receipt)), nil
}

// receiptOf renders a durable command receipt for the wire.
func receiptOf(receipt conversation.CommandReceipt) CommandReceipt {
	return CommandReceipt{
		CommandId:          receipt.CommandID,
		UserMessageId:      receipt.UserMessageID,
		AssistantMessageId: receipt.AssistantMessageID,
		State:              receipt.State,
		Duplicate:          receipt.Duplicate,
	}
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

// SetSessionSettings moves one running session onto other models or another voice. The
// agent config it started from is untouched.
func (s *Server) SetSessionSettings(ctx context.Context, request SetSessionSettingsRequestObject) (SetSessionSettingsResponseObject, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return SetSessionSettings401JSONResponse{missingCustomer()}, nil
		}
		return SetSessionSettings404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if request.Body == nil {
		return SetSessionSettings400JSONResponse{badRequest("a request body is required")}, nil
	}

	body := request.Body
	settings := session.Settings{
		LLM: body.Llm, STT: body.Stt, TTS: body.Tts, STS: body.Sts, Subagent: body.Subagent,
		Voice: body.Voice, Temperature: body.Temperature, MaxOutputTokens: body.MaxOutputTokens,
	}
	if body.Thinking != nil {
		thinking := string(*body.Thinking)
		settings.Thinking = &thinking
	}
	if body.Verbosity != nil {
		verbosity := string(*body.Verbosity)
		settings.Verbosity = &verbosity
	}
	if err := found.SetSettings(ctx, settings); err != nil {
		return SetSessionSettings400JSONResponse{badRequest(err.Error())}, nil
	}
	return SetSessionSettings200JSONResponse(sessionOf(found)), nil
}

// lookupStatus says which way finding a session failed.
type lookupStatus int

const (
	unauthorized lookupStatus = iota
	notFound
	// badInput is a request that could not be understood, as against one that named
	// something real belonging to somebody else.
	badInput
)

// unknownSession is what a caller is told about a session that is not theirs, which is the
// same thing they are told about one that never existed.
const unknownSession = "no such session"

// unknownCommand is what a caller is told about a command this conversation never
// accepted, which is the same thing they are told about one they may not touch.
const unknownCommand = "no such command"

type lookupFailure struct {
	status  lookupStatus
	message string
}

// session finds a session belonging to the calling customer.
func (s *Server) session(ctx context.Context, id string) (*session.Session, *lookupFailure) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, &lookupFailure{status: unauthorized}
	}
	if s.sessions == nil {
		return nil, &lookupFailure{status: notFound, message: noSessions}
	}
	found, ok := s.sessions.Get(id, OwnerFrom(ctx))
	if !ok || !canReadSession(ctx, found.Spec()) {
		return nil, &lookupFailure{status: notFound, message: unknownSession}
	}
	return found, nil
}

// storedOrLiveSession finds a session whether or not it is still running.
//
// Reading back a conversation that ended is the ordinary case for everything that takes a
// session id and does not talk to the agent -- its turns, its items, forking it -- so those
// go through here rather than through session, which only knows about live ones.
//
// A session that ended and one that belongs to somebody else are both reported as not
// found: a different answer for each would make this a way to discover whose an id is.
func (s *Server) storedOrLiveSession(ctx context.Context, id string) (session.Found, *lookupFailure) {
	if _, ok := CustomerFrom(ctx); !ok {
		return session.Found{}, &lookupFailure{status: unauthorized}
	}
	if s.sessions == nil {
		return session.Found{}, &lookupFailure{status: notFound, message: noSessions}
	}

	owner := OwnerFrom(ctx)
	if live, ok := s.sessions.Get(id, owner); ok {
		return session.Found{Live: live}, nil
	}
	if s.store == nil {
		return session.Found{}, &lookupFailure{status: notFound, message: unknownSession}
	}

	row, err := s.store.StoredSession(ctx, owner.CustomerID, id)
	if err != nil {
		return session.Found{}, &lookupFailure{status: notFound, message: unknownSession}
	}
	// The row carries who opened it, which is what the live path checks through the
	// manager. Skipping it here would let one of a customer's users read another's.
	if !owner.Reaches(session.Owner{
		CustomerID: row.CustomerID, UserID: row.UserID, Kind: auth.Kind(row.CallerKind),
	}) {
		return session.Found{}, &lookupFailure{status: notFound, message: unknownSession}
	}
	return session.Found{Stored: &row}, nil
}

// sessionQuery is the filter as it arrives, which is the same set of parameters on listing
// and on searching. Gathered into one struct so the two cannot drift apart in what they
// admit: a filter honoured by one and forgotten by the other is one a caller uses to read
// somebody else's conversations.
type sessionQuery struct {
	Agent, ConfigID, UserID, Project *string
	// State is a string rather than either of the two generated enums, because the
	// generator makes one type per operation and they are the same parameter.
	State         string
	Custom        *string
	After, Before *time.Time
	Limit, Offset *int
}

// sessionFilter turns query parameters into a store filter, refusing what cannot be meant.
//
// The user id is the one parameter a caller does not get to choose freely: anybody who is
// not the app's own backend is narrowed to their own sessions whatever they asked for,
// because a filter a caller can widen is not a boundary. The manager narrows it again for
// the same reason; two checks is the right number for something that decides whose
// conversations a stranger can read.
func sessionFilter(ctx context.Context, query sessionQuery) (store.SessionFilter, error) {
	filter := store.SessionFilter{
		AgentName: value(query.Agent),
		ConfigID:  value(query.ConfigID),
		Project:   value(query.Project),
		State:     query.State,
		Limit:     value(query.Limit),
		Offset:    value(query.Offset),
	}
	if query.After != nil {
		filter.After = *query.After
	}
	if query.Before != nil {
		filter.Before = *query.Before
	}

	if requested := value(query.UserID); requested != "" {
		if KindFrom(ctx) != auth.KindServer {
			return store.SessionFilter{}, errors.New(
				"only a server-side caller may list another user's sessions")
		}
		filter.UserID = requested
	}

	if labels := value(query.Custom); labels != "" {
		decoded := map[string]any{}
		if err := json.Unmarshal([]byte(labels), &decoded); err != nil {
			return store.SessionFilter{}, errors.New("custom must be a JSON object of labels")
		}
		filter.Custom = make(map[string]string, len(decoded))
		for key, held := range decoded {
			// Flattened to strings because that is what a query string carries and what the
			// containment check compares: a caller who labelled a session with the number 4
			// finds it again by typing 4.
			filter.Custom[key] = fmt.Sprint(held)
		}
	}
	return filter, nil
}

// sessionsOf renders a query's results, taking the live half where there is one: a session
// in flight knows what routing resolved its models to, which the row does not carry.
func sessionsOf(found []session.Found) []Session {
	listed := make([]Session, 0, len(found))
	for _, one := range found {
		switch {
		case one.Live != nil:
			rendered := sessionOf(one.Live)
			// A live session that also has a row takes the row's afterthoughts: a title
			// somebody set from another tab is on the row and not on the running spec.
			if one.Stored != nil {
				mergeStored(&rendered, one.Stored)
			}
			listed = append(listed, rendered)
		case one.Stored != nil:
			listed = append(listed, storedSessionOf(*one.Stored))
		}
	}
	return listed
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
	spec.PersistConversation = value(request.PersistConversation)
	spec.ConversationID = value(request.ConversationId)

	spec.Incognito = value(request.Incognito)
	spec.Title = override(spec.Title, request.Title)
	spec.Description = override(spec.Description, request.Description)
	spec.Project = override(spec.Project, request.Project)
	if request.Custom != nil {
		spec.Custom = *request.Custom
	}
	if request.ModelOverwrites != nil {
		spec.ModelOverwrites = modelOverwritesOf(*request.ModelOverwrites)
	}

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
	spec.STSTarget = override(spec.STSTarget, request.Sts)
	// A request that asks for writing gets writing, whatever the config's model: "test
	// this agent in writing" has to work against a native config too.
	if spec.Text {
		spec.STSTarget = ""
	}
	spec.SubagentTarget = override(spec.SubagentTarget, request.Subagent)
	spec.SearchTarget = override(spec.SearchTarget, request.Search)
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
	if request.Video != nil {
		spec.VideoSource = override(spec.VideoSource, request.Video.Source)
		spec.VideoMaxFrames = override(spec.VideoMaxFrames, request.Video.MaxFrames)
	}
	if request.Skills != nil {
		skills := harness.Skills{Skills: make([]harness.Skill, 0, len(*request.Skills))}
		for _, skill := range *request.Skills {
			skills.Skills = append(skills.Skills, harness.Skill{
				Name: skill.Name, Revision: value(skill.Revision),
				CaptureVideo: value(skill.CaptureVideo),
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
	stt, model, voice, think := found.Resolved()
	instructions := spec.Instructions

	rendered := Session{
		ConversationId: &spec.ConversationID, ContextTruncated: &spec.ContextTruncated,
		Id:        found.ID(),
		CallId:    spec.CallID,
		CallType:  spec.CallType,
		UserId:    spec.UserID,
		AgentId:   spec.AgentID,
		State:     SessionState(found.State()),
		CreatedAt: found.CreatedAt(),
	}
	if found.CapturesVideo() {
		rendered.Video = &SessionVideo{Source: &spec.VideoSource, MaxFrames: &spec.VideoMaxFrames}
	}
	if spec.Text {
		rendered.Text = &spec.Text
	}
	describe(&rendered, spec)
	if stt != "" {
		rendered.Stt = &stt
	}
	if model != "" {
		rendered.Llm = &model
	}
	if voice != "" {
		rendered.Tts = &voice
	}
	if think != "" {
		rendered.Subagent = &think
	}
	if speech := found.Speech(); speech != "" {
		rendered.Sts = &speech
	}
	_, speaking := found.Voice()
	rendered.Voice = optional(speaking)
	mode := SessionMode(found.Mode())
	rendered.Mode = &mode
	if instructions != "" {
		rendered.Instructions = &instructions
	}
	return rendered
}

// describe puts the caller's own labels on a rendered session. They are the same fields on
// either side of the wire, so a caller reads back what they asked for rather than having to
// remember it.
func describe(rendered *Session, spec session.Spec) {
	if spec.AgentName != "" {
		rendered.Agent = &spec.AgentName
	}
	if spec.ConfigID != "" {
		rendered.ConfigId = &spec.ConfigID
	}
	if spec.Incognito {
		rendered.Incognito = &spec.Incognito
	}
	if spec.Title != "" {
		rendered.Title = &spec.Title
	}
	if spec.Description != "" {
		rendered.Description = &spec.Description
	}
	if spec.Project != "" {
		rendered.Project = &spec.Project
	}
	if len(spec.Custom) > 0 {
		rendered.Custom = &spec.Custom
	}
	if spec.ForkedFrom != "" {
		rendered.ForkedFrom = &spec.ForkedFrom
	}
	if !spec.ModelOverwrites.Empty() {
		rendered.ModelOverwrites = modelOverwritesFor(spec.ModelOverwrites)
	}
}

// storedSessionOf renders a session that ended, from the row that outlived it.
//
// The resolved models are absent rather than guessed. The row carries what the session was
// asked to use, not what routing picked on each turn, and reporting the request as though it
// were the answer would have a caller reading a failover that happened as a model that ran.
func storedSessionOf(row store.AgentSession) Session {
	rendered := Session{
		Id:        row.ID,
		CallId:    row.CallID,
		CallType:  row.CallType,
		UserId:    row.UserID,
		AgentId:   row.AgentID,
		State:     SessionState(row.State),
		CreatedAt: row.CreatedAt,
	}
	// A session with no call was held in writing, which is what the absence of one means.
	if row.CallID == "" {
		text := true
		rendered.Text = &text
	}
	if row.ConversationID != "" {
		rendered.ConversationId = &row.ConversationID
	}
	mergeStored(&rendered, &row)
	return rendered
}

// mergeStored writes what only the row knows onto a rendered session: the labels, and when
// it ended.
func mergeStored(rendered *Session, row *store.AgentSession) {
	if row.AgentName != "" {
		rendered.Agent = &row.AgentName
	}
	if row.ConfigID != "" {
		rendered.ConfigId = &row.ConfigID
	}
	if row.Title != "" {
		rendered.Title = &row.Title
	}
	if row.Description != "" {
		rendered.Description = &row.Description
	}
	if row.Project != "" {
		rendered.Project = &row.Project
	}
	if len(row.Custom) > 0 {
		custom := row.Custom
		rendered.Custom = &custom
	}
	if row.ForkedFrom != "" {
		rendered.ForkedFrom = &row.ForkedFrom
	}
	if !row.ModelOverwrites.Empty() {
		rendered.ModelOverwrites = modelOverwritesFor(row.ModelOverwrites)
	}
	rendered.ClosedAt = row.ClosedAt
	rendered.LastResponseAt = row.LastResponseAt
}

// modelOverwritesOf reads what the caller asked to change about the models. The name is
// longer than it wants to be because overwritesOf already means the per-provider option
// blocks a router config carries, which are a different thing entirely.
func modelOverwritesOf(sent ModelOverwrites) store.ModelOverwrites {
	return store.ModelOverwrites{
		LLM: value(sent.Llm), STT: value(sent.Stt), TTS: value(sent.Tts),
		STS: value(sent.Sts), Subagent: value(sent.Subagent), Search: value(sent.Search),
		Thinking:        string(value(sent.Thinking)),
		Temperature:     sent.Temperature,
		MaxOutputTokens: sent.MaxOutputTokens,
		Verbosity:       string(value(sent.Verbosity)),
	}
}

// modelOverwritesFor renders them back, so a caller reads what they asked for.
func modelOverwritesFor(held store.ModelOverwrites) *ModelOverwrites {
	rendered := &ModelOverwrites{
		Temperature: held.Temperature, MaxOutputTokens: held.MaxOutputTokens,
	}
	if held.LLM != "" {
		rendered.Llm = &held.LLM
	}
	if held.STT != "" {
		rendered.Stt = &held.STT
	}
	if held.TTS != "" {
		rendered.Tts = &held.TTS
	}
	if held.STS != "" {
		rendered.Sts = &held.STS
	}
	if held.Subagent != "" {
		rendered.Subagent = &held.Subagent
	}
	if held.Search != "" {
		rendered.Search = &held.Search
	}
	if held.Thinking != "" {
		thinking := ModelOverwritesThinking(held.Thinking)
		rendered.Thinking = &thinking
	}
	if held.Verbosity != "" {
		verbosity := ModelOverwritesVerbosity(held.Verbosity)
		rendered.Verbosity = &verbosity
	}
	return rendered
}

// forkSpec is the parent's spec with the fork's request written over it.
//
// The parent is read from whichever half of Found has it. A live parent is preferable -- it
// carries the whole spec, tools and skills included -- but a conversation worth continuing
// has usually ended, so the row has to be enough on its own.
func forkSpec(parent session.Found, request ForkSessionRequest, config *store.AgentConfig) (session.Spec, error) {
	var spec session.Spec
	var parentID string
	var wasText bool
	// What the fork reads its history out of, which is the parent's channel and the agent
	// id that channel belongs to rather than whichever agent the fork runs as.
	var recall *session.Recall

	switch {
	case parent.Live != nil:
		spec = parent.Live.Spec()
		parentID = parent.Live.ID()
		wasText = spec.Text
		if spec.Incognito {
			return session.Spec{}, errors.New(
				"an incognito session is not recorded, so there is nothing to fork from")
		}
		if spec.ConversationID != "" {
			recall = &session.Recall{AgentID: spec.AgentID, ConversationID: spec.ConversationID}
		}
	case parent.Stored != nil:
		row := parent.Stored
		spec = session.Spec{
			AgentName: row.AgentName, ConfigID: row.ConfigID,
			Title: row.Title, Description: row.Description, Project: row.Project,
			Custom: row.Custom, ModelOverwrites: row.ModelOverwrites,
			CallType: row.CallType,
		}
		parentID = row.ID
		wasText = row.CallID == ""
		spec.Text = wasText
		if row.ConversationID != "" {
			recall = &session.Recall{AgentID: row.AgentID, ConversationID: row.ConversationID}
		}
	default:
		return session.Spec{}, errors.New("there is nothing to fork")
	}

	// A config named on the fork replaces the parent's models wholesale rather than merging
	// with them, because asking the same question of a different agent is the reason to
	// fork and a half-replaced agent would be neither one.
	if config != nil {
		fresh := session.FromConfig(*config)
		fresh.Text = spec.Text
		fresh.Title, fresh.Description = spec.Title, spec.Description
		fresh.Project, fresh.Custom = spec.Project, spec.Custom
		fresh.CallType = spec.CallType
		spec = fresh
	}

	spec.ForkedFrom = parentID
	spec.Title = override(spec.Title, request.Title)
	spec.Description = override(spec.Description, request.Description)
	spec.Project = override(spec.Project, request.Project)
	spec.Instructions = override(spec.Instructions, request.Instructions)
	spec.Incognito = value(request.Incognito)
	if request.Custom != nil {
		spec.Custom = *request.Custom
	}
	if request.ModelOverwrites != nil {
		spec.ModelOverwrites = modelOverwritesOf(*request.ModelOverwrites)
	}

	// The fork is its own conversation, so it gets its own channel and its own agent id:
	// sharing the parent's would have two sessions writing into one transcript.
	spec.ConversationID = ""
	spec.AgentID = ""
	spec.CallID = value(request.CallId)
	switch {
	case wasText && spec.CallID != "":
		return session.Spec{}, errors.New("a text session cannot be forked into a call")
	case !wasText && spec.CallID == "":
		return session.Spec{}, errors.New("forking a voice session needs a call to join")
	}

	// History comes across by default: the usual reason to fork is to carry on from what was
	// already said. It needs a channel at both ends -- the parent's to read and the fork's to
	// write -- so a fork that is carrying history persists, and a parent that kept none has
	// none to hand over. Recall is cleared rather than inherited, because a fork of a fork
	// reads its own parent and not its grandparent: the parent's channel already holds both.
	spec.Recall = nil
	if recall != nil && (request.Messages == nil || *request.Messages) {
		spec.PersistConversation = true
		spec.Recall = recall
	}
	return spec, nil
}

var (
	errForkNeedsHistory = errors.New(
		"response_id says where the carried history stops, so it cannot be combined with messages false")
	errNoRecords = errors.New(noStore)
)

// recordedHistory is the history a fork reads out of what its parent recorded rather than
// out of a Chat channel: up to the named response, or all of it for a parent that kept no
// channel to read. Nil leaves the fork carrying whatever forkSpec decided.
func (s *Server) recordedHistory(ctx context.Context, parent session.Found, request ForkSessionRequest, recall *session.Recall) ([]llm.Message, error) {
	carry := request.Messages == nil || *request.Messages
	responseID := value(request.ResponseId)
	switch {
	case responseID != "" && !carry:
		return nil, errForkNeedsHistory
	case responseID == "" && (!carry || recall != nil):
		return nil, nil
	case s.store == nil && responseID != "":
		return nil, errNoRecords
	case s.store == nil:
		return nil, nil
	}

	// A running parent may have said something the writer has not caught up with yet.
	if parent.Live != nil {
		if err := parent.Live.FlushRecords(ctx); err != nil {
			return nil, err
		}
	}
	exchanges, err := s.store.Exchanges(ctx, OwnerFrom(ctx).CustomerID, parent.ID(), responseID)
	if err != nil {
		return nil, err
	}
	if len(exchanges) == 0 && responseID == "" {
		return nil, nil
	}
	return session.HistoryOf(exchanges), nil
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

// Persistent personal sessions use the same caller binding as their Chat channel.
func canReadSession(ctx context.Context, spec session.Spec) bool {
	return !spec.PersistConversation || spec.Caller.UserID == CallerFrom(ctx).UserID
}
