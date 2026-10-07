package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/danielgtaylor/huma/v2"
)

// errNoSessions is what every session path says on a deployment that only inspects routing.
// It is a 404 rather than a 501 because the resource genuinely is not there: this router
// runs no conversations, so it holds no sessions to find.
var errNoSessions = notConfigured("this deployment does not run sessions")

// configFor resolves whichever way the caller addressed the agent.
//
// A name is the one a person actually knows the agent by, so it is worth supporting even
// though it costs a lookup. Both at once is refused rather than picking one: there is no
// sensible answer when they disagree, and quietly preferring the id would leave a caller
// wondering why the name they wrote had no effect.
func (s *Server) configFor(ctx context.Context, customerID string, configID, name *string) (*store.AgentConfig, error) {
	id, named := value(configID), value(name)
	switch {
	case id == "" && named == "":
		return nil, nil
	case id != "" && named != "":
		return nil, invalidRequest("name an agent by config_id or by agent, not both")
	case s.store == nil:
		return nil, errNoConfigs
	}

	if id != "" {
		found, err := s.configs.AgentConfig(ctx, customerID, id)
		if err != nil {
			return nil, errUnknownConfig
		}
		return &found, nil
	}

	found, exists, err := s.configs.AgentConfigByName(ctx, customerID, named)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	if !exists {
		// Refused rather than started unconfigured. A typo in a name would otherwise get a
		// working session with default instructions, which is far harder to notice than an
		// error: the agent answers, just not as the agent that was asked for.
		return nil, notFound("there is no agent called " + named)
	}
	return &found, nil
}

// forkSession continues a conversation as a new one.
func (s *Server) forkSession(ctx context.Context, request *forkSessionRequest) (*forkSessionResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.sessions == nil {
		return nil, errNoSessions
	}

	parent, failure := s.storedOrLiveSession(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}

	body := ForkSessionRequest{}
	if request.Body != nil {
		body = *request.Body
	}

	// A named agent on the fork replaces the parent's config wholesale, which is the point:
	// asking the same question of a different agent is the reason to fork.
	config, failure := s.configFor(ctx, customerID, body.ConfigId, body.Agent)
	if failure != nil {
		return nil, failure
	}

	spec, err := forkSpec(parent, body, config)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	recalled, err := s.recordedHistory(ctx, parent, body, spec.Recall)
	switch {
	case errors.Is(err, store.ErrUnknownResponse):
		return nil, notFound(err.Error())
	case errors.Is(err, errForkNeedsHistory):
		return nil, invalidRequest(err.Error())
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
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		return nil, err
	}
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &forkSessionResponse{Body: sessionOf(created)}, nil
}

// getSession returns one session.
func (s *Server) getSession(ctx context.Context, request *getSessionRequest) (*getSessionResponse, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	return &getSessionResponse{Body: sessionOf(found)}, nil
}

// saySession speaks a piece of text without going through the model.
func (s *Server) saySession(ctx context.Context, request *saySessionRequest) (*struct{}, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	if request.Body == nil || request.Body.Text == "" {
		return nil, invalidRequest("there is nothing to say")
	}

	if err := found.Say(ctx, request.Body.Text); err != nil {
		return nil, invalidRequest(err.Error())
	}
	return nil, nil
}

// respondSession answers a piece of text through the model.
func (s *Server) respondSession(ctx context.Context, request *respondSessionRequest) (*respondSessionResponse, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	if request.Body == nil || request.Body.Text == "" {
		return nil, invalidRequest("there is nothing to answer")
	}

	if id := value(request.Body.CommandId); id != "" {
		receipt, _, err := found.RespondCommand(ctx, id, request.Body.Text, value(request.Body.ClientId))
		if errors.Is(err, conversation.ErrCommandConflict) {
			return nil, conflict(err.Error())
		}
		if err != nil {
			return nil, invalidRequest(err.Error())
		}
		return &respondSessionResponse{Status: http.StatusOK, Body: &CommandReceipt{
			CommandId: receipt.CommandID, UserMessageId: receipt.UserMessageID,
			AssistantMessageId: receipt.AssistantMessageID, State: receipt.State, Duplicate: receipt.Duplicate,
		}}, nil
	}
	if _, err := found.Respond(ctx, request.Body.Text, nil); err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &respondSessionResponse{Status: http.StatusNoContent}, nil
}

// respondSessionResponse is a durable command's receipt, or nothing when the text named no
// command and the model is simply answering.
type respondSessionResponse struct {
	Status int
	Body   *CommandReceipt
}

// interruptSession abandons the reply being spoken.
func (s *Server) interruptSession(ctx context.Context, request *interruptSessionRequest) (*struct{}, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}

	found.Interrupt()
	return nil, nil
}

// rewindSession carries a conversation on from the end of one of its responses.
func (s *Server) rewindSession(ctx context.Context, request *rewindSessionRequest) (*struct{}, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	if request.Body == nil || request.Body.ResponseId == "" {
		return nil, invalidRequest("name the response to carry on from")
	}
	if s.store == nil {
		return nil, errNoStore
	}

	err := found.Rewind(ctx, s.store, request.Body.ResponseId)
	switch {
	case errors.Is(err, store.ErrUnknownResponse):
		return nil, notFound(err.Error())
	case errors.Is(err, session.ErrCannotRewind):
		return nil, invalidRequest(err.Error())
	case err != nil:
		return nil, err
	}
	return nil, nil
}

// getSessionCommand reports what one durable command ended as, without running anything.
func (s *Server) getSessionCommand(ctx context.Context, request *getSessionCommandRequest) (*getSessionCommandResponse, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}

	receipt, err := found.Command(request.CommandId)
	if err != nil {
		return nil, errUnknownCommand
	}
	return &getSessionCommandResponse{Body: receiptOf(receipt)}, nil
}

// interruptSessionCommand stops the named command and leaves every other one alone.
func (s *Server) interruptSessionCommand(ctx context.Context, request *interruptSessionCommandRequest) (*interruptSessionCommandResponse, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}

	receipt, err := found.InterruptCommand(request.CommandId)
	if errors.Is(err, conversation.ErrCommandNotFound) {
		return nil, errUnknownCommand
	}
	if err != nil {
		// The stop was taken but its durable outcome is not known, so the caller is told
		// to keep the intent and retry this command id rather than that it stopped.
		return nil, unavailable(err.Error())
	}
	return &interruptSessionCommandResponse{Body: receiptOf(receipt)}, nil
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

// setSessionInstructions changes what the agent is told to be.
func (s *Server) setSessionInstructions(ctx context.Context, request *setSessionInstructionsRequest) (*struct{}, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	found.SetInstructions(request.Body.Instructions)
	return nil, nil
}

// setSessionSettings moves one running session onto other models or another voice. The
// agent config it started from is untouched.
func (s *Server) setSessionSettings(ctx context.Context, request *setSessionSettingsRequest) (*setSessionSettingsResponse, error) {
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	body := request.Body
	settings := session.Settings{
		LLM: body.Llm, STT: body.Stt, TTS: body.Tts, STS: body.Sts,
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
		return nil, invalidRequest(err.Error())
	}
	return &setSessionSettingsResponse{Body: sessionOf(found)}, nil
}

// errUnknownSession is what a caller is told about a session that is not theirs, which is the
// same thing they are told about one that never existed.
var errUnknownSession = APIError{Type: ErrorTypeNotFound, Code: codeSessionNotFound, Message: "no such session"}

// errUnknownCommand is what a caller is told about a command this conversation never
// accepted, which is the same thing they are told about one they may not touch.
var errUnknownCommand = APIError{Type: ErrorTypeNotFound, Code: codeCommandNotFound, Message: "no such command"}

// session finds a session belonging to the calling customer.
func (s *Server) session(ctx context.Context, id string) (*session.Session, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if s.sessions == nil {
		return nil, errNoSessions
	}
	found, ok := s.sessions.Get(id, OwnerFrom(ctx))
	if !ok || !canReadSession(ctx, found.Spec()) {
		return nil, errUnknownSession
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
func (s *Server) storedOrLiveSession(ctx context.Context, id string) (session.Found, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return session.Found{}, errMissingCustomer
	}
	if s.sessions == nil {
		return session.Found{}, errNoSessions
	}

	owner := OwnerFrom(ctx)
	if live, ok := s.sessions.Get(id, owner); ok {
		return session.Found{Live: live}, nil
	}
	if s.store == nil {
		return session.Found{}, errUnknownSession
	}

	row, err := s.store.StoredSession(ctx, owner.CustomerID, id)
	if err != nil {
		return session.Found{}, errUnknownSession
	}
	// The row carries who opened it, which is what the live path checks through the
	// manager. Skipping it here would let one of a customer's users read another's.
	if !owner.Reaches(session.Owner{
		CustomerID: row.CustomerID, UserID: row.UserID, Kind: auth.Kind(row.CallerKind),
	}) {
		return session.Found{}, errUnknownSession
	}
	return session.Found{Stored: &row}, nil
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
	spec.ID = value(request.Id)
	spec.CallID = value(request.CallId)
	spec.CustomerID = customerID
	spec.Text = value(request.Text)
	// A text conversation is kept in Stream Chat unless it is incognito, which Normalize
	// turns off, so any Chat client can read it back.
	spec.PersistConversation = spec.Text
	spec.ConversationID = value(request.ConversationId)
	for _, said := range value(request.History) {
		spec.History = append(spec.History, conversation.HistoryLine{
			Role: string(said.Role), Text: said.Text, Name: value(said.Name), At: value(said.CreatedAt),
		})
	}

	spec.Incognito = value(request.Incognito)
	spec.Title = override(spec.Title, request.Title)
	spec.Description = override(spec.Description, request.Description)
	spec.Project = override(spec.Project, request.ProjectId)
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
	spec.SearchTarget = override(spec.SearchTarget, request.Search)
	spec.Voice = override(spec.Voice, request.Voice)
	spec.MaxTokens = override(spec.MaxTokens, request.MaxTokens)
	spec.ToolTimeoutMs = override(spec.ToolTimeoutMs, request.ToolTimeoutMs)
	spec.Backchannel = override(spec.Backchannel, request.Backchannel)
	spec.MinConfidence = override(spec.MinConfidence, request.MinConfidence)

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
	if request.Tools != nil {
		for _, tool := range *request.Tools {
			declared := harness.Tool{Name: tool.Name, Description: tool.Description, DisplayTitle: value(tool.DisplayTitle)}
			if tool.Parameters != nil {
				declared.Parameters = *tool.Parameters
			}
			if tool.Executor != nil && *tool.Executor == SessionToolExecutorClient {
				declared.Client = true
			}
			if approval := tool.Approval; approval != nil && approval.Title != "" {
				declared.Approval = &harness.ToolApproval{
					Title: approval.Title, Message: value(approval.Message), ReasonArgument: value(approval.ReasonArgument),
					AllowTitle: value(approval.AllowTitle), DeclineTitle: value(approval.DeclineTitle),
				}
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
		Modality:  SessionModality(found.Modality()),
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
		rendered.ThinkingLlm = &think
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
		rendered.ProjectId = &spec.Project
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
		State:     Live,
		CreatedAt: row.CreatedAt,
	}
	if row.State == store.SessionClosed {
		rendered.State = Ended
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
	// A live session knows its modality before the row that records it.
	if rendered.Modality == "" {
		rendered.Modality = SessionModality(row.Modality)
	}
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
		rendered.ProjectId = &row.Project
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
		STS: value(sent.Sts), Search: value(sent.Search),
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
			return session.Spec{}, stack.Wrap(errors.New(
				"an incognito session is not recorded, so there is nothing to fork from"))
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
		return session.Spec{}, stack.Wrap(errors.New("there is nothing to fork"))
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
	spec.Project = override(spec.Project, request.ProjectId)
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
	spec.ID = ""
	spec.ConversationID = ""
	spec.AgentID = ""
	spec.CallID = value(request.CallId)
	switch {
	case wasText && spec.CallID != "":
		return session.Spec{}, stack.Wrap(errors.New("a text session cannot be forked into a call"))
	case !wasText && spec.CallID == "":
		return session.Spec{}, stack.Wrap(errors.New("forking a voice session needs a call to join"))
	}

	// History comes across by default: the usual reason to fork is to carry on from what was
	// already said. It needs a channel at both ends -- the parent's to read and the fork's to
	// write -- so a fork that is carrying history persists, and a parent that kept none has
	// none to hand over. Recall is cleared rather than inherited, because a fork of a fork
	// reads its own parent and not its grandparent: the parent's channel already holds both.
	spec.PersistConversation = spec.Text
	spec.Recall = nil
	if recall != nil && (request.Messages == nil || *request.Messages) {
		spec.PersistConversation = true
		spec.Recall = recall
	}
	return spec, nil
}

var errForkNeedsHistory = errors.New(
	"response_id says where the carried history stops, so it cannot be combined with messages false")

// recordedHistory is the history a fork reads out of what its parent recorded rather than
// out of a Chat channel: up to the named response, or all of it for a parent that kept no
// channel to read. Nil leaves the fork carrying whatever forkSpec decided.
func (s *Server) recordedHistory(ctx context.Context, parent session.Found, request ForkSessionRequest, recall *session.Recall) ([]llm.Message, error) {
	carry := request.Messages == nil || *request.Messages
	responseID := value(request.ResponseId)
	switch {
	case responseID != "" && !carry:
		return nil, stack.Wrap(errForkNeedsHistory)
	case responseID == "" && (!carry || recall != nil):
		return nil, nil
	case s.store == nil && responseID != "":
		return nil, errNoStore
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

// registerSessions declares the operations served in sessions.go.
func (s *Server) registerSessions(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getSession",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}",
		Summary:     "One session",
		Description: "Reading a session is open to the device holding it, for the same reason listing and " +
			"stopping are: it is the conversation the caller is having. A session belonging to " +
			"somebody else is reported as not found rather than refused, so this is not a way to " +
			"find out whose an id is.",
		Extensions: map[string]any{clientAccessibleExtension: true},
		Responses: map[string]*huma.Response{
			"200": {Description: "The session"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getSession)
	huma.Register(api, huma.Operation{
		OperationID: "forkSession",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/fork",
		Summary:     "Continue a conversation as a new one",
		Description: "Opens a session from another's spec, carrying its history across by default, and " +
			"records where it came from. The usual reason is to ask the same question of a different " +
			"model without losing the original answer, which is why anything in the request is " +
			"written over what the parent was opened with.\n" +
			"The parent is untouched and keeps running if it was running. Forking an incognito " +
			"session is refused rather than answered with an empty conversation: there is nothing " +
			"recorded to fork from, and pretending otherwise would hand back a session that quietly " +
			"lost everything the caller thought they were continuing.",
		Extensions:    map[string]any{clientAccessibleExtension: true},
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The fork is running"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.forkSession)
	huma.Register(api, huma.Operation{
		OperationID: "rewindSession",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/rewind",
		Summary:     "Go back to a response and carry on from there",
		Description: "The conversation continues as though nothing after the named response had been said: " +
			"the reply being spoken is abandoned, the agent's history is cut back to the end of that " +
			"response, and every later response is marked rewound, so neither the responses nor " +
			"their items list them again. The named response itself is kept.\n" +
			"The history is rebuilt from what the session recorded, the question and the answer of " +
			"each turn, so a session that recorded nothing cannot be rewound: an incognito one, one " +
			"on a deployment with no store, and a native speech-to-speech one, whose model keeps its " +
			"own context. A persistent conversation is refused as well, because its transcript lives " +
			"in Chat and would bring the rewound turns back the next time it opened; fork it at the " +
			"response instead.",
		Extensions:    map[string]any{clientAccessibleExtension: true},
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The conversation carries on from the end of that response"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.rewindSession)
	huma.Register(api, huma.Operation{
		OperationID: "saySession",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/say",
		Summary:     "Speak a piece of text without going through the model",
		Description: "For when the caller already knows what should be said, such as a greeting. A model " +
			"would only add latency and cost to words that were never in question.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The text is being spoken"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.saySession)
	huma.Register(api, huma.Operation{
		OperationID: "respondSession",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/respond",
		Summary:     "Answer a piece of text through the model, as though it had been said",
		Responses: map[string]*huma.Response{
			"200": {Description: "Durable command accepted or replayed; only a new command starts inference"},
			"409": errorResponse("The command ID was already accepted with different content"),
			"204": {Description: "The model is answering"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.respondSession)
	huma.Register(api, huma.Operation{
		OperationID: "interruptSession",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/interrupt",
		Summary:     "Abandon the reply being spoken",
		Description: "What a caller outside the call has instead of a voice. A murmur is not interrupted, " +
			"because it was meant to overlap with whoever is talking.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The reply was abandoned, if there was one"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.interruptSession)
	huma.Register(api, huma.Operation{
		OperationID: "getSessionCommand",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/commands/{command_id}",
		Summary:     "What is known about one durable command",
		Description: "Reads a command's receipt without accepting, running or stopping anything. It is how a " +
			"client whose stop or submission had an unknown outcome reconciles the same command id " +
			"rather than inventing another one.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The command's current receipt"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
	}, s.getSessionCommand)
	huma.Register(api, huma.Operation{
		OperationID: "interruptSessionCommand",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/commands/{command_id}/interrupt",
		Summary:     "Stop one named command, and nothing else",
		Description: "Abandons the reply that command is generating. Unlike interrupting the session, a stop " +
			"that arrives after its command finished replays that command's terminal receipt and " +
			"leaves the command running now alone, so a delayed stop for one question can never take " +
			"the answer to the next one.\n" +
			"A command accepted but not yet generating is prevented from starting. A command already " +
			"completed, failed, cancelled or interrupted returns what it ended as. An unknown " +
			"command is a 404, the same answer as a conversation the caller does not own.\n" +
			"Interrupting model work claims nothing about a tool whose external side effect already " +
			"happened.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The command's terminal receipt"},
			"503": errorResponse("The stop was accepted but its durable outcome is unknown. The command is not reported stopped; retry the same command id."),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
	}, s.interruptSessionCommand)
	huma.Register(api, huma.Operation{
		OperationID: "setSessionInstructions",
		Method:      http.MethodPut,
		Path:        "/v1/agents/sessions/{id}/instructions",
		Summary:     "Change what the agent is told to be",
		Description: "Deprecated: use updateSession. Applies from the next turn. The reply being spoken keeps " +
			"the prompt it started with, because rewriting it mid-sentence would have the agent " +
			"change character in the middle of a thought.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The next turn will use them"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.setSessionInstructions)
	huma.Register(api, huma.Operation{
		OperationID: "setSessionSettings",
		Method:      http.MethodPatch,
		Path:        "/v1/agents/sessions/{id}/settings",
		Summary:     "Change the models and voice of one running session",
		Description: "Deprecated: use updateSession. Swaps what the agent runs on without leaving the call, " +
			"for this session only: the agent config it started from is untouched. The new models " +
			"are opened before anything changes, so a target that does not route is refused and the " +
			"agent carries on as it was. They take over from the next turn; a reply being spoken " +
			"finishes on the models it started with.\n" +
			"Naming sts makes the session native, and an empty sts makes it a cascade again, on " +
			"whatever llm, stt and tts it names or had before. The conversation carries across: a " +
			"conversation model is handed the history on every turn, and a speech-to-speech model is " +
			"opened with the recent transcript in its instructions.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The session, on its new models"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.setSessionSettings)
}

type getSessionRequest struct {
	Id string `path:"id" doc:"The session, as returned when it was created."`
}

type getSessionResponse struct {
	Body Session
}

type forkSessionRequest struct {
	Id   string `path:"id" doc:"The session, as returned when it was created."`
	Body *ForkSessionRequest
}

type forkSessionResponse struct {
	Body Session
}

type rewindSessionRequest struct {
	Id   string                `path:"id" doc:"The session, as returned when it was created."`
	Body *RewindSessionRequest `required:"true"`
}

type saySessionRequest struct {
	Id   string      `path:"id" doc:"The session, as returned when it was created."`
	Body *SayRequest `required:"true"`
}

type respondSessionRequest struct {
	Id   string          `path:"id" doc:"The session, as returned when it was created."`
	Body *RespondRequest `required:"true"`
}

type interruptSessionRequest struct {
	Id string `path:"id" doc:"The session, as returned when it was created."`
}

type getSessionCommandRequest struct {
	Id        string `path:"id" doc:"The session, as returned when it was created."`
	CommandId string `path:"command_id" doc:"The client's own command id, as sent when the command was submitted."`
}

type getSessionCommandResponse struct {
	Body CommandReceipt
}

type interruptSessionCommandRequest struct {
	Id        string `path:"id" doc:"The session, as returned when it was created."`
	CommandId string `path:"command_id" doc:"The client's own command id, as sent when the command was submitted."`
}

type interruptSessionCommandResponse struct {
	Body CommandReceipt
}

type setSessionInstructionsRequest struct {
	Id   string               `path:"id" doc:"The session, as returned when it was created."`
	Body *InstructionsRequest `required:"true"`
}

type setSessionSettingsRequest struct {
	Id   string                  `path:"id" doc:"The session, as returned when it was created."`
	Body *SessionSettingsRequest `required:"true"`
}

type setSessionSettingsResponse struct {
	Body Session
}

// ForkSessionRequest Continue a conversation as a new one. Everything the parent was opened with is inherited; anything named here is written over it, which is what makes a fork useful rather than a copy -- the usual reason to fork is to ask the same question of a different model.
type ForkSessionRequest struct {
	Agent           *string                 `json:"agent,omitempty"`
	CallId          *string                 `json:"call_id,omitempty" doc:"The call the fork joins. A voice session cannot be forked into a text one or the other way about, so this is required when the parent held a call and refused when it did not."`
	ConfigId        *string                 `json:"config_id,omitempty"`
	Custom          *map[string]interface{} `json:"custom,omitempty"`
	Description     *string                 `json:"description,omitempty"`
	Incognito       *bool                   `json:"incognito,omitempty" doc:"Hold the fork off the record. The parent still exists; this conversation onwards is simply not kept."`
	Instructions    *string                 `json:"instructions,omitempty"`
	Messages        *bool                   `json:"messages,omitempty" doc:"Carry the parent's history into the fork, so the new conversation continues from what was already said. False starts the same configuration over from nothing, which is what comparing two answers to the same opening question wants." default:"true"`
	ModelOverwrites *ModelOverwrites        `json:"model_overwrites,omitempty"`
	ProjectId       *string                 `json:"project_id,omitempty"`
	ResponseId      *string                 `json:"response_id,omitempty" doc:"Carry the parent's history only up to the end of this response, so the fork continues from that point rather than from where the parent is now. The history is read from what the parent recorded, which also lets a parent that kept no Chat transcript be forked with its history. Cannot be combined with messages false."`
	Title           *string                 `json:"title,omitempty"`
}

func (*ForkSessionRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Continue a conversation as a new one. Everything the parent was opened with is inherited; anything named here is written over it, which is what makes a fork useful rather than a copy -- the usual reason to fork is to ask the same question of a different model."
	return schema
}

// InstructionsRequest is the InstructionsRequest schema.
type InstructionsRequest struct {
	Instructions string `json:"instructions"`
}

// ModelOverwrites What to change about the models for one session, over whatever its agent config decided.
// It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides the config, and a caller reading a session back wants to see what they changed in one place rather than diffed against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because a caller able to rewrite those could make a session impersonate a different agent.
type ModelOverwrites struct {
	Llm             *string                   `json:"llm,omitempty" doc:"A provider/model or a capability shortcut, in place of the config's."`
	MaxOutputTokens *int                      `json:"max_output_tokens,omitempty" doc:"Caps the reply, reasoning included. Omitted leaves the provider's default."`
	Search          *string                   `json:"search,omitempty"`
	Sts             *string                   `json:"sts,omitempty" doc:"A speech-to-speech target. Naming one here makes the session native even if the config did not, which means no transcriber, model or voice is opened."`
	Stt             *string                   `json:"stt,omitempty"`
	Temperature     *float64                  `json:"temperature,omitempty" doc:"How random the answer is. Omitted leaves the provider's own default, which is not the same as zero: zero is a real request for a deterministic model."`
	Thinking        *ModelOverwritesThinking  `json:"thinking,omitempty" doc:"How hard to reason before answering. It becomes the reasoning effort on the request, which is the vocabulary the providers that support one already speak, and means nothing to a model that does not reason." enum:"none,minimal,low,medium,high"`
	Tts             *string                   `json:"tts,omitempty"`
	Verbosity       *ModelOverwritesVerbosity `json:"verbosity,omitempty" doc:"How much detail to give. Dropped for models that do not take it." enum:"low,medium,high"`
}

func (*ModelOverwrites) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_output_tokens"].Format = ""
	schema.Description = "What to change about the models for one session, over whatever its agent config decided.\nIt is one object rather than a dozen fields at the top level because it is one idea: everything here overrides the config, and a caller reading a session back wants to see what they changed in one place rather than diffed against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because a caller able to rewrite those could make a session impersonate a different agent."
	return schema
}

// ModelOverwritesThinking is the ModelOverwritesThinking schema.
type ModelOverwritesThinking string

// Defines values for ModelOverwritesThinking.
const (
	ModelOverwritesThinkingHigh    ModelOverwritesThinking = "high"
	ModelOverwritesThinkingLow     ModelOverwritesThinking = "low"
	ModelOverwritesThinkingMedium  ModelOverwritesThinking = "medium"
	ModelOverwritesThinkingMinimal ModelOverwritesThinking = "minimal"
	ModelOverwritesThinkingNone    ModelOverwritesThinking = "none"
)

// Valid indicates whether the value is a known member of the ModelOverwritesThinking enum.
func (e ModelOverwritesThinking) Valid() bool {
	switch e {
	case ModelOverwritesThinkingHigh:
		return true
	case ModelOverwritesThinkingLow:
		return true
	case ModelOverwritesThinkingMedium:
		return true
	case ModelOverwritesThinkingMinimal:
		return true
	case ModelOverwritesThinkingNone:
		return true
	default:
		return false
	}
}

// ModelOverwritesVerbosity is the ModelOverwritesVerbosity schema.
type ModelOverwritesVerbosity string

// Defines values for ModelOverwritesVerbosity.
const (
	ModelOverwritesVerbosityHigh   ModelOverwritesVerbosity = "high"
	ModelOverwritesVerbosityLow    ModelOverwritesVerbosity = "low"
	ModelOverwritesVerbosityMedium ModelOverwritesVerbosity = "medium"
)

// Valid indicates whether the value is a known member of the ModelOverwritesVerbosity enum.
func (e ModelOverwritesVerbosity) Valid() bool {
	switch e {
	case ModelOverwritesVerbosityHigh:
		return true
	case ModelOverwritesVerbosityLow:
		return true
	case ModelOverwritesVerbosityMedium:
		return true
	default:
		return false
	}
}

// RespondRequest is the RespondRequest schema.
type RespondRequest struct {
	ClientId  *string `json:"client_id,omitempty" doc:"The install the command came from. It is written on the person's message as client_id, and a client tool called while answering is addressed to it." pattern:"^[A-Za-z0-9_.:-]{1,128}$"`
	CommandId *string `json:"command_id,omitempty" doc:"Required for personal persistent text conversations. Reuse this ID and identical text for retries; duplicate acceptance does not restart inference." pattern:"^[A-Za-z0-9_-]{1,128}$"`
	Text      string  `json:"text" minLength:"1"`
}

// RewindSessionRequest is the RewindSessionRequest schema.
type RewindSessionRequest struct {
	ResponseId string `json:"response_id" doc:"The response to carry on from. It is kept; everything after it is not."`
}

// SayRequest is the SayRequest schema.
type SayRequest struct {
	Text string `json:"text"`
}

// SessionSettingsRequest What to change about one running session's models. A field left out is left as it is. The same safe knobs as ModelOverwrites, plus the voice.
type SessionSettingsRequest struct {
	Llm             *string                          `json:"llm,omitempty" doc:"The conversation model, a provider/model or a capability shortcut."`
	MaxOutputTokens *int                             `json:"max_output_tokens,omitempty"`
	Sts             *string                          `json:"sts,omitempty" doc:"A speech-to-speech target, which makes the session native. Empty makes it a cascade again."`
	Stt             *string                          `json:"stt,omitempty"`
	Temperature     *float64                         `json:"temperature,omitempty"`
	Thinking        *SessionSettingsRequestThinking  `json:"thinking,omitempty" enum:"none,minimal,low,medium,high"`
	Tts             *string                          `json:"tts,omitempty"`
	Verbosity       *SessionSettingsRequestVerbosity `json:"verbosity,omitempty" enum:"low,medium,high"`
	Voice           *string                          `json:"voice,omitempty" doc:"The voice to speak in, in the provider's own terms. Empty returns to the provider's default."`
}

func (*SessionSettingsRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_output_tokens"].Format = ""
	schema.Description = "What to change about one running session's models. A field left out is left as it is. The same safe knobs as ModelOverwrites, plus the voice."
	return schema
}
