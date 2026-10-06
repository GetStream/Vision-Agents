package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/danielgtaylor/huma/v2"
)

// noCalls and noTranscripts are what the call paths say on a deployment that cannot
// answer them: a call is only remembered if there is somewhere to remember it, and what
// was said lives in Stream Chat rather than here.
var (
	noCalls       = notConfigured("calls are not available: no database configured")
	noTranscripts = notConfigured("transcripts are not available: no chat credentials configured")
	unknownCall   = APIError{Type: ErrorTypeNotFound, Code: codeCallNotFound, Message: "no such call"}
	noStreamKeys  = notConfigured("joining is not available: no stream credentials configured")
)

// listenerTokenValidity is how long a browser's token lasts. A call outliving it is a call
// nobody is still on, which is the same bet the Python examples make.
const listenerTokenValidity = time.Hour

// defaultCallType is what a call is joined as when nothing said otherwise. It matches the
// session default, which is what created the call in the first place.
const defaultCallType = "agent"

// listCalls returns the calling customer's calls, newest first.
func (s *Server) listCalls(ctx context.Context, request *listCallsRequest) (*listCallsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noCalls
	}

	filter := store.CallFilter{
		AgentID:    value(request.AgentId.ptr()),
		CampaignID: value(request.CampaignId.ptr()),
		Running:    value(request.Running.ptr()),
		Limit:      value(request.Limit.ptr()),
	}
	if request.From.ptr() != nil {
		filter.From = *request.From.ptr()
	}
	if request.To.ptr() != nil {
		filter.To = *request.To.ptr()
	}

	stored, err := s.store.CustomerCalls(ctx, customerID, filter)
	if err != nil {
		return nil, err
	}

	listed := make([]Call, 0, len(stored))
	for _, call := range stored {
		listed = append(listed, callOf(call))
	}
	return &listCallsResponse{Body: listed}, nil
}

// getCall returns one call and whatever was made of it afterwards.
func (s *Server) getCall(ctx context.Context, request *getCallRequest) (*getCallResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noCalls
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownCall
	}
	rendered := callOf(call)
	s.attachUsed(ctx, customerID, call, &rendered)
	s.attachUsage(ctx, customerID, call, &rendered)
	return &getCallResponse{Body: rendered}, nil
}

// attachUsage totals what the call spent, once there is a total to give. A running call is
// left without one: the sum would be read again on every poll and be wrong by a turn each
// time, and what a conversation cost is a question asked after it, not during.
func (s *Server) attachUsage(ctx context.Context, customerID string, call store.Call, rendered *Call) {
	if call.EndedAt == nil || s.store == nil {
		return
	}

	spent, err := s.store.CallUsage(ctx, customerID, call.AgentID, call.StartedAt, call.EndedAt)
	if err != nil {
		s.logger.Error("could not read what a call spent", "call", call.ID, "error", err)
		return
	}
	rendered.Usage = &CallUsage{
		InputTokens:       spent.InputTokens,
		CachedInputTokens: spent.CachedInputTokens,
		OutputTokens:      spent.OutputTokens,
		CostMicros:        spent.CostMicros,
		Requests:          spent.Requests,
	}
}

// createCallToken mints what a browser needs to join a call and talk to the agent.
//
// The token is signed here rather than fetched, so this makes no network calls, and the
// user is not registered either: the coordinator does that when the browser connects. The
// call type comes from the running session when there is one, because only the session
// knows what it joined as.
func (s *Server) createCallToken(ctx context.Context, request *createCallTokenRequest) (*createCallTokenResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noCalls
	}
	if s.streamKey == "" || s.streamSecret == "" {
		return nil, noStreamKeys
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownCall
	}

	var wanted CallTokenRequest
	if request.Body != nil {
		wanted = *request.Body
	}
	userID := value(wanted.UserId)
	if userID == "" {
		userID = "listener-" + call.ID
	}
	userName := value(wanted.UserName)
	if userName == "" {
		userName = userID
	}

	callType := defaultCallType
	if s.sessions != nil {
		if found, running := s.sessions.Get(call.ID, OwnerFrom(ctx)); running {
			callType = found.Spec().CallType
		}
	}

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, stack.Wrap(err)
	}

	return &createCallTokenResponse{Body: CallToken{ApiKey: s.streamKey,
		Token:     token,
		UserId:    userID,
		UserName:  userName,
		CallId:    call.CallID,
		CallType:  callType,
		ExpiresAt: expiresAt}}, nil
}

// createChatToken mints what a browser needs to read an agent's conversation.
//
// The transcript is already a Stream Chat channel, so a client that can reach it needs no
// transcript API and sees a reply while it is still being written. Reading it means being
// in it: the reader is added to the channel here, because a token alone opens nothing.
func (s *Server) createChatToken(ctx context.Context, request *createChatTokenRequest) (*createChatTokenResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.streamKey == "" || s.streamSecret == "" {
		return nil, noStreamKeys
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	agentID := strings.TrimSpace(request.Body.AgentId)
	if agentID == "" {
		return nil, invalidRequest("an agent id is required, since it names the channel")
	}

	userID := value(request.Body.UserId)
	if userID == "" {
		// Somebody reading a conversation is not the agent, and two readers of the same
		// one are not each other, which is why this is per customer rather than shared.
		userID = "reader-" + customerID
	}
	userName := value(request.Body.UserName)
	if userName == "" {
		userName = userID
	}

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return nil, stack.Wrap(err)
	}

	if _, err := client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{
		Users: map[string]getstream.UserRequest{
			agentID: {ID: agentID},
			userID:  {ID: userID, Name: &userName},
		},
	}); err != nil {
		return nil, stack.Wrap(err)
	}

	// The channel is created by whoever holds the conversation, which for an agent nobody
	// has spoken to yet is nobody. Creating it here means a reader can watch it before
	// the first word rather than polling until it exists.
	if _, err := client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, agentID,
		&getstream.GetOrCreateChannelRequest{
			Data: &getstream.ChannelInput{
				CreatedByID: &agentID,
				Members:     []getstream.ChannelMemberRequest{{UserID: userID}},
			},
		}); err != nil {
		return nil, stack.Wrap(err)
	}

	// Members in creation data are ignored when the channel already exists.
	// Add the reader explicitly before minting a token for a members-only channel.
	if _, err := client.Chat().UpdateChannel(ctx, chatlog.ChannelType, agentID,
		&getstream.UpdateChannelRequest{AddMembers: []getstream.ChannelMemberRequest{{UserID: userID}}}); err != nil {
		return nil, stack.Wrap(err)
	}

	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, stack.Wrap(err)
	}

	return &createChatTokenResponse{Body: ChatToken{ApiKey: s.streamKey,
		Token:       token,
		UserId:      userID,
		UserName:    userName,
		ChannelType: chatlog.ChannelType,
		ChannelId:   agentID,
		ExpiresAt:   expiresAt}}, nil
}

// getCallTranscript returns what was said, read back out of the channel it was written to
// while the call was happening.
func (s *Server) getCallTranscript(ctx context.Context, request *getCallTranscriptRequest) (*getCallTranscriptResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noCalls
	}
	if s.transcripts == nil {
		return nil, noTranscripts
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownCall
	}

	said, err := s.transcripts.Transcript(ctx, call.AgentID)
	if err != nil {
		return nil, err
	}

	messages := make([]TranscriptMessage, 0, len(said))
	for _, line := range said {
		messages = append(messages, TranscriptMessage{
			Speaker:   line.Speaker,
			Name:      optional(line.Name),
			Agent:     &line.Agent,
			Text:      line.Text,
			CreatedAt: line.At,
		})
	}
	return &getCallTranscriptResponse{Body: messages}, nil
}

// getCallEvents returns what the conversation decided on one call, oldest first.
func (s *Server) getCallEvents(ctx context.Context, request *getCallEventsRequest) (*getCallEventsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noCalls
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownCall
	}

	stored, err := s.store.CallEvents(
		ctx, customerID, call.CallID, call.StartedAt, call.EndedAt, value(request.Limit.ptr()))
	if err != nil {
		return nil, err
	}

	decisions := make([]CallEvent, 0, len(stored))
	for _, decided := range stored {
		decisions = append(decisions, CallEvent{
			At:          decided.At,
			Kind:        DecisionKind(decided.Kind),
			Reason:      decided.Reason,
			TurnId:      optional(decided.TurnID),
			Participant: optional(decided.Participant),
			Said:        optional(decided.Said),
			LatencyMs:   decided.LatencyMs,
		})
	}
	return &getCallEventsResponse{Body: decisions}, nil
}

// getCallTimeline returns the call as it unfolded: each exchange with what was said in it
// and what the caller waited for it.
func (s *Server) getCallTimeline(ctx context.Context, request *getCallTimelineRequest) (*getCallTimelineResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noCalls
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownCall
	}

	turns, err := s.store.CallTurns(ctx, customerID, call.AgentID, call.StartedAt, call.EndedAt)
	if err != nil {
		return nil, err
	}
	models, err := s.store.CallModelCalls(ctx, customerID, call.AgentID, call.CallID, call.StartedAt, call.EndedAt)
	if err != nil {
		return nil, err
	}

	// The transcript is worth having but not worth failing over: the timings are the
	// part of this view that only this service holds.
	var said []chatlog.Spoken
	if s.transcripts != nil {
		said, err = s.transcripts.Transcript(ctx, call.AgentID)
		if err != nil {
			s.logger.Error("could not read the transcript for a timeline",
				"call", call.ID, "error", err)
		}
	}

	return &getCallTimelineResponse{Body: timelineOf(turns, said, models)}, nil
}

// timelineOf pairs each exchange with the lines said during it.
//
// A turn starts when the transcript settled, and the messages are written as the call
// goes, so a line belongs to the last turn that had started when it was stored. Within a
// turn the first line is what the caller said and the last is what the agent answered,
// which is what a two-line exchange always is.
func timelineOf(turns []store.Turn, said []chatlog.Spoken, models []store.Request) []TimelineEntry {
	modelCalls := make(map[string][]ModelCallTiming)
	for _, request := range models {
		if request.TurnID == "" {
			continue
		}
		modelCalls[request.TurnID] = append(modelCalls[request.TurnID], ModelCallTiming{
			OperationId:  optional(request.OperationID),
			Purpose:      optional(request.Purpose),
			StartedAt:    request.StartedAt,
			Provider:     request.Provider,
			Model:        request.Model,
			TtftMs:       request.LatencyMs,
			DurationMs:   request.DurationMs,
			InputTokens:  &request.InputTokens,
			OutputTokens: &request.OutputTokens,
			Success:      request.Success,
		})
	}
	timeline := make([]TimelineEntry, 0, len(turns))
	for index, turn := range turns {
		entry := TimelineEntry{
			TurnId:             turn.TurnID,
			StartedAt:          turn.StartedAt,
			CadenceMs:          turn.CadenceMs,
			DecisionMs:         turn.DecisionMs,
			ModelToFirstTextMs: turn.ModelToFirstTextMs,
			TextToTtsMs:        turn.TextToTTSMs,
			TtsToAudioMs:       turn.TTSToAudioMs,
			RoundtripMs:        turn.RoundtripMs,
			SttLatencyMs:       turn.STTLatencyMs,
			LlmTtftMs:          turn.LLMTTFTMs,
			TtsTtfbMs:          turn.TTSTTFBMs,
			SpeechEndToAudioMs: turn.SpeechEndToAudioMs,
			AudioOutMs:         turn.AudioOutMs,
			Interrupted:        &turn.Interrupted,
		}
		if calls := modelCalls[turn.TurnID]; len(calls) > 0 {
			entry.ModelCalls = &calls
		}

		var until time.Time
		if index+1 < len(turns) {
			until = turns[index+1].StartedAt
		}
		if spoken := within(said, turn.StartedAt, until); len(spoken) > 0 {
			entry.Heard = optional(spoken[0].Text)
			if len(spoken) > 1 {
				entry.Said = optional(spoken[len(spoken)-1].Text)
			}
		}
		timeline = append(timeline, entry)
	}
	return timeline
}

// within returns the lines stored in a half-open window. A zero end is the rest of them.
func within(said []chatlog.Spoken, from, until time.Time) []chatlog.Spoken {
	var inside []chatlog.Spoken
	for _, line := range said {
		if line.At.Before(from) {
			continue
		}
		if !until.IsZero() && !line.At.Before(until) {
			break
		}
		inside = append(inside, line)
	}
	return inside
}

// callOf renders a call for the wire.
func callOf(call store.Call) Call {
	rendered := Call{
		Id:        call.ID,
		CallId:    call.CallID,
		AgentId:   call.AgentID,
		Direction: CallDirection(call.Direction),
		StartedAt: call.StartedAt,
		EndedAt:   call.EndedAt,
	}
	rendered.ConfigId = optional(call.ConfigID)
	rendered.CampaignId = optional(call.CampaignID)
	rendered.ContactId = optional(call.ContactID)
	rendered.UserId = optional(call.UserID)
	rendered.FromNumber = optional(call.FromNumber)
	rendered.ToNumber = optional(call.ToNumber)
	rendered.Stt = optional(call.STT)
	rendered.Tts = optional(call.TTS)
	rendered.Sts = optional(call.STS)
	rendered.Llm = optional(call.LLM)
	rendered.ThinkingLlm = optional(call.Subagent)
	rendered.Voice = optional(call.Voice)
	mode := SessionModeCascade
	if call.STS != "" {
		mode = SessionModeNative
	}
	rendered.Mode = &mode
	rendered.Instructions = optional(call.Instructions)
	rendered.Summary = optional(call.Summary)
	rendered.ReviewNotes = optional(call.ReviewNotes)
	rendered.ReviewScore = call.ReviewScore
	if len(call.Skills) > 0 {
		skills := call.Skills
		rendered.Skills = &skills
	}
	if len(call.Tags) > 0 {
		tags := call.Tags
		rendered.Tags = &tags
	}
	return rendered
}

// attachUsed fills in the provider/models routing picked, which a shortcut does not name.
//
// A live session knows the current selection before any request row has been written. A
// finished call has only the request rows, and those also cover a live call after the
// first turn. Failover can leave more than one model per modality; the one that still
// satisfies the target is the one shown.
func (s *Server) attachUsed(ctx context.Context, customerID string, call store.Call, rendered *Call) {
	if s.sessions != nil {
		if found, ok := s.sessions.Get(call.ID, OwnerFrom(ctx)); ok {
			// The row is written off the request path, so what a running session is on
			// now is read from it rather than from a row a swap may not have reached yet.
			spec := found.Spec()
			rendered.Stt = optional(spec.STTTarget)
			rendered.Tts = optional(spec.TTSTarget)
			rendered.Llm = optional(spec.LLMTarget)
			rendered.Sts = optional(spec.STSTarget)
			rendered.ThinkingLlm = optional(spec.SubagentTarget)
			asked, voiceUsed := found.Voice()
			rendered.Voice = optional(asked)
			rendered.VoiceUsed = optional(voiceUsed)
			mode := SessionMode(found.Mode())
			rendered.Mode = &mode
			stt, llm, tts, subagent := found.Resolved()
			rendered.SttUsed = optional(stt)
			rendered.LlmUsed = optional(llm)
			rendered.TtsUsed = optional(tts)
			rendered.ThinkingLlmUsed = optional(subagent)
			rendered.StsUsed = optional(found.Speech())
		}
	}

	if filledUsed(rendered) || s.store == nil {
		return
	}

	used, err := s.store.CallUsedModels(ctx, customerID, call.AgentID, call.StartedAt, call.EndedAt)
	if err != nil {
		s.logger.Error("could not read the models a call used", "call", call.ID, "error", err)
		return
	}

	if value(rendered.Sts) != "" {
		rendered.StsUsed = firstUsed(rendered.StsUsed, matchUsed(value(rendered.Sts), namesOf(used, "sts"), s.candidateNames(ctx, routing.STS, value(rendered.Sts))))
	} else {
		rendered.SttUsed = firstUsed(rendered.SttUsed, matchUsed(value(rendered.Stt), namesOf(used, "stt"), s.candidateNames(ctx, routing.STT, value(rendered.Stt))))
		rendered.TtsUsed = firstUsed(rendered.TtsUsed, matchUsed(value(rendered.Tts), namesOf(used, "tts"), s.candidateNames(ctx, routing.TTS, value(rendered.Tts))))
		rendered.LlmUsed = firstUsed(rendered.LlmUsed, matchUsed(value(rendered.Llm), namesOf(used, "llm"), s.candidateNames(ctx, routing.LLM, value(rendered.Llm))))
	}
	rendered.ThinkingLlmUsed = firstUsed(rendered.ThinkingLlmUsed, matchUsed(value(rendered.ThinkingLlm), namesOf(used, "llm"), s.candidateNames(ctx, routing.LLM, value(rendered.ThinkingLlm))))
}

func filledUsed(call *Call) bool {
	if value(call.Sts) != "" {
		return call.StsUsed != nil && (value(call.ThinkingLlm) == "" || call.ThinkingLlmUsed != nil)
	}
	return call.SttUsed != nil && call.TtsUsed != nil && call.LlmUsed != nil && call.ThinkingLlmUsed != nil
}

func firstUsed(existing *string, used string) *string {
	if existing != nil {
		return existing
	}
	return optional(used)
}

func namesOf(used []store.UsedModel, modality string) []string {
	var names []string
	for _, model := range used {
		if model.Modality == modality {
			names = append(names, model.Provider+"/"+model.Model)
		}
	}
	return names
}

// matchUsed picks which of the used provider/models served a target. used is most-recent
// first. candidates are the names that target can resolve to; when they are known only a
// used model that is still a candidate is returned, which is how the voice model and the
// thinking model are told apart when both wrote llm rows.
func matchUsed(asked string, used []string, candidates []string) string {
	if asked == "" || len(used) == 0 {
		return ""
	}
	if len(candidates) > 0 {
		allowed := make(map[string]struct{}, len(candidates))
		for _, name := range candidates {
			allowed[name] = struct{}{}
		}
		for _, name := range used {
			if _, ok := allowed[name]; ok {
				return name
			}
		}
		return ""
	}
	for _, name := range used {
		if name == asked {
			return name
		}
	}
	if strings.Contains(asked, "/") {
		return ""
	}
	return used[0]
}

func (s *Server) candidateNames(ctx context.Context, modality routing.Modality, target string) []string {
	if target == "" {
		return nil
	}
	router, ok := s.routers[modality]
	if !ok {
		return nil
	}
	candidates, err := router.Resolve(ctx, target, nil)
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		names = append(names, candidate.Config.Name())
	}
	return names
}

// registerCalls declares the operations served in calls.go.
func (s *Server) registerCalls(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listCalls",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls",
		Summary:     "The calls the calling customer has run",
		Description: "A session lives in memory and is gone when the process is, so a call is recorded as it " +
			"starts and again as it ends. This is what answers what happened yesterday, and what is " +
			"happening now after a restart.",
		// Declared rather than read off the input, so the default is documented without
		// being filled in: the handler tells a parameter left out from one sent.
		Parameters: []*huma.Param{
			{Name: "running", In: "query", Description: "Only calls that have not ended.", Schema: &huma.Schema{Type: huma.TypeBoolean, Default: false}},
			{Name: "limit", In: "query", Schema: &huma.Schema{Type: huma.TypeInteger, Default: 50}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's calls, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listCalls)
	huma.Register(api, huma.Operation{
		OperationID: "getCall",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}",
		Summary:     "One call, with whatever was made of it afterwards",
		Responses: map[string]*huma.Response{
			"200": {Description: "The call"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCall)
	huma.Register(api, huma.Operation{
		OperationID: "getCallTranscript",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}/transcript",
		Summary:     "What was said on a call",
		Description: "Read back from the chat channel the conversation was written to as it happened, rather " +
			"than copied into a second place that could disagree with it.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The conversation, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCallTranscript)
	huma.Register(api, huma.Operation{
		OperationID: "getCallTimeline",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}/timeline",
		Summary:     "The call as it unfolded, said and measured together",
		Description: "Each exchange with what was said in it and what it cost the caller in waiting: how long " +
			"the answer took to start, how much the agent spoke, and whether it was talked over.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The exchanges, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCallTimeline)
	huma.Register(api, huma.Operation{
		OperationID: "getCallEvents",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}/events",
		Summary:     "What the conversation decided, and why",
		Description: "A timeline says what a call cost the caller in waiting. This says why the call went the " +
			"way it did: why the agent waited rather than answering, why it read something as not " +
			"meant for it, why it stopped mid-sentence. Read in order they are the reasoning behind " +
			"the conversation, which is the only thing that explains a call that surprised somebody. " +
			"A call still running reports the same decisions live on the session socket.",
		// Declared rather than read off the input, so the default is documented without
		// being filled in: the handler tells a parameter left out from one sent.
		Parameters: []*huma.Param{
			{Name: "limit", In: "query", Description: "How many to return, oldest first.", Schema: &huma.Schema{Type: huma.TypeInteger, Default: 1000}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The judgements, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCallEvents)
	huma.Register(api, huma.Operation{
		OperationID: "createCallToken",
		Method:      http.MethodPost,
		Path:        "/v1/agents/calls/{id}/token",
		Summary:     "What a browser needs to join this call",
		Description: "Mints a Stream token so a person can join the call from a browser and talk to the " +
			"agent, and says which call to join with it. The secret stays here: the browser is " +
			"handed a token that expires, never the key that signs one.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The credentials, and the call they are for"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.createCallToken)
	huma.Register(api, huma.Operation{
		OperationID: "createChatToken",
		Method:      http.MethodPost,
		Path:        "/v1/agents/chat-token",
		Summary:     "What a browser needs to read an agent's conversation",
		Description: "An agent writes what was said into the Stream Chat channel agent:{agent_id}, so a " +
			"client that can read that channel needs no transcript API. This mints the token to read " +
			"it with, and adds the reader to the channel, since a conversation they are not a member " +
			"of is one they cannot watch.\n" +
			"The secret stays here, the same as for a call token: the browser is handed something " +
			"that expires.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The credentials, and the channel they are for"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createChatToken)
}

type listCallsRequest struct {
	AgentId    optionalParam[string]    `query:"agent_id" doc:"Narrow to one agent."`
	CampaignId optionalParam[string]    `query:"campaign_id" doc:"Narrow to the calls one campaign placed."`
	Running    optionalParam[bool]      `query:"running" doc:"Only calls that have not ended."`
	From       optionalParam[time.Time] `query:"from" doc:"Only calls that started at or after this, inclusive."`
	To         optionalParam[time.Time] `query:"to" doc:"Only calls that started before this, exclusive."`
	Limit      optionalParam[int]       `query:"limit"`
}

type listCallsResponse struct {
	Body []Call `nullable:"false"`
}

type getCallRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getCallResponse struct {
	Body Call
}

type getCallTranscriptRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getCallTranscriptResponse struct {
	Body []TranscriptMessage `nullable:"false"`
}

type getCallTimelineRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getCallTimelineResponse struct {
	Body []TimelineEntry `nullable:"false"`
}

type getCallEventsRequest struct {
	Id    string             `path:"id" doc:"The resource, as returned when it was created."`
	Limit optionalParam[int] `query:"limit" doc:"How many to return, oldest first."`
}

type getCallEventsResponse struct {
	Body []CallEvent `nullable:"false"`
}

type createCallTokenRequest struct {
	Id   string `path:"id" doc:"The resource, as returned when it was created."`
	Body *CallTokenRequest
}

type createCallTokenResponse struct {
	Body CallToken
}

type createChatTokenRequest struct {
	Body *ChatTokenRequest `required:"true"`
}

type createChatTokenResponse struct {
	Body ChatToken
}

// CallDirection is the CallDirection schema.
type CallDirection string

// Defines values for CallDirection.
const (
	Inbound  CallDirection = "inbound"
	Outbound CallDirection = "outbound"
)

// Valid indicates whether the value is a known member of the CallDirection enum.
func (e CallDirection) Valid() bool {
	switch e {
	case Inbound:
		return true
	case Outbound:
		return true
	default:
		return false
	}
}

// CallToken is the CallToken schema.
type CallToken struct {
	ApiKey    string    `json:"api_key" doc:"The Stream app the call is in, which the browser SDK joins against."`
	CallId    string    `json:"call_id" doc:"The Stream call to join, which is not the id this call is held by here."`
	CallType  string    `json:"call_type"`
	ExpiresAt time.Time `json:"expires_at"`
	Token     string    `json:"token"`
	UserId    string    `json:"user_id"`
	UserName  string    `json:"user_name"`
}

// CallTokenRequest is the CallTokenRequest schema.
type CallTokenRequest struct {
	UserId   *string `json:"user_id,omitempty" doc:"Who the browser joins as. Somebody watching a call is not the agent, so this defaults to a listener of its own rather than to the agent's user."`
	UserName *string `json:"user_name,omitempty" doc:"The name the other participants see. Defaults to the user id."`
}

// CallUsage What the call spent, summed over every request it made. Counted once the call is over, so it is absent while one is still running. Requests that failed are included: a model that read the prompt and then fell over is still billed for it.
type CallUsage struct {
	CachedInputTokens int64 `json:"cached_input_tokens" doc:"The part of those prompts a provider served from its own cache."`
	CostMicros        int64 `json:"cost_micros" doc:"Millionths of a dollar, priced from the providers' configured rates."`
	InputTokens       int64 `json:"input_tokens" doc:"Every prompt the models read, the cached part included."`
	OutputTokens      int64 `json:"output_tokens" doc:"Everything the models generated, reasoning included."`
	Requests          int64 `json:"requests" doc:"How many calls to a model it took, transcription and speech included."`
}

func (*CallUsage) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What the call spent, summed over every request it made. Counted once the call is over, so it is absent while one is still running. Requests that failed are included: a model that read the prompt and then fell over is still billed for it."
	return schema
}

// ChatToken is the ChatToken schema.
type ChatToken struct {
	ApiKey      string    `json:"api_key" doc:"The Stream app the channel is in, which the browser SDK connects to."`
	ChannelId   string    `json:"channel_id" doc:"The channel holding the conversation, which is the agent id."`
	ChannelType string    `json:"channel_type" doc:"Always agent, which is the type a conversation is written to."`
	ExpiresAt   time.Time `json:"expires_at"`
	Token       string    `json:"token"`
	UserId      string    `json:"user_id"`
	UserName    string    `json:"user_name"`
}

// ChatTokenRequest is the ChatTokenRequest schema.
type ChatTokenRequest struct {
	AgentId  string  `json:"agent_id" doc:"Whose conversation to read. This is the session's agent id, which is what names the channel it is written to."`
	UserId   *string `json:"user_id,omitempty" doc:"Who the browser reads as. Somebody watching is not the agent, so this defaults to a reader of its own rather than to the agent's user."`
	UserName *string `json:"user_name,omitempty" doc:"The name shown against anything they write. Defaults to the user id."`
}

// DecisionKind What a conversation decided. Asking puts a settled turn to the flow controller; waiting leaves it because the caller has not finished; ignoring drops speech meant for somebody else; answering replies to it; queueing holds it until the agent has stopped talking; interrupting abandons the reply being spoken and shortening ends it early; a backchannel is a listening noise that never reaches the model; superseding drops a ruling about words that have since changed; compacting replaces old history with a summary; delegating hands work to the subagent and settling is that work coming back, answered or not.
type DecisionKind string

// Defines values for DecisionKind.
const (
	DecisionKindAnswer      DecisionKind = "answer"
	DecisionKindAsk         DecisionKind = "ask"
	DecisionKindBackchannel DecisionKind = "backchannel"
	DecisionKindCompact     DecisionKind = "compact"
	DecisionKindDelegate    DecisionKind = "delegate"
	DecisionKindFail        DecisionKind = "fail"
	DecisionKindIgnore      DecisionKind = "ignore"
	DecisionKindInterrupt   DecisionKind = "interrupt"
	DecisionKindQueue       DecisionKind = "queue"
	DecisionKindSettle      DecisionKind = "settle"
	DecisionKindShorten     DecisionKind = "shorten"
	DecisionKindSupersede   DecisionKind = "supersede"
	DecisionKindWait        DecisionKind = "wait"
)

// Valid indicates whether the value is a known member of the DecisionKind enum.
func (e DecisionKind) Valid() bool {
	switch e {
	case DecisionKindAnswer:
		return true
	case DecisionKindAsk:
		return true
	case DecisionKindBackchannel:
		return true
	case DecisionKindCompact:
		return true
	case DecisionKindDelegate:
		return true
	case DecisionKindFail:
		return true
	case DecisionKindIgnore:
		return true
	case DecisionKindInterrupt:
		return true
	case DecisionKindQueue:
		return true
	case DecisionKindSettle:
		return true
	case DecisionKindShorten:
		return true
	case DecisionKindSupersede:
		return true
	case DecisionKindWait:
		return true
	default:
		return false
	}
}

func (DecisionKind) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "DecisionKind", "What a conversation decided. Asking puts a settled turn to the flow controller; waiting leaves it because the caller has not finished; ignoring drops speech meant for somebody else; answering replies to it; queueing holds it until the agent has stopped talking; interrupting abandons the reply being spoken and shortening ends it early; a backchannel is a listening noise that never reaches the model; superseding drops a ruling about words that have since changed; compacting replaces old history with a summary; delegating hands work to the subagent and settling is that work coming back, answered or not.", "ask", "wait", "ignore", "answer", "queue", "interrupt", "shorten", "backchannel", "supersede", "compact", "delegate", "settle", "fail")
}

// ModelCallTiming is the ModelCallTiming schema.
type ModelCallTiming struct {
	DurationMs   *float64  `json:"duration_ms,omitempty" doc:"Request to completed response or failed create."`
	InputTokens  *int64    `json:"input_tokens,omitempty"`
	Model        string    `json:"model"`
	OperationId  *string   `json:"operation_id,omitempty" doc:"Response ID for this operation; retries can share an ID."`
	OutputTokens *int64    `json:"output_tokens,omitempty"`
	Provider     string    `json:"provider"`
	Purpose      *string   `json:"purpose,omitempty" doc:"reply, flow or subagent."`
	StartedAt    time.Time `json:"started_at"`
	Success      bool      `json:"success"`
	TtftMs       *float64  `json:"ttft_ms,omitempty" doc:"Request to first token."`
}

// TimelineEntry is the TimelineEntry schema.
type TimelineEntry struct {
	AudioOutMs         *float64           `json:"audio_out_ms,omitempty" doc:"How much the agent spoke."`
	CadenceMs          *float64           `json:"cadence_ms,omitempty" doc:"Last transcript revision to a stable turn ready for the flow controller."`
	DecisionMs         *float64           `json:"decision_ms,omitempty" doc:"Stable turn to the main model request, including flow and queueing."`
	Heard              *string            `json:"heard,omitempty" doc:"What the caller said, when it can be matched to this exchange."`
	Interrupted        *bool              `json:"interrupted,omitempty" doc:"Whether the caller talked over the answer."`
	LlmTtftMs          *float64           `json:"llm_ttft_ms,omitempty" doc:"The wait between asking the model and its first token." nullable:"true"`
	ModelCalls         *[]ModelCallTiming `json:"model_calls,omitempty" doc:"Individual model requests for this turn, including flow and delegated work."`
	ModelToFirstTextMs *float64           `json:"model_to_first_text_ms,omitempty" doc:"Main model request to the first text delta admitted to the voice pipeline."`
	RoundtripMs        *float64           `json:"roundtrip_ms,omitempty" doc:"Last transcript revision to first audio published; includes cadence settling."`
	Said               *string            `json:"said,omitempty" doc:"What the agent answered."`
	SpeechEndToAudioMs *float64           `json:"speech_end_to_audio_ms,omitempty" doc:"Last input audio to first output audio, estimated using provider STT processing time plus roundtrip. It excludes network transport and playback." nullable:"true"`
	StartedAt          time.Time          `json:"started_at"`
	SttLatencyMs       *float64           `json:"stt_latency_ms,omitempty" doc:"The provider's decode time for the transcript that settled the turn." nullable:"true"`
	TextToTtsMs        *float64           `json:"text_to_tts_ms,omitempty" doc:"First text delta to the first TTS request."`
	TtsToAudioMs       *float64           `json:"tts_to_audio_ms,omitempty" doc:"First TTS request to the first audio chunk published to the edge."`
	TtsTtfbMs          *float64           `json:"tts_ttfb_ms,omitempty" doc:"The wait between sending the first sentence and the first audio." nullable:"true"`
	TurnId             string             `json:"turn_id"`
}

// TranscriptMessage is the TranscriptMessage schema.
type TranscriptMessage struct {
	Agent     *bool     `json:"agent,omitempty" doc:"Whether the agent said it rather than somebody it was talking to. It is what the line was stored as, so it holds however the agent was named."`
	CreatedAt time.Time `json:"created_at"`
	Name      *string   `json:"name,omitempty" doc:"That speaker's display name, when they have one."`
	Speaker   string    `json:"speaker" doc:"Who said it, the agent under its own user id."`
	Text      string    `json:"text"`
}
