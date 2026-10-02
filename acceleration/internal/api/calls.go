package api

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	getstream "github.com/GetStream/getstream-go/v5"
)

// noCalls and noTranscripts are what the call paths say on a deployment that cannot
// answer them: a call is only remembered if there is somewhere to remember it, and what
// was said lives in Stream Chat rather than here.
const (
	noCalls       = "calls are not available: no database configured"
	noTranscripts = "transcripts are not available: no chat credentials configured"
	unknownCall   = "no such call"
	noStreamKeys  = "joining is not available: no stream credentials configured"
)

// listenerTokenValidity is how long a browser's token lasts. A call outliving it is a call
// nobody is still on, which is the same bet the Python examples make.
const listenerTokenValidity = time.Hour

// defaultCallType is what a call is joined as when nothing said otherwise. It matches the
// session default, which is what created the call in the first place.
const defaultCallType = "agent"

// ListCalls returns the calling customer's calls, newest first.
func (s *Server) ListCalls(ctx context.Context, request ListCallsRequestObject) (ListCallsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListCalls401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListCalls400JSONResponse{badRequest(noCalls)}, nil
	}

	filter := store.CallFilter{
		AgentID:    value(request.Params.AgentId),
		CampaignID: value(request.Params.CampaignId),
		Running:    value(request.Params.Running),
		Limit:      value(request.Params.Limit),
	}
	if request.Params.From != nil {
		filter.From = *request.Params.From
	}
	if request.Params.To != nil {
		filter.To = *request.Params.To
	}

	stored, err := s.store.CustomerCalls(ctx, customerID, filter)
	if err != nil {
		return nil, err
	}

	listed := make([]Call, 0, len(stored))
	for _, call := range stored {
		listed = append(listed, callOf(call))
	}
	return ListCalls200JSONResponse(listed), nil
}

// GetCall returns one call and whatever was made of it afterwards.
func (s *Server) GetCall(ctx context.Context, request GetCallRequestObject) (GetCallResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetCall401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetCall400JSONResponse{badRequest(noCalls)}, nil
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return GetCall404JSONResponse{NotFoundJSONResponse{Error: unknownCall}}, nil
	}
	rendered := callOf(call)
	s.attachUsed(ctx, customerID, call, &rendered)
	s.attachUsage(ctx, customerID, call, &rendered)
	return GetCall200JSONResponse(rendered), nil
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

// CreateCallToken mints what a browser needs to join a call and talk to the agent.
//
// The token is signed here rather than fetched, so this makes no network calls, and the
// user is not registered either: the coordinator does that when the browser connects. The
// call type comes from the running session when there is one, because only the session
// knows what it joined as.
func (s *Server) CreateCallToken(ctx context.Context, request CreateCallTokenRequestObject) (CreateCallTokenResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateCallToken401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateCallToken400JSONResponse{badRequest(noCalls)}, nil
	}
	if s.stream == nil {
		return CreateCallToken400JSONResponse{badRequest(noStreamKeys)}, nil
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return CreateCallToken404JSONResponse{NotFoundJSONResponse{Error: unknownCall}}, nil
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

	// The token is for the app the agent joined the call in, which the running session
	// knows best and the call row remembers after it.
	callType, pin := defaultCallType, call.StreamAppPK
	if s.sessions != nil {
		if found, running := s.sessions.Get(call.ID, OwnerFrom(ctx)); running {
			callType, pin = found.Spec().CallType, found.Spec().StreamApp
		}
	}
	bound, err := s.streamForApp(ctx, customerID, pin)
	switch {
	case errors.Is(err, errNoStream):
		return CreateCallToken400JSONResponse{badRequest(noStreamKeys)}, nil
	case elsewhere(err):
		return CreateCallToken400JSONResponse{badRequest(callElsewhere)}, nil
	case err != nil:
		return nil, err
	}

	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := bound.Client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, err
	}

	return CreateCallToken200JSONResponse{
		ApiKey:    bound.Identity.APIKey,
		Token:     token,
		UserId:    userID,
		UserName:  userName,
		CallId:    call.CallID,
		CallType:  callType,
		ExpiresAt: expiresAt,
	}, nil
}

// CreateChatToken mints what a browser needs to read an agent's conversation.
//
// The transcript is already a Stream Chat channel, so a client that can reach it needs no
// transcript API and sees a reply while it is still being written. Reading it means being
// in it: the reader is added to the channel here, because a token alone opens nothing.
func (s *Server) CreateChatToken(ctx context.Context, request CreateChatTokenRequestObject) (CreateChatTokenResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateChatToken401JSONResponse{missingCustomer()}, nil
	}
	if s.stream == nil {
		return CreateChatToken400JSONResponse{badRequest(noStreamKeys)}, nil
	}
	if request.Body == nil {
		return CreateChatToken400JSONResponse{badRequest("a request body is required")}, nil
	}

	agentID := strings.TrimSpace(request.Body.AgentId)
	if agentID == "" {
		return CreateChatToken400JSONResponse{badRequest("an agent id is required, since it names the channel")}, nil
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

	bound, err := s.agentStream(ctx, customerID, agentID)
	switch {
	case errors.Is(err, errNoStream):
		return CreateChatToken400JSONResponse{badRequest(noStreamKeys)}, nil
	case elsewhere(err):
		return CreateChatToken400JSONResponse{badRequest("that agent's conversation is kept in a Stream app this customer no longer acts in")}, nil
	case err != nil:
		return nil, err
	}
	client := bound.Client
	if err := conversation.CreateMissingUsers(ctx, client, map[string]getstream.UserRequest{
		agentID: {ID: agentID},
		userID:  {ID: userID, Name: &userName},
	}); err != nil {
		return nil, err
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
		return nil, err
	}

	// Members in creation data are ignored when the channel already exists.
	// Add the reader explicitly before minting a token for a members-only channel.
	if _, err := client.Chat().UpdateChannel(ctx, chatlog.ChannelType, agentID,
		&getstream.UpdateChannelRequest{AddMembers: []getstream.ChannelMemberRequest{{UserID: userID}}}); err != nil {
		return nil, err
	}

	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, err
	}

	return CreateChatToken200JSONResponse{
		ApiKey:      bound.Identity.APIKey,
		Token:       token,
		UserId:      userID,
		UserName:    userName,
		ChannelType: chatlog.ChannelType,
		ChannelId:   agentID,
		ExpiresAt:   expiresAt,
	}, nil
}

// GetCallTranscript returns what was said, read back out of the channel it was written to
// while the call was happening.
func (s *Server) GetCallTranscript(ctx context.Context, request GetCallTranscriptRequestObject) (GetCallTranscriptResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetCallTranscript401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetCallTranscript400JSONResponse{badRequest(noCalls)}, nil
	}
	if s.stream == nil {
		return GetCallTranscript400JSONResponse{badRequest(noTranscripts)}, nil
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return GetCallTranscript404JSONResponse{NotFoundJSONResponse{Error: unknownCall}}, nil
	}

	said, err := s.transcriptOf(ctx, customerID, call)
	if errors.Is(err, errNoStream) {
		return GetCallTranscript400JSONResponse{badRequest(noTranscripts)}, nil
	}
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
	return GetCallTranscript200JSONResponse(messages), nil
}

// streamFor is the Stream app a customer's new work is done in. A deployment with no
// Stream app, or no app for this customer, is not an error: it is ok false, and the path
// says what it cannot do.
func (s *Server) streamFor(ctx context.Context, customerID string) (streamapp.Bound, bool, error) {
	if s.stream == nil {
		return streamapp.Bound{}, false, nil
	}
	bound, err := s.stream.For(ctx, customerID)
	if errors.Is(err, streamapp.ErrNoIdentity) {
		return streamapp.Bound{}, false, nil
	}
	if err != nil {
		return streamapp.Bound{}, false, err
	}
	return bound, true, nil
}

// errNoStream is a deployment with no Stream app, or none for the customer.
var errNoStream = errors.New("api: no Stream app is configured for this customer")

// streamForApp is the Stream app work already pinned to one is finished in: the app the
// call or session was made in, wherever its customer acts now. errNoStream says there is no
// Stream at all; streamapp's errors say the pinned app is not one this customer can act in.
func (s *Server) streamForApp(ctx context.Context, customerID string, app int64) (streamapp.Bound, error) {
	if s.stream == nil {
		return streamapp.Bound{}, errNoStream
	}
	bound, err := s.stream.ForApp(ctx, customerID, app)
	if errors.Is(err, streamapp.ErrNoIdentity) {
		return streamapp.Bound{}, errNoStream
	}
	return bound, err
}

// elsewhere reports whether a pinned app is one the customer cannot act in from here:
// moved away from, disconnected, or never theirs.
func elsewhere(err error) bool {
	return errors.Is(err, streamapp.ErrStreamAppMoved) || errors.Is(err, streamapp.ErrStreamAppDisconnected)
}

// callElsewhere is what a call made in an app the customer no longer acts in answers.
const callElsewhere = "that call was made in a Stream app this customer no longer acts in"

// transcriptOf is what was said on a call, read in the app the call was made in. A call
// made in an app the customer no longer acts in has nothing readable from here.
func (s *Server) transcriptOf(ctx context.Context, customerID string, call store.Call) ([]chatlog.Spoken, error) {
	bound, err := s.streamForApp(ctx, customerID, call.StreamAppPK)
	if elsewhere(err) {
		return []chatlog.Spoken{}, nil
	}
	if err != nil {
		return nil, err
	}
	return chatlog.NewReaderFromClient(bound.Client).Transcript(ctx, s.transcriptRead(ctx, customerID, call))
}

// agentStream is the app an agent's channel is in: the one its most recent session was
// made in, or for an agent nobody has spoken to yet, the customer's own.
func (s *Server) agentStream(ctx context.Context, customerID, agentID string) (streamapp.Bound, error) {
	if s.store != nil {
		latest, err := s.store.QuerySessions(ctx, customerID, store.SessionFilter{AgentID: agentID, Limit: 1})
		if err != nil {
			return streamapp.Bound{}, err
		}
		if len(latest) == 1 {
			return s.streamForApp(ctx, customerID, latest[0].StreamAppPK)
		}
	}
	bound, ok, err := s.streamFor(ctx, customerID)
	if err == nil && !ok {
		return streamapp.Bound{}, errNoStream
	}
	return bound, err
}

// transcriptSlack widens a call's window by the time the router's clock and Stream's may
// disagree, so the first and last lines are not lost to a timestamp a moment off.
const transcriptSlack = 2 * time.Second

// transcriptRead is where a call's transcript was written. A call bound to a conversation
// wrote into that conversation's channel, beside what was typed before and after it, so
// only the call's own window is read. A call with no session row read its agent's channel
// before there were session rows, and still does.
func (s *Server) transcriptRead(ctx context.Context, customerID string, call store.Call) chatlog.Read {
	channel := call.AgentID
	if stored, err := s.store.StoredSession(ctx, customerID, call.ID); err == nil {
		channel = transcriptChannel(stored)
	}
	read := chatlog.Read{
		Channel:  channel,
		Customer: customerID,
		Agent:    call.AgentID,
		From:     call.StartedAt.Add(-transcriptSlack),
	}
	if call.EndedAt != nil {
		read.To = call.EndedAt.Add(transcriptSlack)
	}
	return read
}

// GetCallEvents returns what the conversation decided on one call, oldest first.
func (s *Server) GetCallEvents(ctx context.Context, request GetCallEventsRequestObject) (GetCallEventsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetCallEvents401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetCallEvents400JSONResponse{badRequest(noCalls)}, nil
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return GetCallEvents404JSONResponse{NotFoundJSONResponse{Error: unknownCall}}, nil
	}

	stored, err := s.store.CallEvents(
		ctx, customerID, call.CallID, call.StartedAt, call.EndedAt, value(request.Params.Limit))
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
	return GetCallEvents200JSONResponse(decisions), nil
}

// GetCallTimeline returns the call as it unfolded: each exchange with what was said in it
// and what the caller waited for it.
func (s *Server) GetCallTimeline(ctx context.Context, request GetCallTimelineRequestObject) (GetCallTimelineResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetCallTimeline401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetCallTimeline400JSONResponse{badRequest(noCalls)}, nil
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return GetCallTimeline404JSONResponse{NotFoundJSONResponse{Error: unknownCall}}, nil
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
	said, err := s.transcriptOf(ctx, customerID, call)
	if err != nil && !errors.Is(err, errNoStream) {
		s.logger.Error("could not read the transcript for a timeline", "call", call.ID, "error", err)
	}

	return GetCallTimeline200JSONResponse(timelineOf(turns, said, models)), nil
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
	rendered.Subagent = optional(call.Subagent)
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
			rendered.Subagent = optional(spec.SubagentTarget)
			asked, voiceUsed := found.Voice()
			rendered.Voice = optional(asked)
			rendered.VoiceUsed = optional(voiceUsed)
			mode := SessionMode(found.Mode())
			rendered.Mode = &mode
			stt, llm, tts, subagent := found.Resolved()
			rendered.SttUsed = optional(stt)
			rendered.LlmUsed = optional(llm)
			rendered.TtsUsed = optional(tts)
			rendered.SubagentUsed = optional(subagent)
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
	rendered.SubagentUsed = firstUsed(rendered.SubagentUsed, matchUsed(value(rendered.Subagent), namesOf(used, "llm"), s.candidateNames(ctx, routing.LLM, value(rendered.Subagent))))
}

func filledUsed(call *Call) bool {
	if value(call.Sts) != "" {
		return call.StsUsed != nil && (value(call.Subagent) == "" || call.SubagentUsed != nil)
	}
	return call.SttUsed != nil && call.TtsUsed != nil && call.LlmUsed != nil && call.SubagentUsed != nil
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
