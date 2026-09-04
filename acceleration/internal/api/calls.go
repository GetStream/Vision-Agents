package api

import (
	"context"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
	return GetCall200JSONResponse(rendered), nil
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
	if s.streamKey == "" || s.streamSecret == "" {
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

	callType := defaultCallType
	if s.sessions != nil {
		if found, running := s.sessions.Get(call.ID, customerID); running {
			callType = found.Spec().CallType
		}
	}

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return nil, err
	}
	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, err
	}

	return CreateCallToken200JSONResponse{
		ApiKey:    s.streamKey,
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
	if s.streamKey == "" || s.streamSecret == "" {
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

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return nil, err
	}

	if _, err := client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{
		Users: map[string]getstream.UserRequest{
			userID: {ID: userID, Name: &userName},
		},
	}); err != nil {
		return nil, err
	}

	// The channel is created by whoever holds the conversation, which for an agent nobody
	// has spoken to yet is nobody. Creating it here means a reader can watch it before
	// the first word rather than polling until it exists.
	if _, err := client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, agentID,
		&getstream.GetOrCreateChannelRequest{
			Data: &getstream.ChannelInput{
				CreatedByID: &userID,
				Members:     []getstream.ChannelMemberRequest{{UserID: userID}},
			},
		}); err != nil {
		return nil, err
	}

	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, err
	}

	return CreateChatToken200JSONResponse{
		ApiKey:      s.streamKey,
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
	if s.transcripts == nil {
		return GetCallTranscript400JSONResponse{badRequest(noTranscripts)}, nil
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return GetCallTranscript404JSONResponse{NotFoundJSONResponse{Error: unknownCall}}, nil
	}

	said, err := s.transcripts.Transcript(ctx, call.AgentID)
	if err != nil {
		return nil, err
	}

	messages := make([]TranscriptMessage, 0, len(said))
	for _, line := range said {
		messages = append(messages, TranscriptMessage{
			Speaker:   line.Speaker,
			Text:      line.Text,
			CreatedAt: line.At,
		})
	}
	return GetCallTranscript200JSONResponse(messages), nil
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

	return GetCallTimeline200JSONResponse(timelineOf(turns, said)), nil
}

// timelineOf pairs each exchange with the lines said during it.
//
// A turn starts when the transcript settled, and the messages are written as the call
// goes, so a line belongs to the last turn that had started when it was stored. Within a
// turn the first line is what the caller said and the last is what the agent answered,
// which is what a two-line exchange always is.
func timelineOf(turns []store.Turn, said []chatlog.Spoken) []TimelineEntry {
	timeline := make([]TimelineEntry, 0, len(turns))
	for index, turn := range turns {
		entry := TimelineEntry{
			TurnId:      turn.TurnID,
			StartedAt:   turn.StartedAt,
			RoundtripMs: turn.RoundtripMs,
			AudioOutMs:  turn.AudioOutMs,
			Interrupted: &turn.Interrupted,
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
	rendered.FromNumber = optional(call.FromNumber)
	rendered.ToNumber = optional(call.ToNumber)
	rendered.Stt = optional(call.STT)
	rendered.Tts = optional(call.TTS)
	rendered.Llm = optional(call.LLM)
	rendered.Subagent = optional(call.Subagent)
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
		if found, ok := s.sessions.Get(call.ID, customerID); ok {
			stt, llm, tts, subagent := found.Resolved()
			rendered.SttUsed = optional(stt)
			rendered.LlmUsed = optional(llm)
			rendered.TtsUsed = optional(tts)
			rendered.SubagentUsed = optional(subagent)
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

	rendered.SttUsed = firstUsed(rendered.SttUsed, matchUsed(value(rendered.Stt), namesOf(used, "stt"), s.candidateNames(ctx, routing.STT, value(rendered.Stt))))
	rendered.TtsUsed = firstUsed(rendered.TtsUsed, matchUsed(value(rendered.Tts), namesOf(used, "tts"), s.candidateNames(ctx, routing.TTS, value(rendered.Tts))))
	rendered.LlmUsed = firstUsed(rendered.LlmUsed, matchUsed(value(rendered.Llm), namesOf(used, "llm"), s.candidateNames(ctx, routing.LLM, value(rendered.Llm))))
	rendered.SubagentUsed = firstUsed(rendered.SubagentUsed, matchUsed(value(rendered.Subagent), namesOf(used, "llm"), s.candidateNames(ctx, routing.LLM, value(rendered.Subagent))))
}

func filledUsed(call *Call) bool {
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
