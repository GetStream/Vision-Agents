package api

import (
	"cmp"
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/danielgtaylor/huma/v2"
)

// errNoCalls and errNoTranscripts are what the call paths say on a deployment that cannot
// answer them: a call is only remembered if there is somewhere to remember it, and what
// was said lives in Stream Chat rather than here.
var (
	errNoCalls       = notConfigured("calls are not available: no database configured")
	errNoTranscripts = notConfigured("transcripts are not available: no chat credentials configured")
	errUnknownCall   = APIError{Type: ErrorTypeNotFound, Code: codeCallNotFound, Message: "no such call"}
	errNoStreamKeys  = notConfigured("joining is not available: no stream credentials configured")
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
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
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
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownCall
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

// getCallTokens says what the call's models read and wrote, and what their prompts were made of.
func (s *Server) getCallTokens(ctx context.Context, request *getCallTokensRequest) (*getCallTokensResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownCall
	}
	used, err := s.store.CallTokens(ctx, customerID, call.AgentID, call.StartedAt, call.EndedAt)
	if err != nil {
		return nil, err
	}

	spent := CallTokens{Models: make([]ModelTokens, 0, len(used))}
	for _, model := range used {
		parts := inputPartsOf(model.InputParts)
		spent.Models = append(spent.Models, ModelTokens{
			Modality: model.Modality, Provider: model.Provider, Model: model.Model,
			InputTokens: model.InputTokens, CachedInputTokens: model.CachedInputTokens,
			OutputTokens: model.OutputTokens, CostMicros: model.CostMicros, Requests: model.Requests,
			InputParts: parts,
		})
		spent.InputTokens += model.InputTokens
		spent.CachedInputTokens += model.CachedInputTokens
		spent.OutputTokens += model.OutputTokens
		spent.CostMicros += model.CostMicros
		spent.Requests += model.Requests
		spent.InputParts.Instructions += parts.Instructions
		spent.InputParts.Messages += parts.Messages
		spent.InputParts.ToolDefinitions += parts.ToolDefinitions
		spent.InputParts.ToolUse += parts.ToolUse
		spent.InputParts.Images += parts.Images
		spent.InputParts.Video += parts.Video
	}
	spent.CostSources = costSources(used)
	return &getCallTokensResponse{Body: spent}, nil
}

// costSources splits what the models cost by where it came from. A model's prompt cost is
// shared between the parts of its prompt by their tokens, and what is left over, from rows
// recorded without the split, is input.
func costSources(used []store.ModelTokens) []CostSource {
	costs := map[string]int64{}
	for _, model := range used {
		if model.InputTokens+model.OutputTokens == 0 {
			costs[model.Modality] += model.CostMicros
			continue
		}
		costs["output"] += model.OutputCostMicros
		input := model.CostMicros - model.OutputCostMicros
		if model.InputTokens == 0 {
			costs["input"] += input
			continue
		}
		parts := model.InputParts
		left, split, largest, most := input, int64(0), "input", int64(0)
		for _, part := range []struct {
			source string
			tokens int64
		}{
			{"instructions", parts.InstructionTokens}, {"messages", parts.MessageTokens},
			{"tool_definitions", parts.ToolDefinitionTokens}, {"tool_use", parts.ToolUseTokens},
			{"images", parts.ImageTokens}, {"video", parts.VideoTokens},
		} {
			share := input * part.tokens / model.InputTokens
			costs[part.source] += share
			left -= share
			split += part.tokens
			if part.tokens > most {
				largest, most = part.source, part.tokens
			}
		}
		// A prompt split whole leaves only rounding over, which is the largest part's.
		if split < model.InputTokens {
			largest = "input"
		}
		costs[largest] += left
	}

	sources := make([]CostSource, 0, len(costs))
	for source, cost := range costs {
		if cost > 0 {
			sources = append(sources, CostSource{Source: source, CostMicros: cost})
		}
	}
	slices.SortFunc(sources, func(a, b CostSource) int {
		return cmp.Or(cmp.Compare(b.CostMicros, a.CostMicros), cmp.Compare(a.Source, b.Source))
	})
	return sources
}

func inputPartsOf(stored store.InputParts) InputParts {
	return InputParts{
		Instructions: stored.InstructionTokens, Messages: stored.MessageTokens,
		ToolDefinitions: stored.ToolDefinitionTokens, ToolUse: stored.ToolUseTokens,
		Images: stored.ImageTokens, Video: stored.VideoTokens,
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
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
	}
	if s.stream == nil {
		return nil, errNoStreamKeys
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownCall
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
	bound, err := s.streamForApp(ctx, customerID, pin, false)
	if failure, refused := refusal(err, callElsewhere, callReadOnly); refused {
		return nil, failure
	}
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if bound, err = s.minting(ctx, bound); err != nil {
		return nil, err
	}

	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := bound.Client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, stack.Wrap(err)
	}

	return &createCallTokenResponse{Body: CallToken{ApiKey: bound.Identity.APIKey,
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
		return nil, errMissingCustomer
	}
	if s.stream == nil {
		return nil, errNoStreamKeys
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

	bound, err := s.agentStream(ctx, customerID, agentID)
	if failure, refused := refusal(err,
		"that agent's conversation is kept in a Stream app this customer no longer acts in",
		"that agent's conversation is kept in the router's shared Stream app, where this app no longer mints tokens"); refused {
		return nil, failure
	}
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if bound, err = s.minting(ctx, bound); err != nil {
		return nil, err
	}
	client := bound.Client
	if err := conversation.CreateMissingUsers(ctx, client, map[string]getstream.UserRequest{
		agentID: {ID: agentID},
		userID:  {ID: userID, Name: &userName},
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

	return &createChatTokenResponse{Body: ChatToken{ApiKey: bound.Identity.APIKey,
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
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
	}
	if s.stream == nil {
		return nil, errNoTranscripts
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownCall
	}

	said, err := s.transcriptOf(ctx, customerID, call)
	if errors.Is(err, errNoStream) {
		return nil, errNoTranscripts
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
	return &getCallTranscriptResponse{Body: messages}, nil
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
func (s *Server) streamForApp(ctx context.Context, customerID string, app int64, reading bool) (streamapp.Bound, error) {
	if s.stream == nil {
		return streamapp.Bound{}, errNoStream
	}
	resolve := s.stream.ForApp
	if reading {
		// Read back, work kept in the router's shared app is reached even once the customer
		// may no longer write there.
		resolve = s.stream.ForAppReading
	}
	bound, err := resolve(ctx, customerID, app)
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

// refusal is what work pinned to an app answers when nothing is to be minted for it: no
// Stream app at all, an app the customer left, or one it may only read there.
func refusal(err error, left, readOnly string) (APIError, bool) {
	switch {
	case errors.Is(err, errNoStream):
		return errNoStreamKeys, true
	case elsewhere(err):
		return invalidRequest(left), true
	case errors.Is(err, streamapp.ErrReadOnly):
		return invalidRequest(readOnly), true
	}
	return APIError{}, false
}

// callElsewhere is what a call made in an app the customer no longer acts in answers.
const callElsewhere = "that call was made in a Stream app this customer no longer acts in"

// callReadOnly is what a call made in the router's shared app answers once this customer
// may no longer act there: what was said can be read, and nothing more is minted.
const callReadOnly = "that call was made in the router's shared Stream app, where this app no longer mints tokens"

// transcriptOf is what was said on a call, read in the app the call was made in. A call
// made in an app the customer no longer acts in has nothing readable from here.
func (s *Server) transcriptOf(ctx context.Context, customerID string, call store.Call) ([]chatlog.Spoken, error) {
	bound, err := s.streamForApp(ctx, customerID, call.StreamAppPK, true)
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
			return s.streamForApp(ctx, customerID, latest[0].StreamAppPK, false)
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

// getCallEvents returns what the conversation decided on one call, oldest first.
func (s *Server) getCallEvents(ctx context.Context, request *getCallEventsRequest) (*getCallEventsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownCall
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
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
	}

	call, err := s.store.Call(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownCall
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
			TurnId:               turn.TurnID,
			StartedAt:            turn.StartedAt,
			CadenceMs:            turn.CadenceMs,
			DecisionMs:           turn.DecisionMs,
			ModelToFirstTextMs:   turn.ModelToFirstTextMs,
			TextToTtsMs:          turn.TextToTTSMs,
			TtsToAudioMs:         turn.TTSToAudioMs,
			ReplyHoldMs:          turn.ReplyHoldMs,
			RoundtripMs:          turn.RoundtripMs,
			SttLatencyMs:         turn.STTLatencyMs,
			LlmTtftMs:            turn.LLMTTFTMs,
			TtsTtfbMs:            turn.TTSTTFBMs,
			SpeechEndToAudioMs:   turn.SpeechEndToAudioMs,
			FirstFrameQueuedMs:   turn.FirstFrameQueuedMs,
			FirstAudibleFrameMs:  turn.FirstAudibleFrameMs,
			SpeechEndToAudibleMs: turn.SpeechEndToAudibleMs,
			AudioOutMs:           turn.AudioOutMs,
			Interrupted:          &turn.Interrupted,
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
		OperationID: "getCallTokens",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}/tokens",
		Summary:     "What a call's models read and wrote, and what their prompts were made of",
		Description: "Read while the call is going as well as after it, unlike the usage on the call, " +
			"which is counted once it is over. A prompt's parts are estimated: no provider says how " +
			"much of a prompt was instructions, tools or images, so the router estimates it from each " +
			"request and scales it to what the provider counted. The parts sum to the input tokens.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The call's tokens"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCallTokens)
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

type getCallTokensRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getCallTokensResponse struct {
	Body CallTokens
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

// CallTokens What a call's models read and wrote, summed over every request, with what their prompts were made of.
type CallTokens struct {
	CachedInputTokens int64         `json:"cached_input_tokens" doc:"The part of the prompts a provider served from its own cache."`
	CostMicros        int64         `json:"cost_micros" doc:"Millionths of a dollar, priced from the providers' configured rates."`
	CostSources       []CostSource  `json:"cost_sources" nullable:"false" doc:"Where the cost came from, the costliest first. Sources that cost nothing are left out."`
	InputParts        InputParts    `json:"input_parts"`
	InputTokens       int64         `json:"input_tokens" doc:"Every prompt the models read, the cached part included."`
	Models            []ModelTokens `json:"models" nullable:"false" doc:"Each model the call used, the busiest first. One billed by audio or characters reads zero tokens."`
	OutputTokens      int64         `json:"output_tokens" doc:"Everything the models generated, reasoning included."`
	Requests          int64         `json:"requests" doc:"How many calls to those models it took."`
}

func (*CallTokens) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What a call's models read and wrote, summed over every request, with what their prompts were made of."
	return schema
}

// InputParts What prompts were made of, in tokens. Estimated from each request and scaled to what the provider counted, so the parts sum to the input tokens and only the split between them is a guess. Requests recorded before the split was kept read zero throughout.
type InputParts struct {
	Images          int64 `json:"images" doc:"Pictures, attached or returned by a tool."`
	Instructions    int64 `json:"instructions" doc:"The system prompt: the agent's instructions, skills and plugin guidance."`
	Messages        int64 `json:"messages" doc:"The conversation's words, from either side."`
	ToolDefinitions int64 `json:"tool_definitions" doc:"The tools the model was offered: their names, descriptions and schemas."`
	ToolUse         int64 `json:"tool_use" doc:"The tools the model called, and what they returned."`
	Video           int64 `json:"video" doc:"Frames of a video, from the call's camera or an attached clip."`
}

func (*InputParts) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What prompts were made of, in tokens. Estimated from each request and scaled to what the provider counted, so the parts sum to the input tokens and only the split between them is a guess. Requests recorded before the split was kept read zero throughout."
	return schema
}

// CostSource One place a call's cost came from.
type CostSource struct {
	Source     string `json:"source" doc:"For a model billed by tokens, the part of its prompt (instructions, messages, tool_definitions, tool_use, images or video), output for what it wrote, or input for prompt tokens recorded without a breakdown. For any other model, its modality: stt, tts, search and the rest."`
	CostMicros int64  `json:"cost_micros" doc:"Millionths of a dollar. A part of the prompt is given its share of what the prompt cost, by tokens."`
}

func (*CostSource) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One place a call's cost came from."
	return schema
}

// ModelTokens What one model read and wrote over a call.
type ModelTokens struct {
	CachedInputTokens int64      `json:"cached_input_tokens"`
	CostMicros        int64      `json:"cost_micros"`
	InputParts        InputParts `json:"input_parts"`
	InputTokens       int64      `json:"input_tokens"`
	Modality          string     `json:"modality" doc:"What the model does: llm for a language model, sts for speech-to-speech, and so on."`
	Model             string     `json:"model"`
	OutputTokens      int64      `json:"output_tokens"`
	Provider          string     `json:"provider"`
	Requests          int64      `json:"requests"`
}

func (*ModelTokens) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What one model read and wrote over a call."
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
	AudioOutMs           *float64           `json:"audio_out_ms,omitempty" doc:"How much the agent spoke."`
	CadenceMs            *float64           `json:"cadence_ms,omitempty" doc:"Last transcript revision to a stable turn ready for the flow controller."`
	DecisionMs           *float64           `json:"decision_ms,omitempty" doc:"Stable turn to the main model request, including flow and queueing."`
	FirstAudibleFrameMs  *float64           `json:"first_audible_frame_ms,omitempty" doc:"Last transcript revision to the outgoing track taking the first frame of the reply that was not silence, which is when it could first be heard. Unlike roundtrip_ms it does not include the wait for a long first chunk to be queued. Absent where the edge does not report it." nullable:"true"`
	FirstFrameQueuedMs   *float64           `json:"first_frame_queued_ms,omitempty" doc:"Last transcript revision to the first frame of the reply being queued for the outgoing track. Absent where the edge does not report it." nullable:"true"`
	Heard                *string            `json:"heard,omitempty" doc:"What the caller said, when it can be matched to this exchange."`
	Interrupted          *bool              `json:"interrupted,omitempty" doc:"Whether the caller talked over the answer."`
	LlmTtftMs            *float64           `json:"llm_ttft_ms,omitempty" doc:"The wait between asking the model and its first token." nullable:"true"`
	ModelCalls           *[]ModelCallTiming `json:"model_calls,omitempty" doc:"Individual model requests for this turn, including flow and delegated work."`
	ModelToFirstTextMs   *float64           `json:"model_to_first_text_ms,omitempty" doc:"Main model request to the first text delta admitted to the voice pipeline."`
	ReplyHoldMs          *float64           `json:"reply_hold_ms,omitempty" doc:"How long the reply's audio was held for the caller to have been quiet, before its first sound and before each sentence that followed a pause in it, added together. A hold before the first sound is inside tts_to_audio_ms, roundtrip_ms and the fields that run to the first frame; one before a later sentence comes after them and is inside none. Absent where nothing was held." nullable:"true"`
	RoundtripMs          *float64           `json:"roundtrip_ms,omitempty" doc:"Last transcript revision to first audio published; includes cadence settling and any hold of its first audio for the caller to have been quiet."`
	Said                 *string            `json:"said,omitempty" doc:"What the agent answered."`
	SpeechEndToAudioMs   *float64           `json:"speech_end_to_audio_ms,omitempty" doc:"Last input audio to first output audio, estimated using provider STT processing time plus roundtrip. It excludes network transport and playback." nullable:"true"`
	SpeechEndToAudibleMs *float64           `json:"speech_end_to_audible_ms,omitempty" doc:"Last input audio to the outgoing track taking the first frame of the reply that was not silence, estimated like speech_end_to_audio_ms. It excludes network transport and playback." nullable:"true"`
	StartedAt            time.Time          `json:"started_at"`
	SttLatencyMs         *float64           `json:"stt_latency_ms,omitempty" doc:"The provider's decode time for the transcript that settled the turn." nullable:"true"`
	TextToTtsMs          *float64           `json:"text_to_tts_ms,omitempty" doc:"First text delta to the first TTS request."`
	TtsToAudioMs         *float64           `json:"tts_to_audio_ms,omitempty" doc:"First TTS request to the first audio chunk published to the edge; includes any hold of its first audio for the caller to have been quiet."`
	TtsTtfbMs            *float64           `json:"tts_ttfb_ms,omitempty" doc:"The wait between sending the first sentence and the first audio." nullable:"true"`
	TurnId               string             `json:"turn_id"`
}

// TranscriptMessage is the TranscriptMessage schema.
type TranscriptMessage struct {
	Agent     *bool     `json:"agent,omitempty" doc:"Whether the agent said it rather than somebody it was talking to. It is what the line was stored as, so it holds however the agent was named."`
	CreatedAt time.Time `json:"created_at"`
	Name      *string   `json:"name,omitempty" doc:"That speaker's display name, when they have one."`
	Speaker   string    `json:"speaker" doc:"Who said it, the agent under its own user id."`
	Text      string    `json:"text"`
}
