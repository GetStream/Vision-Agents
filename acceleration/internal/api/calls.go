package api

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noCalls and noTranscripts are what the call paths say on a deployment that cannot
// answer them: a call is only remembered if there is somewhere to remember it, and what
// was said lives in Stream Chat rather than here.
const (
	noCalls       = "calls are not available: no database configured"
	noTranscripts = "transcripts are not available: no chat credentials configured"
	unknownCall   = "no such call"
)

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
	return GetCall200JSONResponse(callOf(call)), nil
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
	rendered.Summary = optional(call.Summary)
	rendered.ReviewNotes = optional(call.ReviewNotes)
	rendered.ReviewScore = call.ReviewScore
	if len(call.Tags) > 0 {
		tags := call.Tags
		rendered.Tags = &tags
	}
	return rendered
}
