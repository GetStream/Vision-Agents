package session

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
)

// ToolCallStatus is where one tool call stands.
type ToolCallStatus string

const (
	ToolCallInProgress ToolCallStatus = "in_progress"
	// ToolCallWaiting is a call held for a person's approval.
	ToolCallWaiting   ToolCallStatus = "waiting"
	ToolCallSuccess   ToolCallStatus = "success"
	ToolCallFailed    ToolCallStatus = "failed"
	ToolCallCancelled ToolCallStatus = "cancelled"
)

// ToolCallPhase says which of a call's changes a ToolCallChanged is: its first, one while it
// runs, or its last.
type ToolCallPhase string

const (
	ToolCallStarted  ToolCallPhase = "started"
	ToolCallProgress ToolCallPhase = "progress"
	ToolCallFinished ToolCallPhase = "finished"
)

// ToolCallState is everything a client shows of one tool call, as it is now.
type ToolCallState struct {
	ID     string
	Tool   string
	TurnID string
	Status ToolCallStatus
	// DisplayTitle is what the call is doing, in words for the people in the conversation,
	// as its tool declared it. Empty when the tool declared none.
	DisplayTitle string
	// Progress is the last thing said about the call while it runs, such as "Searching for
	// flights", or the question it waits on.
	Progress  string
	StartedAt time.Time
	// Duration is from StartedAt to this change, or to the end for a finished call.
	Duration   time.Duration
	ApprovedAt time.Time
	RejectedAt time.Time
	// Error is why a call failed, or what a declined one shows, such as "Not allowed in time".
	Error string
}

// ToolCallChanged is one change to a tool call's state, sent to every watcher.
type ToolCallChanged struct {
	Phase ToolCallPhase
	State ToolCallState
}

// toolCalls follows the session's tool calls from their agent's events to the state a client
// shows. A call is forgotten once it finishes, so a late event about it changes nothing.
type toolCalls struct {
	// approves reports whether a call to a tool waits for a person before it runs.
	approves func(tool string) bool
	// titles is what a call to a tool is shown doing, or empty for nothing.
	titles func(tool string) string

	mu    sync.Mutex
	calls map[string]*ToolCallState
	// early are the calls a connector began holding for approval before their agent's
	// ToolStarted came through, which the two report on separate goroutines.
	early map[string]agent.ToolAwaitingApproval
}

// observe applies event and returns the change it made, reporting false for none.
func (t *toolCalls) observe(event agent.Event) (ToolCallChanged, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now().UTC()
	switch typed := event.(type) {
	case agent.ToolStarted:
		if typed.Tool == agent.ConfirmTool || t.calls[typed.ID] != nil {
			return ToolCallChanged{}, false
		}
		state := &ToolCallState{ID: typed.ID, Tool: typed.Tool, TurnID: typed.TurnID, Status: ToolCallInProgress,
			Progress: typed.PreSpeech, StartedAt: typed.StartedAt}
		if state.StartedAt.IsZero() {
			state.StartedAt = now
		}
		if t.approves != nil && t.approves(typed.Tool) {
			state.Status = ToolCallWaiting
		}
		if t.titles != nil {
			state.DisplayTitle = t.titles(typed.Tool)
		}
		if waiting, ok := t.early[typed.ID]; ok {
			delete(t.early, typed.ID)
			state.Status, state.Progress = ToolCallWaiting, waiting.Question
		}
		if t.calls == nil {
			t.calls = map[string]*ToolCallState{}
		}
		t.calls[typed.ID] = state
		return ToolCallChanged{Phase: ToolCallStarted, State: *state}, true
	case agent.ToolAwaitingApproval:
		if t.calls[typed.ID] == nil {
			if t.early == nil {
				t.early = map[string]agent.ToolAwaitingApproval{}
			}
			t.early[typed.ID] = typed
			return ToolCallChanged{}, false
		}
		return t.changeLocked(typed.ID, now, func(state *ToolCallState) {
			state.Status = ToolCallWaiting
			if typed.Question != "" {
				state.Progress = typed.Question
			}
		})
	case agent.ToolUpdated:
		return t.changeLocked(typed.ID, now, func(state *ToolCallState) { state.Progress = typed.Text })
	case agent.ToolApprovalDecided:
		if !typed.Allowed {
			return t.finishLocked(typed.ID, now, ToolCallCancelled, typed.Summary)
		}
		return t.changeLocked(typed.ID, now, func(state *ToolCallState) {
			state.Status, state.ApprovedAt = ToolCallInProgress, now
		})
	case agent.ToolRan:
		if typed.Asked {
			return ToolCallChanged{}, false
		}
		id := typed.ID
		// An approved call runs inside the confirmation that answered its question.
		if typed.Tool == agent.ConfirmTool {
			var answer struct {
				ToolCallID string `json:"tool_call_id"`
			}
			if json.Unmarshal([]byte(typed.Arguments), &answer) != nil {
				return ToolCallChanged{}, false
			}
			id = answer.ToolCallID
		}
		switch {
		case errors.Is(typed.Err, context.Canceled):
			return t.finishLocked(id, now, ToolCallCancelled, "")
		case typed.Err != nil:
			return t.finishLocked(id, now, ToolCallFailed, typed.Err.Error())
		}
		if state := t.calls[id]; state != nil && state.Status == ToolCallWaiting {
			return t.finishLocked(id, now, ToolCallCancelled, "")
		}
		return t.finishLocked(id, now, ToolCallSuccess, "")
	}
	return ToolCallChanged{}, false
}

// changeLocked applies change to a call still running. The caller holds t.mu.
func (t *toolCalls) changeLocked(id string, now time.Time, change func(*ToolCallState)) (ToolCallChanged, bool) {
	state := t.calls[id]
	if state == nil {
		return ToolCallChanged{}, false
	}
	before := *state
	change(state)
	if *state == before {
		return ToolCallChanged{}, false
	}
	state.Duration = now.Sub(state.StartedAt)
	return ToolCallChanged{Phase: ToolCallProgress, State: *state}, true
}

// finishLocked ends a call as status and forgets it. A call declined while it waited is
// rejected now. The caller holds t.mu.
func (t *toolCalls) finishLocked(id string, now time.Time, status ToolCallStatus, reason string) (ToolCallChanged, bool) {
	state := t.calls[id]
	if state == nil {
		return ToolCallChanged{}, false
	}
	delete(t.calls, id)
	if status == ToolCallCancelled && state.Status == ToolCallWaiting {
		state.RejectedAt = now
	}
	state.Status, state.Error, state.Duration = status, reason, now.Sub(state.StartedAt)
	return ToolCallChanged{Phase: ToolCallFinished, State: *state}, true
}

// trackToolCall sends every watcher the change event made to a tool call, if it made one.
func (s *Session) trackToolCall(event agent.Event) {
	if changed, ok := s.toolCalls.observe(event); ok {
		s.broadcast(changed)
	}
}

// approves reports whether a call to the caller's tool waits for a person before it runs.
func (s *Session) approves(tool string) bool {
	for _, declared := range s.declaredTools() {
		if declared.Name == tool {
			return declared.Approval != nil
		}
	}
	return false
}

// toolTitle is what a call to the tool is shown doing, as whoever declared it said: the
// caller, or the worker hosting it.
func (s *Session) toolTitle(tool string) string {
	all := slices.Concat(s.declaredTools(), s.otherTools)
	title := ""
	if at := slices.IndexFunc(all, named(tool)); at >= 0 {
		title = all[at].DisplayTitle
	}
	return persistent.ToolTitle(tool, title)
}
