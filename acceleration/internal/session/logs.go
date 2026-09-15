package session

import (
	"context"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"log/slog"
	"sync"
	"time"
)

type logRecorder struct {
	queue  chan store.AgentLog
	done   chan struct{}
	once   sync.Once
	mu     sync.RWMutex
	closed bool
	cancel context.CancelFunc
	store  *store.Store
	logger *slog.Logger
}

func newLogRecorder(db *store.Store, logger *slog.Logger) *logRecorder {
	writerCtx, cancelWriter := context.WithCancel(context.Background())
	r := &logRecorder{cancel: cancelWriter, queue: make(chan store.AgentLog, 1024), done: make(chan struct{}), store: db, logger: logger}
	go func() {
		defer close(r.done)
		for entry := range r.queue {
			if writerCtx.Err() != nil {
				db.LogDrops.Add(1)
				continue
			}
			ctx, cancel := context.WithTimeout(writerCtx, 3*time.Second)
			err := db.RecordAgentLog(ctx, &entry)
			cancel()
			if err != nil {
				db.LogDrops.Add(1)
				logger.Error("agent log persistence failed")
			}
		}
	}()
	return r
}
func (r *logRecorder) close() {
	r.once.Do(func() {
		r.mu.Lock()
		r.closed = true
		close(r.queue)
		r.mu.Unlock()
		deadline := time.AfterFunc(10*time.Second, r.cancel)
		defer deadline.Stop()
		defer r.cancel()
		<-r.done
	})
}
func (s *Session) recordLog(event Event) {
	if s.logs == nil {
		return
	}
	row := store.AgentLog{CustomerID: s.spec.CustomerID, ConfigID: s.spec.ConfigID, AgentID: s.spec.AgentID, SessionID: s.id, Source: "agent", Severity: "info", Details: map[string]any{}}
	switch e := event.(type) {
	case agent.Joined:
		row.EventType = "joined"
		row.Message = "Agent joined"
		row.OccurredAt = e.At
	case agent.Left:
		row.EventType = "left"
		row.Message = "Agent left"
		row.OccurredAt = e.At
	case agent.ParticipantJoined:
		row.EventType = "participant_joined"
		row.Source = "user"
		row.UserID = e.Participant.UserID
		row.Message = "Participant joined"
	case agent.ParticipantLeft:
		row.EventType = "participant_left"
		row.Source = "user"
		row.UserID = e.Participant.UserID
		row.Message = "Participant left"
	case agent.Heard:
		row.EventType = "user_message"
		row.Source = "user"
		row.UserID = e.Participant.UserID
		row.Message = e.Text
	case agent.Responding:
		if !s.spec.Text {
			return
		}
		row.EventType = "user_message"
		row.Source = "user"
		row.UserID = e.Participant.UserID
		row.Message = e.Prompt
		row.Details["turn_id"] = e.TurnID
	case agent.Responded:
		row.EventType = "agent_message"
		row.Message = e.Text
		row.Details["turn_id"] = e.TurnID
	case agent.Decided:
		row.EventType = "decision"
		row.Message = e.Reason
		row.UserID = e.Participant.UserID
		row.Details["kind"] = e.Kind
		row.Details["turn_id"] = e.TurnID
		row.OccurredAt = e.At
	case agent.ToolStarted:
		row.EventType = "tool_started"
		row.Source = "tool"
		row.Message = e.Tool + " started"
		row.Details["tool_call_id"] = e.ID
		row.Details["tool"] = e.Tool
		row.Details["turn_id"] = e.TurnID
	case agent.ToolRan:
		row.EventType = "tool_finished"
		row.Source = "tool"
		row.Message = e.Tool + " completed"
		if e.Err != nil {
			row.Severity = "error"
			row.Details["error_message"] = e.Err.Error()
			row.Message = e.Tool + " failed"
		}
		row.Details["tool_call_id"] = e.ID
		row.Details["turn_id"] = e.TurnID
		row.Details["tool"] = e.Tool
	case agent.Delegated:
		row.EventType = "delegated"
		row.Message = "Delegated to " + e.Skill
		row.Details["turn_id"] = e.TurnID
		row.Details["tool_call_id"] = e.TaskID
	case agent.TaskSettled:
		row.EventType = "task_settled"
		row.Message = e.Skill + " completed"
		if e.Err != nil {
			row.Severity = "error"
			row.Details["error_message"] = e.Err.Error()
			row.Message = e.Skill + " failed"
		}
		row.Details["duration_ms"] = e.ElapsedMs
		row.Details["tool_call_id"] = e.TaskID
	case agent.TaskCancelled:
		row.EventType = "task_cancelled"
		row.Message = e.Skill + " cancelled"
		row.Details["tool_call_id"] = e.TaskID
	case agent.Error:
		row.EventType = "runtime_error"
		row.Severity = "error"
		if e.Err != nil {
			row.Details["error_message"] = e.Err.Error()
		}
		row.Source = "system"
		row.Message = "Agent runtime failed: " + store.SafeLogText(e.Context)
		row.Details["context"] = e.Context
	default:
		return
	}
	row.Message = store.SafeLogText(row.Message)
	if message, ok := row.Details["error_message"].(string); ok {
		row.Details["error_message"] = store.SafeLogText(message)
	}
	s.logs.mu.RLock()
	defer s.logs.mu.RUnlock()
	if s.logs.closed {
		return
	}
	select {
	case s.logs.queue <- row:
	default:
		s.logs.store.LogDrops.Add(1)
		s.logger.Warn("agent log buffer full")
	}
}
