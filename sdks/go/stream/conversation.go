package stream

import (
	"context"
	"encoding/json"
	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"net/http"
	"time"
)

type ToolActivity struct {
	Type               string     `json:"type"`
	Product            string     `json:"product,omitempty"`
	SDK                string     `json:"sdk,omitempty"`
	ID                 string     `json:"tool_call_id"`
	Name               string     `json:"name"`
	Title              string     `json:"title"`
	Status             string     `json:"status"`
	Phase              string     `json:"phase"`
	Summary            string     `json:"summary"`
	StartedAt          time.Time  `json:"started_at"`
	ExecutionStartedAt *time.Time `json:"execution_started_at,omitempty"`
	FinishedAt         *time.Time `json:"finished_at,omitempty"`
	DurationMS         int64      `json:"duration_ms"`
}
type ConversationMessage struct {
	ID             string         `json:"id"`
	QuestionID     string         `json:"question_id,omitempty"`
	Role           string         `json:"role"`
	Text           string         `json:"text"`
	State          string         `json:"state"`
	StartedAt      time.Time      `json:"response_started_at"`
	StateStartedAt time.Time      `json:"state_started_at"`
	FinishedAt     *time.Time     `json:"finished_at,omitempty"`
	DurationMS     int64          `json:"duration_ms"`
	Tools          []ToolActivity `json:"attachments"`
	Saved          bool           `json:"saved"`
	Error          string         `json:"persistence_error"`
}
type ConversationPage struct {
	Messages  []ConversationMessage `json:"messages"`
	Before    string                `json:"before"`
	Truncated bool                  `json:"context_truncated"`
}

func (b Backend) ConversationHistory(ctx context.Context, cid, agentID, before string) (ConversationPage, error) {
	client, err := b.Client()
	if err != nil {
		return ConversationPage{}, err
	}
	p := acceleration.GetConversationMessagesParams{AgentId: agentID}
	if before != "" {
		p.Before = &before
	}
	r, err := client.GetConversationMessagesWithResponse(ctx, cid, &p)
	if err != nil {
		return ConversationPage{}, err
	}
	if r.JSON200 == nil {
		return ConversationPage{}, NewRouterError(r.HTTPResponse, r.Body, "", "conversation history", r.Status())
	}
	raw, _ := json.Marshal(r.JSON200)
	var page ConversationPage
	err = json.Unmarshal(raw, &page)
	return page, err
}

// AnswerApproval answers, as the backend, a connector call held for a person to allow it in the
// conversation cid (an ai_tool_call step awaiting_approval): the step's id, the request_id of the
// command its reply answers (support_runtime.command_id), whether it is allowed and, declined,
// what the step says. It is how a backend answers for the writer of a message in a channel of a
// type an agent config answers in, whose session it holds no socket of.
func (b Backend) AnswerApproval(ctx context.Context, cid, toolCallID, requestID string, allowed bool, summary string) error {
	client, err := b.Client()
	if err != nil {
		return err
	}
	body := acceleration.AnswerConversationToolApprovalJSONRequestBody{ToolCallId: toolCallID, RequestId: requestID, Allowed: allowed}
	if summary != "" {
		body.Summary = &summary
	}
	r, err := client.AnswerConversationToolApprovalWithResponse(ctx, cid, body)
	if err != nil {
		return err
	}
	if r.StatusCode() != http.StatusNoContent {
		return NewRouterError(r.HTTPResponse, r.Body, "", "tool approval", r.Status())
	}
	return nil
}

func (e Event) ConversationMessage() (ConversationMessage, bool) {
	var m ConversationMessage
	if e.Kind != "conversation_updated" {
		return m, false
	}
	b, err := json.Marshal(e.Frame["message"])
	if err != nil {
		return m, false
	}
	err = json.Unmarshal(b, &m)
	return m, err == nil
}
