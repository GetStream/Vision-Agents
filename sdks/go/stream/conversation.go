package stream

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
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
		return ConversationPage{}, fmt.Errorf("conversation history: %s", r.Status())
	}
	raw, _ := json.Marshal(r.JSON200)
	var page ConversationPage
	err = json.Unmarshal(raw, &page)
	return page, err
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
