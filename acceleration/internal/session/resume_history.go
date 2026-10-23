package session

import (
	"context"
	"encoding/json"
	"log/slog"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ResumeHistories reads the resume_history a customer's app and organization policies set.
type ResumeHistories interface {
	ResumeHistory(ctx context.Context, customerID string) store.ResumeHistory
}

// restoring is ctx with the session's resume_history applied to every conversation it
// reads back. A session asking for nothing gets ctx as it was, and the default with it.
func (m *Manager) restoring(ctx context.Context, spec Spec) context.Context {
	asked := spec.ResumeHistory
	if asked == (store.ResumeHistory{}) {
		return ctx
	}
	restore := persistent.Restore{MaxMessages: asked.MaxMessages, MaxRunes: asked.MaxCharacters}
	if asked.Tools() && m.options.Store != nil {
		restore.Recorded = recordedConversation{store: m.options.Store, customerID: spec.CustomerID, logger: m.logger}
	}
	return persistent.WithRestore(ctx, restore)
}

// recordedConversation is what a customer's sessions recorded about a conversation.
//
// A read that fails is logged and taken as nothing recorded: the transcript in Chat is
// still whole, so Postgres being unreachable costs the tool calls, not the conversation.
type recordedConversation struct {
	store      *store.Store
	customerID string
	logger     *slog.Logger
}

func (r recordedConversation) Rounds(ctx context.Context, cid string, turns []string) map[string][]llm.Message {
	replies, err := r.store.ReplyItems(ctx, r.customerID, cid, turns)
	if err != nil {
		r.logger.Warn("resuming a conversation without its tool calls", "conversation", cid, "error", err)
		return nil
	}
	rounds := make(map[string][]llm.Message, len(replies))
	for turn, items := range replies {
		if replayed := roundsOf(items); len(replayed) > 0 {
			rounds[turn] = replayed
		}
	}
	return rounds
}

func (r recordedConversation) Opening(ctx context.Context, cid string) string {
	said, err := r.store.Opening(ctx, r.customerID, cid)
	if err != nil {
		r.logger.Warn("resuming a conversation without its first message", "conversation", cid, "error", err)
		return ""
	}
	return said
}

// roundsOf are one response's tool calls as the model was given them: for each step it
// took, an assistant message asking for its calls, then each call's result. A step ends
// where the model spoke again, at an answer item.
//
// Only results are read, since a result names its call, tool and arguments: a call that
// never answered has nothing to pair with and is left out. A response with any result
// redacted under a retention setting is handed back as text alone, as it would be with its
// items gone, rather than as calls whose arguments and output were taken out.
func roundsOf(items []store.AgentResponseItem) []llm.Message {
	var out []llm.Message
	step := llm.Message{Role: llm.Assistant}
	var results []llm.Message
	finish := func() {
		if len(step.ToolCalls) > 0 {
			out = append(out, step)
			out = append(out, results...)
		}
		step, results = llm.Message{Role: llm.Assistant}, nil
	}
	for _, item := range items {
		switch item.Kind {
		case store.ItemAnswer:
			finish()
		case store.ItemToolResult:
			if redacted, _ := item.Payload["redacted"].(bool); redacted {
				return nil
			}
			id, _ := item.Payload["call_id"].(string)
			if id == "" || item.ToolName == "" {
				continue
			}
			step.ToolCalls = append(step.ToolCalls, llm.ToolCall{ID: id, Name: item.ToolName, Arguments: argumentsOf(item.Payload["arguments"])})
			results = append(results, llm.Message{Role: llm.ToolResult, ToolCallID: id, Content: item.Text})
		}
	}
	finish()
	return out
}

// argumentsOf is a call's arguments as the model wrote them. They are recorded as the JSON
// text the model sent, which reads back as a string.
func argumentsOf(recorded any) string {
	switch arguments := recorded.(type) {
	case string:
		if arguments != "" {
			return arguments
		}
	case nil:
	default:
		if b, err := json.Marshal(arguments); err == nil {
			return string(b)
		}
	}
	return "{}"
}
