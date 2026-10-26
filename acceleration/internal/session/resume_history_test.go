package session

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type ResumeHistorySuite struct {
	suite.Suite
	held    *heldRecorder
	session *Session
}

func TestResumeHistorySuite(t *testing.T) {
	suite.Run(t, new(ResumeHistorySuite))
}

func (s *ResumeHistorySuite) SetupTest() {
	s.held = &heldRecorder{}
	s.session = &Session{id: "session-1", spec: Spec{CustomerID: "acme"}, records: s.held, recording: true}
}

func (s *ResumeHistorySuite) took(events ...Event) {
	for _, event := range events {
		s.session.record(event)
	}
}

// A turn that called two tools at once, then one more, then answered: the steps the model
// took come back as it took them, each call before its result.
func (s *ResumeHistorySuite) TestARecordedTurnIsReplayedStepByStep() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "Compare the two plans"},
		agent.Responded{TurnID: "turn-1", Text: "Looking.", PendingWork: true},
		agent.ToolStarted{TurnID: "turn-1", ID: "call-a", Tool: "plan"},
		agent.ToolStarted{TurnID: "turn-1", ID: "call-b", Tool: "plan"},
		agent.ToolRan{TurnID: "turn-1", ID: "call-a", Tool: "plan", Arguments: `{"id":"a"}`, Result: "A costs 10"},
		agent.ToolRan{TurnID: "turn-1", ID: "call-b", Tool: "plan", Arguments: `{"id":"b"}`, Result: "B costs 12"},
		agent.Responding{TurnID: "turn-2", Continues: "turn-1"},
		agent.Responded{TurnID: "turn-2", PendingWork: true},
		agent.ToolStarted{TurnID: "turn-2", ID: "call-c", Tool: "discount"},
		agent.ToolRan{TurnID: "turn-2", ID: "call-c", Tool: "discount", Arguments: `{}`, Result: "none"},
		agent.Responding{TurnID: "turn-3", Continues: "turn-2"},
		agent.Responded{TurnID: "turn-3", Text: "A is cheaper."},
	)

	s.Equal([]llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{
			{ID: "call-a", Name: "plan", Arguments: `{"id":"a"}`},
			{ID: "call-b", Name: "plan", Arguments: `{"id":"b"}`},
		}},
		{Role: llm.ToolResult, ToolCallID: "call-a", Content: "A costs 10"},
		{Role: llm.ToolResult, ToolCallID: "call-b", Content: "B costs 12"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "call-c", Name: "discount", Arguments: `{}`}}},
		{Role: llm.ToolResult, ToolCallID: "call-c", Content: "none"},
	}, roundsOf(s.held.items))
}

// The transcript writes a reply under the turn that finished it, so every answer says which
// turn it was.
func (s *ResumeHistorySuite) TestEveryAnswerNamesItsTurn() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "hi"},
		agent.Responded{TurnID: "turn-1", Text: "hello"},
	)

	answer := s.held.items[len(s.held.items)-1]
	s.Equal(store.ItemAnswer, answer.Kind)
	s.Equal("turn-1", answer.Payload["turn_id"])
}

func (s *ResumeHistorySuite) TestACallThatNeverAnsweredIsLeftOut() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "Do it"},
		agent.Responded{TurnID: "turn-1", PendingWork: true},
		agent.ToolStarted{TurnID: "turn-1", ID: "call-a", Tool: "slow"},
		agent.Interrupted{TurnID: "turn-1"},
	)

	s.Empty(roundsOf(s.held.items))
}

func (s *ResumeHistorySuite) TestAResponseWithARedactedResultIsHandedBackAsText() {
	items := []store.AgentResponseItem{
		{Kind: store.ItemToolResult, ToolName: "plan", Text: "kept", Payload: map[string]any{"call_id": "call-a", "arguments": "{}"}},
		{Kind: store.ItemToolResult, ToolName: "plan", Text: "taken out", Payload: map[string]any{"call_id": "call-b", "arguments": "{}"}},
	}
	items[1].Redact()

	s.Nil(roundsOf(items))
}

func (s *ResumeHistorySuite) TestArgumentsThatWereNeverRecordedAreAnEmptyObject() {
	items := []store.AgentResponseItem{{Kind: store.ItemToolResult, ToolName: "ping", Text: "pong", Payload: map[string]any{"call_id": "call-a"}}}

	s.Equal("{}", roundsOf(items)[0].ToolCalls[0].Arguments)
}

func (s *ResumeHistorySuite) TestASessionAskingForNothingReadsHistoryAsItAlwaysHas() {
	m := &Manager{}
	ctx := s.T().Context()

	s.Equal(ctx, m.restoring(ctx, Spec{}))
}

// The levels a session's resume_history is read from, most specific first, each winning
// over the next field by field: the session, its config, its app, its organization.
func (s *ResumeHistorySuite) TestTheMostSpecificLevelWinsFieldByField() {
	on, off := true, false
	organization := store.ResumeHistory{IncludeTools: &on, MaxMessages: 10, MaxCharacters: 10000}
	app := store.ResumeHistory{MaxMessages: 20}
	config := store.ResumeHistory{IncludeTools: &off}
	session := store.ResumeHistory{MaxCharacters: 30000}

	resolved := session.Over(config).Over(app.Over(organization))

	s.Equal(store.ResumeHistory{IncludeTools: &off, MaxMessages: 20, MaxCharacters: 30000}, resolved)
	s.False(resolved.Tools())
}

func (s *ResumeHistorySuite) TestAnAppAndOrganizationPolicyFillWhatTheSessionLeftOut() {
	on := true
	m := &Manager{options: ManagerOptions{ResumeHistories: fixedResumeHistory{IncludeTools: &on}}}
	spec := Spec{CustomerID: "acme", ResumeHistory: store.ResumeHistory{MaxMessages: 5}}

	spec.ResumeHistory = spec.ResumeHistory.Over(m.options.ResumeHistories.ResumeHistory(s.T().Context(), spec.CustomerID))

	s.True(spec.ResumeHistory.Tools())
	s.Equal(5, spec.ResumeHistory.MaxMessages)
}

type fixedResumeHistory store.ResumeHistory

func (f fixedResumeHistory) ResumeHistory(context.Context, string) store.ResumeHistory {
	return store.ResumeHistory(f)
}
