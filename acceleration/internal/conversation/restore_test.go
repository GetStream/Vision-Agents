package conversation

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// recorded is a conversation's records as a test sets them: rounds by turn id and the
// first thing said.
type recorded struct {
	rounds  map[string][]llm.Message
	opening string
	asked   []string
}

func (r *recorded) Rounds(_ context.Context, _ string, turns []string) map[string][]llm.Message {
	r.asked = append(r.asked, turns...)
	return r.rounds
}

func (r *recorded) Opening(context.Context, string) string { return r.opening }

func round(callID, tool, arguments, result string) []llm.Message {
	return []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: callID, Name: tool, Arguments: arguments}}},
		{Role: llm.ToolResult, ToolCallID: callID, Content: result},
	}
}

func turn(role, text, turnID string) Message {
	return Message{Role: role, Text: text, State: "completed", TurnID: turnID}
}

func TestARestoredReplyIsHandedBackAfterTheToolCallsItMade(t *testing.T) {
	p := Page{Messages: []Message{
		turn("user", "Find the open invoices", ""),
		turn("assistant", "There are two.", "turn-1"),
		turn("user", "Thanks", ""),
	}}
	r := Restore{Recorded: &recorded{}}

	h, tr := r.history(p, map[string][]llm.Message{"turn-1": round("call-1", "invoices", `{"status":"open"}`, `[1,2]`)}, "")

	require.False(t, tr)
	require.Equal(t, []llm.Message{
		{Role: llm.User, Content: "Find the open invoices"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "invoices", Arguments: `{"status":"open"}`}}},
		{Role: llm.ToolResult, ToolCallID: "call-1", Content: `[1,2]`},
		{Role: llm.Assistant, Content: "There are two."},
		{Role: llm.User, Content: "Thanks"},
	}, h)
}

func TestAReplyWithNothingRecordedIsHandedBackAsText(t *testing.T) {
	p := Page{Messages: []Message{
		turn("user", "Hello", ""),
		turn("assistant", "Hi", "pruned-turn"),
	}}

	h, _ := Restore{Recorded: &recorded{}}.history(p, nil, "")

	require.Equal(t, []llm.Message{{Role: llm.User, Content: "Hello"}, {Role: llm.Assistant, Content: "Hi"}}, h)
}

func TestAReplyThatOnlyCalledToolsKeepsItsCalls(t *testing.T) {
	p := Page{Messages: []Message{turn("user", "Run it", ""), turn("assistant", "", "turn-1")}}

	h, _ := Restore{Recorded: &recorded{}}.history(p, map[string][]llm.Message{"turn-1": round("call-1", "run", "{}", "done")}, "")

	require.Equal(t, append([]llm.Message{{Role: llm.User, Content: "Run it"}}, round("call-1", "run", "{}", "done")...), h)
}

func TestOverTheSizeTheOldestToolResultsAreTrimmedBeforeAnyTurn(t *testing.T) {
	p := Page{Messages: []Message{
		turn("user", "task", ""),
		turn("assistant", "first", "turn-1"),
		turn("user", "more", ""),
		turn("assistant", "second", "turn-2"),
	}}
	rounds := map[string][]llm.Message{
		"turn-1": round("call-1", "read", "{}", strings.Repeat("a", 400)),
		"turn-2": round("call-2", "read", "{}", strings.Repeat("b", 400)),
	}

	h, tr := Restore{MaxRunes: 600, Recorded: &recorded{}}.history(p, rounds, "")

	require.True(t, tr)
	require.Len(t, h, 8)
	require.Equal(t, ResultTrimmed, h[2].Content)
	require.Equal(t, strings.Repeat("b", 400), h[6].Content)
	require.Equal(t, "second", h[7].Content)
}

func TestTheFirstMessageIsKeptWhenOlderTurnsAreLeftOut(t *testing.T) {
	p := Page{Messages: []Message{
		turn("assistant", "Welcome", ""),
		turn("user", "the task", ""),
		turn("assistant", "one", ""),
		turn("user", "two", ""),
		turn("assistant", "three", ""),
	}}

	h, tr := Restore{MaxMessages: 3, Recorded: &recorded{}}.history(p, nil, "")

	require.True(t, tr)
	require.Equal(t, []llm.Message{
		{Role: llm.User, Content: "the task"},
		{Role: llm.User, Content: "two"},
		{Role: llm.Assistant, Content: "three"},
	}, h)
}

func TestTheFirstMessageOutlastsTheTurnsAfterItWhenOverTheSize(t *testing.T) {
	p := Page{Messages: []Message{
		turn("user", "the task", ""),
		turn("assistant", strings.Repeat("x", 500), ""),
		turn("user", "latest", ""),
	}}

	h, tr := Restore{MaxRunes: 100, Recorded: &recorded{}}.history(p, nil, "")

	require.True(t, tr)
	require.Equal(t, []llm.Message{{Role: llm.User, Content: "the task"}, {Role: llm.User, Content: "latest"}}, h)
}

func TestAConversationLongerThanWhatWasReadKeepsItsRecordedOpening(t *testing.T) {
	p := Page{Truncated: true, Messages: []Message{turn("user", "later", ""), turn("assistant", "answer", "")}}

	h, tr := Restore{Recorded: &recorded{}}.history(p, nil, "the task")

	require.True(t, tr)
	require.Equal(t, []llm.Message{
		{Role: llm.User, Content: "the task"},
		{Role: llm.User, Content: "later"},
		{Role: llm.Assistant, Content: "answer"},
	}, h)
}

func TestCapsAloneChangeOnlyHowMuchTextIsHandedBack(t *testing.T) {
	p := Page{Messages: []Message{turn("user", "the task", ""), turn("assistant", "one", "turn-1"), turn("user", "two", "")}}

	h, tr := Restore{MaxMessages: 2}.history(p, map[string][]llm.Message{"turn-1": round("call-1", "x", "{}", "y")}, "")

	require.True(t, tr)
	require.Equal(t, []llm.Message{{Role: llm.Assistant, Content: "one"}, {Role: llm.User, Content: "two"}}, h)
}

// TestAResumedConversationReadsTheToolsOfTheRepliesItHolds goes through Chat: the turn id
// a reply was written under is what its recorded calls are found by.
func TestAResumedConversationReadsTheToolsOfTheRepliesItHolds(t *testing.T) {
	_, client := newChat(t)
	service := NewForChat(client)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	require.NoError(t, c.Begin("Look it up"))
	c.Observe(agent.Responding{TurnID: "turn-1"})
	c.Observe(agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "lookup", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "call-1", TurnID: "turn-1", Tool: "lookup", Result: "42"})
	c.Observe(agent.ResponseDelta{TurnID: "turn-1", Text: "It is 42."})
	c.Observe(agent.Responded{TurnID: "turn-1"})
	saved(t, c)
	cid := c.CID()
	c.Release()

	records := &recorded{rounds: map[string][]llm.Message{"turn-1": round("call-1", "lookup", "{}", "42")}}
	ctx := WithRestore(t.Context(), Restore{Recorded: records})
	messages, _, err := service.ContextForCaller(ctx, "customer", "agent", cid, "employee")

	require.NoError(t, err)
	require.Equal(t, []string{"turn-1"}, records.asked)
	require.Equal(t, append(append([]llm.Message{{Role: llm.User, Content: "Look it up"}},
		round("call-1", "lookup", "{}", "42")...), llm.Message{Role: llm.Assistant, Content: "It is 42."}), messages)
}

func TestAHistoryNeverOpensOnAToolRound(t *testing.T) {
	p := Page{Truncated: true, Messages: []Message{turn("assistant", "done", "turn-1"), turn("user", "next", "")}}

	h, _ := Restore{Recorded: &recorded{}}.history(p, map[string][]llm.Message{"turn-1": round("call-1", "x", "{}", "y")}, "")

	require.Equal(t, []llm.Message{{Role: llm.Assistant, Content: "done"}, {Role: llm.User, Content: "next"}}, h)
}

func TestARecordedOpeningAlreadyReadIsNotHandedBackTwice(t *testing.T) {
	p := Page{Truncated: true, Messages: []Message{turn("user", "the task", ""), turn("assistant", "answer", "")}}

	h, _ := Restore{Recorded: &recorded{}}.history(p, nil, "the task")

	require.Equal(t, []llm.Message{{Role: llm.User, Content: "the task"}, {Role: llm.Assistant, Content: "answer"}}, h)
}
