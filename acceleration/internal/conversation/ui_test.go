package conversation

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

func TestSplitUIKeepsTheWordsAndTakesTheBlock(t *testing.T) {
	text := "Here's a weekend in Ohrid.\n\n```openui-lang\nroot = Card([title])\ntitle = CardHeader(\"Ohrid\")\n```\n\nEnjoy the lake!"

	words, ui := splitUI(text, true)

	require.Equal(t, "Here's a weekend in Ohrid.\n\n\nEnjoy the lake!", words)
	require.Equal(t, map[string]any{
		"v": uiVersion, "format": "openui", "status": "complete",
		"content": "root = Card([title])\ntitle = CardHeader(\"Ohrid\")",
	}, ui)
}

func TestSplitUIFollowsABlockAsItIsWritten(t *testing.T) {
	words, ui := splitUI("Here's a plan.\n```openui-lang\nroot = Card([ti", false)
	require.Equal(t, "Here's a plan.", words)
	require.Equal(t, "streaming", ui["status"])
	require.Equal(t, "root = Card([ti", ui["content"])

	words, ui = splitUI("Here's a plan.\n```open", false)
	require.Equal(t, "Here's a plan.", words, "a line that may open a block is held back")
	require.Nil(t, ui)

	_, ui = splitUI("Here's a plan.\n```openui-lang\nroot = Card([])\n```", false)
	require.Equal(t, "complete", ui["status"], "a closed block is done while the words go on")
}

func TestSplitUIReadsAnyFormatNamedAsUI(t *testing.T) {
	_, ui := splitUI("```ui:a2ui@1.0.0\n[{\"version\":\"v1.0\"}]\n```", true)
	require.Equal(t, "a2ui", ui["format"])
	require.Equal(t, "1.0.0", ui["version"])
	require.Equal(t, `[{"version":"v1.0"}]`, ui["content"])

	_, ui = splitUI("```ui:\nnothing\n```", true)
	require.Nil(t, ui, "a format has a name")
}

func TestSplitUILeavesOtherBlocksAndDropsALaterUI(t *testing.T) {
	text := "Code:\n```swift\nlet a = 1\n```\n```openui-lang\nroot = Card([a])\n```\n```openui-lang\nroot = Card([b])\n```\nDone."

	words, ui := splitUI(text, true)

	require.Equal(t, "Code:\n```swift\nlet a = 1\n```\nDone.", words)
	require.Equal(t, "root = Card([a])", ui["content"])

	words, ui = splitUI("Just words, no UI.", true)
	require.Equal(t, "Just words, no UI.", words)
	require.Nil(t, ui)
}

func TestTheModelReadsBackItsUIAndWhatThePersonDidInIt(t *testing.T) {
	ui := map[string]any{"format": "openui", "content": "root = Card([])"}
	action := map[string]any{"message_id": "m1", "format": "openui", "name": "ask_about", "payload": map[string]any{"title": "Ohrid"}}

	reply := entryOf(Page{}, Message{Role: "assistant", Text: "A weekend in Ohrid.", UI: ui, State: "completed"})
	question := entryOf(Page{}, Message{Role: "user", Text: "Tell me more about Ohrid", UIAction: action, State: "completed"})

	require.Equal(t, "A weekend in Ohrid.\n\n```openui-lang\nroot = Card([])\n```", reply.message.Content)
	require.Equal(t, "Tell me more about Ohrid\n\n(They did this in the openui UI of your earlier reply: ask_about, with {\"title\":\"Ohrid\"})", question.message.Content)
	require.Equal(t, "```ui:a2ui\n[]\n```", uiBlock(map[string]any{"format": "a2ui", "content": "[]"}))
}

func TestOnlyASmallNamedUIActionIsRecorded(t *testing.T) {
	require.True(t, ValidUIAction(nil))
	require.True(t, ValidUIAction(map[string]any{"message_id": "m1", "format": "openui", "name": "continue_conversation"}))
	require.False(t, ValidUIAction(map[string]any{"format": "openui", "name": "continue_conversation"}))
	require.False(t, ValidUIAction(map[string]any{"message_id": "m1", "format": "openui", "name": "x", "payload": string(make([]byte, MaxUIAction))}))
	require.Empty(t, DescribeUIAction(nil, ""))
}

func TestAReplysUIIsStoredBesideItsWordsAndReadBack(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "juno", "")
	require.NoError(t, err)
	action := map[string]any{"message_id": "m0", "format": "openui", "name": "continue_conversation", "payload": map[string]any{"values": map[string]any{"pace": "relaxed"}}}
	receipt, err := c.BeginCommandWithAction("command-1", "Update the plan", "", action)
	require.NoError(t, err)
	c.Observe(agent.ResponseDelta{Text: "A relaxed weekend in Ohrid.\n```openui-lang\nroot = Card([title])\n"})
	c.Observe(agent.ResponseDelta{Text: "title = CardHeader(\"Ohrid\")\n```"})
	c.Observe(agent.Responded{})
	saved(t, c)

	db.mu.Lock()
	reply, question := db.messages[receipt.AssistantMessageID], db.messages[receipt.UserMessageID]
	db.mu.Unlock()
	require.Equal(t, "A relaxed weekend in Ohrid.", reply["text"])
	require.Equal(t, map[string]any{
		"v": float64(uiVersion), "format": "openui", "status": "complete",
		"content": "root = Card([title])\ntitle = CardHeader(\"Ohrid\")",
	}, reply["custom"].(map[string]any)["ai_ui"])
	require.Equal(t, "Update the plan", question["text"])
	require.Equal(t, "continue_conversation", question["custom"].(map[string]any)["ai_ui_action"].(map[string]any)["name"])

	// A session carried on later reads both back, and its model sees what it showed and what
	// the person did.
	cold := NewForChat(client)
	defer cold.Close()
	page, err := cold.History(context.Background(), "customer", "juno", c.CID(), "")
	require.NoError(t, err)
	require.Len(t, page.Messages, 2)
	history, _ := history(page)
	require.Len(t, history, 2)
	require.Contains(t, history[0].Content, "(They did this in the openui UI of your earlier reply: continue_conversation")
	require.Contains(t, history[1].Content, "```openui-lang\nroot = Card([title])")
}

// A program of empty strings doubles as JSON, each quote escaped: 33 KB of it is 55 KB of
// custom data, over Stream's 50 KB.
var escapedProgram = "[" + strings.Repeat(`"",`, 11000) + `""]`

func TestAUIIsSetOnlyWhenItFitsBesideTheRestOfTheCustomData(t *testing.T) {
	words := map[string]any{"text": strings.Repeat("words ", 20000), "ai_ui": map[string]any{"content": "[]"}}
	require.Nil(t, uiUnset(words), "a message's text is not custom data")
	require.Contains(t, words, "ai_ui")

	escaped := map[string]any{"text": "A plan.", "ai_ui": map[string]any{"content": escapedProgram}}
	require.Less(t, len(escapedProgram), maxMessageCustom)
	require.Equal(t, []string{"ai_ui"}, uiUnset(escaped), "an update leaving the UI out unsets the one set before")
	require.NotContains(t, escaped, "ai_ui")

	created := map[string]any{"source": "agent", "ai_ui": map[string]any{"content": escapedProgram}}
	require.True(t, fitUI(created))
	require.NotContains(t, created, "ai_ui")
	require.Nil(t, uiUnset(map[string]any{"text": "No UI."}))
}

func TestAUITooLargeForStreamLeavesTheReplyItsWords(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "juno", "")
	require.NoError(t, err)
	receipt, err := c.BeginCommand("command-1", "Plan a weekend in Ohrid", "")
	require.NoError(t, err)

	c.Observe(agent.ResponseDelta{Text: "A weekend in Ohrid.\n```ui:a2ui\n" + escapedProgram + "\n```"})
	c.Observe(agent.Responded{})
	saved(t, c)

	db.mu.Lock()
	reply := db.messages[receipt.AssistantMessageID]
	db.mu.Unlock()
	require.Equal(t, "A weekend in Ohrid.", reply["text"])
	require.NotContains(t, reply["custom"], "ai_ui", "Stream would refuse it, holding back every write after it")
	_, err = c.BeginCommand("command-2", "Thanks", "")
	require.NoError(t, err, "the conversation goes on")
}

func TestAPersonsMessageKeepsTheUIBlockTheyWrote(t *testing.T) {
	db, client := newChat(t)
	service := NewForChat(client)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	question := "Please fix this UI:\n```ui:a2ui\n[{\"version\":\"v1.0\"}]\n```"
	receipt, err := c.BeginCommand("command-1", question, "")
	require.NoError(t, err)
	c.Observe(agent.ResponseDelta{Text: "It needs a createSurface."})
	c.Observe(agent.Responded{})
	saved(t, c)

	db.mu.Lock()
	stored := db.messages[receipt.UserMessageID]
	db.mu.Unlock()
	require.Equal(t, question, stored["text"])
	require.NotContains(t, stored["custom"], "ai_ui", "only the agent writes UI")
	c.Release()
	service.Close()

	// Reopened, the model reads what the person sent, and the same question retried is the
	// same command.
	recovered := NewForChat(client)
	t.Cleanup(recovered.Close)
	page, err := recovered.HistoryForCaller(t.Context(), "customer", "agent", c.CID(), "", "employee")
	require.NoError(t, err)
	history, _ := history(page)
	require.Equal(t, question, history[0].Content)
	c, _, _, err = recovered.OpenForCaller(t.Context(), "customer", "agent", c.CID(), "employee")
	require.NoError(t, err)
	replay, err := c.BeginCommand("command-1", question, "")
	require.NoError(t, err)
	require.True(t, replay.Duplicate)
}

func TestARetryCarryingAnotherActionIsAnotherCommand(t *testing.T) {
	_, client := newChat(t)
	service := NewForChat(client)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	active := map[string]any{"message_id": "m0", "format": "a2ui", "name": "update_plan", "payload": map[string]any{"pace": "active"}}
	relaxed := map[string]any{"message_id": "m0", "format": "a2ui", "name": "update_plan", "payload": map[string]any{"pace": "relaxed"}}
	_, err = c.BeginCommandWithAction("command-1", "Update the plan", "", active)
	require.NoError(t, err)
	c.Observe(agent.ResponseDelta{Text: "An active weekend."})
	c.Observe(agent.Responded{})
	saved(t, c)

	conflicts := func(c *Conversation) {
		t.Helper()
		_, err := c.BeginCommandWithAction("command-1", "Update the plan", "", relaxed)
		require.ErrorIs(t, err, ErrCommandConflict, "other values are another command")
		_, err = c.BeginCommand("command-1", "Update the plan", "")
		require.ErrorIs(t, err, ErrCommandConflict, "the same words without the action are another command")
		replay, err := c.BeginCommandWithAction("command-1", "Update the plan", "", active)
		require.NoError(t, err)
		require.True(t, replay.Duplicate)
	}
	conflicts(c)
	c.Release()
	service.Close()

	// A restart keeps nothing local: the action is read back from Stream with the words.
	recovered := NewForChat(client)
	t.Cleanup(recovered.Close)
	c, _, _, err = recovered.OpenForCaller(t.Context(), "customer", "agent", c.CID(), "employee")
	require.NoError(t, err)
	conflicts(c)
}

func TestAnActionTellsTheModelWhichReplyItWasIn(t *testing.T) {
	_, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "juno", "")
	require.NoError(t, err)
	plan, err := c.BeginCommand("command-1", "Plan a weekend in Ohrid", "")
	require.NoError(t, err)
	c.Observe(agent.ResponseDelta{Text: "A weekend in Ohrid, by the lake.\n```openui-lang\nroot = Card([])\n```"})
	c.Observe(agent.Responded{})
	saved(t, c)
	other, _, _, err := s.Open(context.Background(), "customer", "juno", "")
	require.NoError(t, err)
	elsewhere, err := other.BeginCommand("command-1", "Plan a weekend in Paris", "")
	require.NoError(t, err)
	other.Observe(agent.ResponseDelta{Text: "A weekend in Paris."})
	other.Observe(agent.Responded{})
	saved(t, other)

	action := map[string]any{"message_id": plan.AssistantMessageID, "format": "openui", "name": "update_plan", "payload": map[string]any{"pace": "relaxed"}}
	require.Equal(t, "A weekend in Ohrid, by the lake.", c.ActedOn(t.Context(), action))
	require.Empty(t, c.ActedOn(t.Context(), map[string]any{"message_id": plan.UserMessageID}), "a person's message is no reply")
	require.Empty(t, c.ActedOn(t.Context(), map[string]any{"message_id": elsewhere.AssistantMessageID}), "another conversation's reply is not read")
	require.Empty(t, c.ActedOn(t.Context(), map[string]any{"message_id": "gone"}))
	require.Equal(t,
		`(They did this in the openui UI of your earlier reply "A weekend in Ohrid, by the lake.": update_plan, with {"pace":"relaxed"})`,
		DescribeUIAction(action, c.ActedOn(t.Context(), action)))
	long := DescribeUIAction(action, strings.Repeat("Ohrid ", 100))
	require.Contains(t, long, `"`+strings.Repeat("Ohrid ", 33)+"Oh…\"")

	// History names the reply too.
	_, err = c.BeginCommandWithAction("command-2", "Update the plan", "", action)
	require.NoError(t, err)
	c.Observe(agent.ResponseDelta{Text: "A relaxed weekend."})
	c.Observe(agent.Responded{})
	saved(t, c)
	page, err := s.History(t.Context(), "customer", "juno", c.CID(), "")
	require.NoError(t, err)
	history, _ := history(page)
	require.Contains(t, history[2].Content, `your earlier reply "A weekend in Ohrid, by the lake."`)
}
