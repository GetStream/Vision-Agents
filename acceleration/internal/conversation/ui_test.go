package conversation

import (
	"context"
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
	require.Empty(t, DescribeUIAction(nil))
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
