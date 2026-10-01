package conversation

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

func partsConversation() (*Conversation, *Message) {
	c := &Conversation{data: disk{Commands: map[string]commandRecord{
		"command": {Initiator: "emp_alice", ClientID: "ios-7F3A"},
	}}}
	c.tools = map[string]ToolDisplay{"athena_device_location": {Title: "Checking your location", Client: true}}
	return c, &Message{CommandID: "command", Role: "assistant"}
}

func TestEachRoundOfThinkingIsAStep(t *testing.T) {
	c, m := partsConversation()
	start := time.Now()
	require.True(t, c.thought(m, "Needs the user's location first. Then the weather.", start))
	require.False(t, c.thought(m, " More.", start.Add(time.Second)), "the same round is one step")
	require.Equal(t, "streaming", m.Parts[0].Status)

	require.True(t, c.settleThinking(m, start.Add(2*time.Second)))
	c.called(m, toolCall{id: "call-1", name: "web_search", startedAt: start})
	require.True(t, c.thought(m, "Now that I know the city", start.Add(3*time.Second)))

	require.Len(t, m.Parts, 3)
	first := m.Parts[0]
	require.Equal(t, "r1", first.ID)
	require.Equal(t, "completed", first.Status)
	require.Equal(t, "Needs the user's location first.", first.Summary)
	require.Equal(t, "Needs the user's location first. Then the weather. More.", first.Preview)
	require.EqualValues(t, 1000, first.DurationMS)
	require.Equal(t, partToolCall, m.Parts[1].Type)
	require.Equal(t, "server", m.Parts[1].Executor)
	require.Equal(t, "r2", m.Parts[2].ID)
	require.Equal(t, "r2", c.reasoning.id, "windows follow the streaming step")
}

func TestTheSummaryIsTheFirstSentence(t *testing.T) {
	require.Equal(t, "We need to solve a puzzle.", summaryOf("We need to solve a puzzle. Let me parse."))
	require.Equal(t, "Five engineers", summaryOf("  Five engineers\n4 weeks"))
	require.Equal(t, "Is 2^30 bigger?", summaryOf("Is 2^30 bigger? Compute."))
	require.Equal(t, "3.14 is pi", summaryOf("3.14 is pi"), "a decimal point does not end a sentence")
	long := summaryOf(strings.Repeat("word ", 60))
	require.Equal(t, maxSummary, utf8.RuneCountInString(long))
	require.True(t, strings.HasSuffix(long, "…"))
}

func TestTheStoredPreviewIsTheOpening(t *testing.T) {
	c, m := partsConversation()
	now := time.Now()
	c.thought(m, strings.Repeat("considering ", 100)+"THE END", now)
	c.settleThinking(m, now)
	preview := m.Parts[0].Preview
	require.Equal(t, maxStoredPreview, utf8.RuneCountInString(preview))
	require.True(t, strings.HasPrefix(preview, "considering considering"))
	require.True(t, strings.HasSuffix(preview, "…"))
	require.NotContains(t, preview, "THE END", "only the opening is stored")
}

func TestAClientToolAwaitsItsPersonsDevice(t *testing.T) {
	c, m := partsConversation()
	now := time.Now()
	c.called(m, toolCall{id: "toolu_01A", name: "athena_device_location", arguments: `{"purpose":"weather"}`, startedAt: now})
	part := m.Parts[0]
	require.Equal(t, "awaiting_client", part.Status)
	require.Equal(t, "client", part.Executor)
	require.Equal(t, "Checking your location", part.DisplayTitle)
	require.Equal(t, "emp_alice", part.TargetUserID)
	require.Equal(t, "ios-7F3A", part.TargetClientID)
	require.JSONEq(t, `{"purpose":"weather"}`, string(part.Arguments))

	ran(m, "toolu_01A", "completed", `{"summary":"Shared approximate location","city":"Skopje"}`, "", now.Add(2*time.Second))
	require.Equal(t, "completed", m.Parts[0].Status)
	require.Equal(t, "Shared approximate location", m.Parts[0].Summary)
	require.EqualValues(t, 2000, m.Parts[0].DurationMS)
	raw, _ := json.Marshal(m.Parts)
	require.NotContains(t, string(raw), "Skopje", "a client tool's result is the model's, not the channel's")

	// Without an install to address, nobody waits on it, and large arguments stay hidden.
	c.data.Commands["command"] = commandRecord{Initiator: "emp_alice"}
	c.called(m, toolCall{id: "toolu_01B", name: "athena_device_location", arguments: `{"purpose":"` + strings.Repeat("x", 600) + `"}`, startedAt: now})
	require.Equal(t, "running", m.Parts[1].Status)
	require.Empty(t, m.Parts[1].Arguments)

	// A device that could not run it says why.
	ran(m, "toolu_01B", "failed", "That did not work", "Location not shared", now)
	require.Equal(t, "failed", m.Parts[1].Status)
	require.Equal(t, "Location not shared", m.Parts[1].Summary)
}

func TestToolsPeopleMayNotSeeAreNotSteps(t *testing.T) {
	c, m := partsConversation()
	c.called(m, toolCall{id: "call-1", name: "crm_update_contact", arguments: `{"email":"a@b.c"}`, startedAt: time.Now()})
	require.Empty(t, m.Parts)
	c.called(m, toolCall{id: "call-2", name: "athena_save_canvas", arguments: `{"title":"x"}`, startedAt: time.Now()})
	require.Len(t, m.Parts, 1)
	require.Empty(t, m.Parts[0].Arguments, "a server tool's arguments are never shown")
}

func TestAFinishedReplyEndsItsSteps(t *testing.T) {
	c, m := partsConversation()
	now := time.Now()
	c.called(m, toolCall{id: "toolu_01A", name: "athena_device_location", startedAt: now})
	c.thought(m, "Still thinking", now)
	c.stopParts(m, now)
	require.Equal(t, "cancelled", m.Parts[0].Status)
	require.Equal(t, "completed", m.Parts[1].Status)
	require.Equal(t, "Still thinking", m.Parts[1].Summary)
}

func TestAttachmentsStayWithinStreamsLimits(t *testing.T) {
	var parts []Part
	for i := 0; i < 20; i++ {
		parts = append(parts, Part{Type: partReasoning, V: 1, ID: fmt.Sprintf("r%d", i+1), Status: "completed",
			Summary: strings.Repeat("s", 120), Preview: strings.Repeat("p", maxStoredPreview), DurationMS: 1000})
		parts = append(parts, Part{Type: partToolCall, V: 1, ID: fmt.Sprintf("call-%d", i), Status: "completed",
			Name: "web_search", DisplayTitle: "Searching the web", Executor: "server"})
	}
	parts = append(parts, Part{Type: partToolCall, V: 1, ID: "toolu_live", Status: "awaiting_client", Name: "athena_device_location", Executor: "client"})
	artifacts := []map[string]any{{"type": "athena_image", "artifact_id": "art_1", "revision": 1, "title": "Chart"}}

	out := messageAttachments(parts, artifacts)
	raw, _ := json.Marshal(out)
	require.LessOrEqual(t, len(raw), maxAttachmentBytes)
	require.LessOrEqual(t, len(out), maxMessageAttachments)
	require.Equal(t, "athena_image", out[len(out)-1]["type"], "artifacts are kept, after the steps")
	require.Equal(t, "toolu_live", out[len(out)-2]["id"], "a step still in progress is kept")

	// A short reply keeps everything as it is.
	short := messageAttachments(parts[:3], artifacts)
	require.Len(t, short, 4)
	require.Equal(t, strings.Repeat("p", maxStoredPreview), short[0]["preview"])
}

func TestAStreamingStepShowsItsLatestThoughts(t *testing.T) {
	c, m := partsConversation()
	start := time.Now()
	c.thought(m, strings.Repeat("a", 300)+"LATEST", start)
	c.reasoning.add("!", start.Add(1500*time.Millisecond))
	live := liveParts(m.Parts, c.reasoning.snapshot())
	require.Equal(t, maxLivePreview, utf8.RuneCountInString(live[0].Preview))
	require.True(t, strings.HasSuffix(live[0].Preview, "LATEST!"))
	require.EqualValues(t, 1500, live[0].DurationMS)
	require.Empty(t, m.Parts[0].Preview, "the live preview is never stored")
}
