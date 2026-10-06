package conversation

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/suite"
)

// PartsSuite covers a reply's steps: how rounds of thinking and tool calls become its
// ai_reasoning and ai_tool_call attachments.
type PartsSuite struct {
	suite.Suite
	c *Conversation
	m *Message
}

func TestPartsSuite(t *testing.T) { suite.Run(t, new(PartsSuite)) }

// SetupTest is a reply to a command from Alice's phone, in a conversation whose agent config
// shows its athena_* tools and web search (visible_tools), with one tool her device runs.
func (s *PartsSuite) SetupTest() {
	s.c = &Conversation{data: ledger{
		Commands:     map[string]commandRecord{"command": {Initiator: "emp_alice", ClientID: "ios-7F3A"}},
		VisibleTools: []string{"athena_*", "web_search"},
	}}
	s.c.tools = map[string]ToolDisplay{"athena_device_location": {Title: "Checking your location", Client: true}}
	s.m = &Message{CommandID: "command", Role: "assistant"}
}

func (s *PartsSuite) TestEachRoundOfThinkingIsAStep() {
	start := time.Now()
	s.True(s.c.thought(s.m, "Needs the user's location first. Then the weather.", start))
	s.False(s.c.thought(s.m, " More.", start.Add(time.Second)), "the same round is one step")
	s.Equal("streaming", s.m.Parts[0].Status)

	s.True(s.c.settleThinking(s.m, start.Add(2*time.Second)))
	s.c.called(s.m, toolCall{id: "call-1", name: "web_search", startedAt: start})
	s.True(s.c.thought(s.m, "Now that I know the city", start.Add(3*time.Second)))

	s.Require().Len(s.m.Parts, 3)
	first := s.m.Parts[0]
	s.Equal("r1", first.ID)
	s.Equal("completed", first.Status)
	s.Equal("Needs the user's location first.", first.Summary)
	s.Equal("Needs the user's location first. Then the weather. More.", first.Preview)
	s.EqualValues(1000, first.DurationMS)
	s.Equal(partToolCall, s.m.Parts[1].Type)
	s.Equal("server", s.m.Parts[1].Executor)
	s.Equal("Searching the web", s.m.Parts[1].DisplayTitle)
	s.Equal("r2", s.m.Parts[2].ID)
	s.Equal("r2", s.c.reasoning.id, "windows follow the streaming step")
}

func (s *PartsSuite) TestTheSummaryIsTheFirstSentence() {
	s.Equal("We need to solve a puzzle.", summaryOf("We need to solve a puzzle. Let me parse."))
	s.Equal("Five engineers", summaryOf("  Five engineers\n4 weeks"))
	s.Equal("Is 2^30 bigger?", summaryOf("Is 2^30 bigger? Compute."))
	s.Equal("3.14 is pi", summaryOf("3.14 is pi"), "a decimal point does not end a sentence")
	long := summaryOf(strings.Repeat("word ", 60))
	s.Equal(maxSummary, utf8.RuneCountInString(long))
	s.True(strings.HasSuffix(long, "…"))
}

func (s *PartsSuite) TestTheStoredPreviewIsTheOpening() {
	now := time.Now()
	s.c.thought(s.m, strings.Repeat("considering ", 100)+"THE END", now)
	s.c.settleThinking(s.m, now)
	preview := s.m.Parts[0].Preview
	s.Equal(maxStoredPreview, utf8.RuneCountInString(preview))
	s.True(strings.HasPrefix(preview, "considering considering"))
	s.True(strings.HasSuffix(preview, "…"))
	s.NotContains(preview, "THE END", "only the opening is stored")
}

func (s *PartsSuite) TestAClientToolAwaitsItsPersonsDevice() {
	now := time.Now()
	s.c.called(s.m, toolCall{id: "toolu_01A", name: "athena_device_location", arguments: `{"purpose":"weather"}`, startedAt: now})
	part := s.m.Parts[0]
	s.Equal("awaiting_client", part.Status)
	s.Equal("client", part.Executor)
	s.Equal("Checking your location", part.DisplayTitle)
	s.Equal("emp_alice", part.TargetUserID)
	s.Equal("ios-7F3A", part.TargetClientID)
	s.JSONEq(`{"purpose":"weather"}`, string(part.Arguments))

	ran(s.m, "toolu_01A", "completed", `{"summary":"Shared approximate location","city":"Skopje"}`, "", now.Add(2*time.Second))
	s.Equal("completed", s.m.Parts[0].Status)
	s.Equal("Shared approximate location", s.m.Parts[0].Summary)
	s.EqualValues(2000, s.m.Parts[0].DurationMS)
	raw, _ := json.Marshal(s.m.Parts)
	s.NotContains(string(raw), "Skopje", "a client tool's result is the model's, not the channel's")

	// Without an install to address, nobody waits on it, and large arguments stay hidden.
	s.c.data.Commands["command"] = commandRecord{Initiator: "emp_alice"}
	s.c.called(s.m, toolCall{id: "toolu_01B", name: "athena_device_location", arguments: `{"purpose":"` + strings.Repeat("x", 600) + `"}`, startedAt: now})
	s.Equal("running", s.m.Parts[1].Status)
	s.Empty(s.m.Parts[1].Arguments)

	// A device that could not run it says why.
	ran(s.m, "toolu_01B", "failed", "That did not work", "Location not shared", now)
	s.Equal("failed", s.m.Parts[1].Status)
	s.Equal("Location not shared", s.m.Parts[1].Summary)
}

func (s *PartsSuite) TestAClientToolIsAStepWhateverVisibleToolsSays() {
	s.c.data.VisibleTools = nil
	s.c.called(s.m, toolCall{id: "toolu_01A", name: "athena_device_location", startedAt: time.Now()})
	s.Require().Len(s.m.Parts, 1)
	s.Equal("awaiting_client", s.m.Parts[0].Status)
}

func (s *PartsSuite) TestACallAPersonMustAllowWaitsForTheirAnswer() {
	s.c.tools["athena_device_location"] = ToolDisplay{Title: "Checking your location", Client: true, Approval: &ToolApproval{
		Title: "Share your location?", Message: "Only your city is shared.", ReasonArgument: "purpose",
		AllowTitle: "Share location", DeclineTitle: "Don't share",
	}}
	now := time.Now()
	s.c.called(s.m, toolCall{id: "toolu_01A", name: "athena_device_location", arguments: `{"purpose":"  to check\nthe weather "}`, startedAt: now})
	part := s.m.Parts[0]
	s.Equal("awaiting_approval", part.Status)
	s.Equal("emp_alice", part.TargetUserID)
	s.Equal("ios-7F3A", part.TargetClientID)
	s.Equal(&Approval{Title: "Share your location?", Message: "Only your city is shared.", Reason: "to check the weather",
		AllowTitle: "Share location", DeclineTitle: "Don't share"}, part.Approval)

	s.False(decided(s.m, "toolu_other", true, "", now), "only the call waiting is answered")
	s.True(decided(s.m, "toolu_01A", true, "", now))
	s.Equal("awaiting_client", s.m.Parts[0].Status, "allowed, the device runs it")
	s.Equal("allowed", s.m.Parts[0].Approval.Decision)
	s.False(decided(s.m, "toolu_01A", false, "Location not shared", now), "an answer is given once")
	ran(s.m, "toolu_01A", "completed", `{"summary":"Shared approximate location"}`, "", now.Add(time.Second))
	s.Equal("completed", s.m.Parts[0].Status)
	s.Equal("Shared approximate location", s.m.Parts[0].Summary)

	// Declined, the call is cancelled with the person's summary, and its result changes nothing.
	s.c.called(s.m, toolCall{id: "toolu_01B", name: "athena_device_location", arguments: `{}`, startedAt: now})
	s.Empty(s.m.Parts[1].Approval.Reason)
	s.True(decided(s.m, "toolu_01B", false, "Location not shared", now.Add(2*time.Second)))
	s.Equal("cancelled", s.m.Parts[1].Status)
	s.Equal("declined", s.m.Parts[1].Approval.Decision)
	s.Equal("Location not shared", s.m.Parts[1].Summary)
	s.EqualValues(2000, s.m.Parts[1].DurationMS)
	ran(s.m, "toolu_01B", "failed", "", "The employee declined", now.Add(3*time.Second))
	s.Equal("cancelled", s.m.Parts[1].Status)

	// Nobody answered: the caller's reason is the summary.
	s.c.called(s.m, toolCall{id: "toolu_01C", name: "athena_device_location", startedAt: now})
	ran(s.m, "toolu_01C", "failed", "", "Not answered in time", now)
	s.Equal("failed", s.m.Parts[2].Status)
	s.Equal("Not answered in time", s.m.Parts[2].Summary)

	// A client tool with no install to address has nobody to ask.
	s.c.data.Commands["command"] = commandRecord{Initiator: "emp_alice"}
	s.c.called(s.m, toolCall{id: "toolu_01D", name: "athena_device_location", startedAt: now})
	s.Equal("running", s.m.Parts[3].Status)
	s.Nil(s.m.Parts[3].Approval)
}

func (s *PartsSuite) TestAServerToolAPersonMustAllowIsShownAndRunsOnceAllowed() {
	s.c.data.VisibleTools = nil
	s.c.tools["send_email"] = ToolDisplay{Title: "Sending the email", Approval: &ToolApproval{
		Title: "Send this email?", ReasonArgument: "subject",
	}}
	now := time.Now()
	s.c.called(s.m, toolCall{id: "call-1", name: "send_email", arguments: `{"subject":"Q3 numbers","to":"a@b.c"}`, startedAt: now})
	s.Require().Len(s.m.Parts, 1, "a question is shown whatever visible_tools says")
	part := s.m.Parts[0]
	s.Equal("awaiting_approval", part.Status)
	s.Equal("server", part.Executor)
	s.Equal("emp_alice", part.TargetUserID)
	s.Empty(part.TargetClientID, "any of the person's clients may answer")
	s.Empty(part.Arguments, "only the reason of a server tool's arguments is shown")
	s.Equal("Q3 numbers", part.Approval.Reason)
	s.True(decided(s.m, "call-1", true, "", now))
	s.Equal("running", s.m.Parts[0].Status)
}

func (s *PartsSuite) TestToolsPeopleMayNotSeeAreNotSteps() {
	s.c.called(s.m, toolCall{id: "call-1", name: "crm_update_contact", arguments: `{"email":"a@b.c"}`, startedAt: time.Now()})
	s.Empty(s.m.Parts)
	s.c.called(s.m, toolCall{id: "call-2", name: "athena_save_canvas", arguments: `{"title":"x"}`, startedAt: time.Now()})
	s.Require().Len(s.m.Parts, 1)
	s.Empty(s.m.Parts[0].Arguments, "a server tool's arguments are never shown")
}

func (s *PartsSuite) TestWithoutVisibleToolsOnlySearchIsAStep() {
	s.c.data.VisibleTools = nil
	s.c.called(s.m, toolCall{id: "call-1", name: "athena_save_canvas", startedAt: time.Now()})
	s.Empty(s.m.Parts, "a tool the agent config does not show is not a step")
	s.c.called(s.m, toolCall{id: "call-2", name: "search", startedAt: time.Now()})
	s.Require().Len(s.m.Parts, 1)
	s.Equal("Searching the web", s.m.Parts[0].DisplayTitle)
}

func (s *PartsSuite) TestAFinishedReplyEndsItsSteps() {
	now := time.Now()
	s.c.called(s.m, toolCall{id: "toolu_01A", name: "athena_device_location", startedAt: now})
	s.c.thought(s.m, "Still thinking", now)
	s.c.stopParts(s.m, now)
	s.Equal("cancelled", s.m.Parts[0].Status)
	s.Equal("completed", s.m.Parts[1].Status)
	s.Equal("Still thinking", s.m.Parts[1].Summary)
}

func (s *PartsSuite) TestAttachmentsStayWithinStreamsLimits() {
	var parts []Part
	for i := 0; i < 20; i++ {
		parts = append(parts, Part{Type: partReasoning, V: 1, ID: fmt.Sprintf("r%d", i+1), Status: "completed",
			Summary: strings.Repeat("s", 120), Preview: strings.Repeat("p", maxStoredPreview), DurationMS: 1000})
		parts = append(parts, Part{Type: partToolCall, V: 1, ID: fmt.Sprintf("call-%d", i), Status: "completed",
			Name: "web_search", DisplayTitle: "Searching the web", Executor: "server"})
	}
	parts = append(parts, Part{Type: partToolCall, V: 1, ID: "toolu_asked", Status: "cancelled", Name: "athena_device_location", Executor: "client",
		Approval: &Approval{Title: "Share your location?", Message: strings.Repeat("m", 240), Reason: strings.Repeat("r", 160), Decision: "declined"}})
	parts = append(parts, Part{Type: partToolCall, V: 1, ID: "toolu_waiting", Status: "awaiting_approval", Name: "athena_device_location", Executor: "client",
		Approval: &Approval{Title: "Share your location?", Message: strings.Repeat("m", 240), Reason: strings.Repeat("r", 160)}})
	parts = append(parts, Part{Type: partToolCall, V: 1, ID: "toolu_live", Status: "awaiting_client", Name: "athena_device_location", Executor: "client"})
	artifacts := partialAttachments([]ArtifactAttachment{{Type: "athena_image", ArtifactID: "art_1", Revision: 1, Title: "Chart"}})

	out := messageAttachments(parts, artifacts)
	raw, _ := json.Marshal(out)
	s.LessOrEqual(len(raw), maxAttachmentBytes)
	s.LessOrEqual(len(out), maxMessageAttachments)
	s.Equal("athena_image", out[len(out)-1]["type"], "artifacts are kept, after the steps")
	s.Equal("art_1", out[len(out)-1]["artifact_id"], "an artifact's fields are on the attachment itself")
	s.Equal("toolu_live", out[len(out)-2]["id"], "a step still in progress is kept")
	waiting := out[len(out)-3]
	s.Equal("toolu_waiting", waiting["id"], "a question still open is kept whole")
	s.Equal(strings.Repeat("r", 160), waiting["approval"].(map[string]any)["reason"])

	// A short reply keeps everything as it is.
	short := messageAttachments(parts[:3], artifacts)
	s.Len(short, 4)
	s.Equal(strings.Repeat("p", maxStoredPreview), short[0]["preview"])
}

func (s *PartsSuite) TestAStreamingStepShowsItsLatestThoughts() {
	start := time.Now()
	s.c.thought(s.m, strings.Repeat("a", 300)+"LATEST", start)
	s.c.reasoning.add("!", start.Add(1500*time.Millisecond))
	live := liveParts(s.m.Parts, s.c.reasoning.snapshot())
	s.Equal(maxLivePreview, utf8.RuneCountInString(live[0].Preview))
	s.True(strings.HasSuffix(live[0].Preview, "LATEST!"))
	s.EqualValues(1500, live[0].DurationMS)
	s.Empty(s.m.Parts[0].Preview, "the live preview is never stored")
}
