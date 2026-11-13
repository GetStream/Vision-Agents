package agent

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type UIStreamSuite struct {
	suite.Suite
}

func TestUIStreamSuite(t *testing.T) {
	suite.Run(t, new(UIStreamSuite))
}

func streamUI(turnID string, deltas ...string) (string, []Event, string) {
	var stream uiStream
	var prose strings.Builder
	var events []Event

	for _, delta := range deltas {
		words, emitted := stream.Add(turnID, delta)
		prose.WriteString(words)
		events = append(events, emitted...)
	}
	tail, flushed := stream.Flush(turnID)
	prose.WriteString(tail)
	events = append(events, flushed...)
	return prose.String(), events, stream.HistoryBlock()
}

func joinedUIDeltas(events []Event) string {
	var out strings.Builder
	for _, ev := range events {
		if delta, ok := ev.(UIDelta); ok {
			out.WriteString(delta.Delta)
		}
	}
	return out.String()
}

func (s *UIStreamSuite) TestPlainProseStreamsWithoutBuffering() {
	var stream uiStream
	words, events := stream.Add("turn-1", "Kaneo has a cliffside table at 19:15.")

	s.Equal("Kaneo has a cliffside table at 19:15.", words)
	s.Empty(events)
}

func (s *UIStreamSuite) TestAnOpenUIFenceStreamsImmediatelyWithoutWaitingForClose() {
	var stream uiStream

	words1, events1 := stream.Add("turn-1", "Here is the card.\n```openui-lang\nroot = Card([header, ")
	s.Equal("Here is the card.\n", words1)
	s.Require().Len(events1, 1, "the first UI line streams before the fence closes")
	s.Equal(UIDelta{
		TurnID: "turn-1",
		Format: "openui",
		Mode:   "replace",
		Delta:  "root = Card([header, ",
	}, events1[0])

	words2, events2 := stream.Add("turn-1", "map])\nheader = CardHeader(\"Kaneo\")\n```\nEnjoy!")
	s.Equal("Enjoy!", words2)
	s.Equal("root = Card([header, map])\nheader = CardHeader(\"Kaneo\")",
		joinedUIDeltas(append(events1, events2...)))

	completed, ok := firstOf[UICompleted](events2)
	s.Require().True(ok)
	s.Equal(UICompleted{
		TurnID:  "turn-1",
		Format:  "openui",
		Mode:    "replace",
		Content: "root = Card([header, map])\nheader = CardHeader(\"Kaneo\")",
	}, completed)
}

func (s *UIStreamSuite) TestAFenceSplitSingleCharacterByCharacterProducesExactProseAndUI() {
	full := "Here's a weekend in Ohrid.\n\n```openui-lang\nroot = Card([title])\ntitle = CardHeader(\"Ohrid\")\n```\n\nEnjoy the lake!"
	var chars []string
	for _, r := range full {
		chars = append(chars, string(r))
	}

	prose, events, history := streamUI("turn-1", chars...)

	s.Equal("Here's a weekend in Ohrid.\n\n\nEnjoy the lake!", prose)
	s.Equal("root = Card([title])\ntitle = CardHeader(\"Ohrid\")", joinedUIDeltas(events))
	completed, ok := firstOf[UICompleted](events)
	s.Require().True(ok)
	s.Equal("root = Card([title])\ntitle = CardHeader(\"Ohrid\")", completed.Content)
	s.Equal("```openui-lang\nroot = Card([title])\ntitle = CardHeader(\"Ohrid\")\n```", history)
}

func (s *UIStreamSuite) TestAClosingFenceAtEndOfReplyWithoutTrailingNewlineStillClosesCleanly() {
	prose, events, _ := streamUI(
		"turn-1",
		"A plan.\n```ui:a2ui@1.0\n",
		`{"version":"1.0"}`+"\n```",
	)

	s.Equal("A plan.\n", prose)
	completed, ok := firstOf[UICompleted](events)
	s.Require().True(ok)
	s.Equal("a2ui", completed.Format)
	s.Equal("1.0", completed.Version)
	s.Equal(`{"version":"1.0"}`, completed.Content)
}

func (s *UIStreamSuite) TestOrdinaryCodeBlocksStayInProseAndLaterUIBlocksAreDropped() {
	prose, events, _ := streamUI(
		"turn-1",
		"Code:\n```swift\nlet a = [1, 2]\n```\n```openui-lang\nroot = Card([a])\n```\n```openui-lang\nroot = Card([b])\n```\nDone.",
	)

	s.Equal("Code:\n```swift\nlet a = [1, 2]\n```\nDone.", prose)
	completed, ok := firstOf[UICompleted](events)
	s.Require().True(ok)
	s.Equal("root = Card([a])", completed.Content)
}

func (s *UIStreamSuite) TestAUITagPreservesComparisonOperatorsAndPatchMode() {
	prose, events, history := streamUI(
		"turn-1",
		`Updated to four guests. <ui format="openui" mode="patch">`+"\n",
		"$party = 4\n",
		`note = $party < 6 ? "Terrace" : "Courtyard"`+"\n</ui>",
	)

	s.Equal("Updated to four guests. ", prose)
	completed, ok := firstOf[UICompleted](events)
	s.Require().True(ok)
	s.Equal("openui", completed.Format)
	s.Equal("patch", completed.Mode)
	s.Equal("$party = 4\nnote = $party < 6 ? \"Terrace\" : \"Courtyard\"", completed.Content)
	s.Equal("```openui-patch\n$party = 4\nnote = $party < 6 ? \"Terrace\" : \"Courtyard\"\n```", history)
}

func (s *AgentSuite) TestAVoiceReplyWithOpenUIDoesNotSpeakOrStripBracketedArrays() {
	s.performing = "You may write stage directions in square brackets."
	message, reported := s.repliedWith(
		false,
		"[warmly] Kaneo has a terrace table at seven fifteen.\n",
		"```openui-lang\n",
		"root = Card([header, map, slots])\n",
		"header = CardHeader(\"Kaneo Cliffside\")\n",
		"```",
	)

	s.Equal("Kaneo has a terrace table at seven fifteen.", message)
	spoken := said(s.voice.spoken())
	s.Contains(spoken, "Kaneo has a terrace table at seven fifteen.")
	s.NotContains(spoken, "openui", "the voice never reads UI syntax aloud")
	s.NotContains(spoken, "CardHeader", "the voice never reads UI components aloud")

	completed, ok := firstOf[UICompleted](reported)
	s.Require().True(ok)
	s.Equal("openui", completed.Format)
	s.Equal("root = Card([header, map, slots])\nheader = CardHeader(\"Kaneo Cliffside\")",
		completed.Content, "stage-direction stripping never touches bracketed UI arrays")
}
