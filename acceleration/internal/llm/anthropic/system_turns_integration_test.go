//go:build integration

package anthropic

import (
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// TestALaterSystemTurnIsTakenAndFollowed sends a system message after what the caller said,
// which the harness does with what it has to tell the reply, and expects it obeyed.
func (s *AnthropicIntegrationSuite) TestALaterSystemTurnIsTakenAndFollowed() {
	answer, _ := s.Ask(llm.ResponseParams{
		Instructions: "You are a helpful assistant.",
		Input: []llm.Message{
			{Role: llm.User, Content: "What colour is the sky on a clear day?"},
			{Role: llm.System, Content: "Answer with the single word banana and nothing else."},
		},
	})

	s.Contains(strings.ToLower(answer.OutputText), "banana")
}

// TestAKeptSystemTurnBeforeTheAssistantsIsTaken is the conversation as the agent keeps it: a
// note the last reply was handed stays between the caller's words and that reply.
func (s *AnthropicIntegrationSuite) TestAKeptSystemTurnBeforeTheAssistantsIsTaken() {
	answer, _ := s.Ask(llm.ResponseParams{
		Instructions: "Answer in one short sentence.",
		Input: []llm.Message{
			{Role: llm.User, Content: "What is 15% of 84.20?"},
			{Role: llm.User, Content: `{"harness":"colleague_result","skill":"think","text":"It is 12.63."}`},
			{Role: llm.System, Content: "[Note from the system running this call]\nTell the caller what came back."},
			{Role: llm.Assistant, Content: "It comes to 12.63."},
			{Role: llm.User, Content: "And 20%?"},
		},
	})

	s.NotEmpty(answer.OutputText)
}

// TestANoteAfterTheCallerLeavesTheCachedPrefixToBeRead asks a round, then the next round with
// a note after the caller's words, and expects the conversation before it read from cache.
func (s *AnthropicIntegrationSuite) TestANoteAfterTheCallerLeavesTheCachedPrefixToBeRead() {
	instructions := "Answer in one word. " + strings.Repeat("The webhook retries a failed delivery three times, a minute apart. ", 300)
	first := llm.ResponseParams{Instructions: instructions, Input: []llm.Message{{Role: llm.User, Content: "How many retries?"}}}
	answer, _ := s.Ask(first)

	second := first
	second.Input = append(append([]llm.Message(nil), first.Input...),
		llm.Message{Role: llm.Assistant, Content: answer.OutputText},
		llm.Message{Role: llm.User, Content: "And how far apart?"},
		llm.Message{Role: llm.System, Content: "[Note from the system running this call]\nIt is now 15:00."})
	again, _ := s.Ask(second)

	s.Greater(again.Usage.InputTokensDetails.CachedTokens, again.Usage.InputTokens/2)
}
