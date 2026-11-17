package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// noteTurn is a conversation whose last turn carries a note after what the caller said.
func noteTurn() llm.ResponseParams {
	return llm.ResponseParams{
		Instructions: "You are a support engineer.",
		Input: []llm.Message{
			{Role: llm.User, Content: "Why did the webhook fail?"},
			{Role: llm.Assistant, Content: "Let me check."},
			{Role: llm.User, Content: "Any news?"},
			{Role: llm.System, Content: "Your colleague has come back."},
		},
	}
}

func roles(body map[string]any) []string {
	var out []string
	for _, message := range body["messages"].([]any) {
		out = append(out, message.(map[string]any)["role"].(string))
	}
	return out
}

func (s *AnthropicSuite) TestALaterSystemTurnIsASystemMessageOnAModelThatTakesOne() {
	_, err := s.collect(s.provider(Options{Model: "claude-opus-5"}), noteTurn())
	s.Require().NoError(err)

	body := s.sent()[0].body
	s.Equal([]string{"user", "assistant", "user", "system"}, roles(body))
	s.Len(body["system"].([]any), 1, "the note stays out of the system prompt")
}

func (s *AnthropicSuite) TestALaterSystemTurnIsToldOnTheUsersSideOnOneThatDoesNot() {
	for _, model := range []string{"claude-sonnet-5-5", "claude-haiku-5-5", "claude-sonnet-5", "claude-sonnet-5-20260301", "claude-unknown-9"} {
		s.SetupTest()
		_, err := s.collect(s.provider(Options{Model: model}), noteTurn())
		s.Require().NoError(err)

		body := s.sent()[0].body
		s.Equal([]string{"user", "assistant", "user"}, roles(body), model)
		s.Contains(string(s.sent()[0].raw), "Your colleague has come back.", model)
	}
}

func (s *AnthropicSuite) TestASystemTurnTheAPIWouldRefuseThereIsToldOnTheUsersSide() {
	params := noteTurn()
	// Followed by the caller rather than the assistant.
	params.Input = append(params.Input, llm.Message{Role: llm.User, Content: "Hello?"})
	_, err := s.collect(s.provider(Options{}), params)
	s.Require().NoError(err)

	s.Equal([]string{"user", "assistant", "user"}, roles(s.sent()[0].body))
}

func (s *AnthropicSuite) TestConsecutiveSystemTurnsAreOneMessage() {
	params := noteTurn()
	params.Input = append(params.Input, llm.Message{Role: llm.System, Content: "Answer now."})
	_, err := s.collect(s.provider(Options{}), params)
	s.Require().NoError(err)

	s.Equal([]string{"user", "assistant", "user", "system"}, roles(s.sent()[0].body))
}

func (s *AnthropicSuite) TestASystemTurnIsNeverABreakpoint() {
	_, err := s.collect(s.provider(Options{}), noteTurn())
	s.Require().NoError(err)

	messages := s.sent()[0].body["messages"].([]any)
	s.Nil(lastBlockOf(messages[3])["cache_control"])
	s.Equal(ephemeral, lastBlockOf(messages[2])["cache_control"], "the caller's turn before the note ends the cached prefix")
}

func (s *AnthropicSuite) TestATrailingSystemTurnLeavesTheCachedPrefixBytesUnchanged() {
	provider := s.provider(Options{})
	without := noteTurn()
	without.Input = without.Input[:3]
	_, err := s.collect(provider, without)
	s.Require().NoError(err)
	_, err = s.collect(provider, noteTurn())
	s.Require().NoError(err)

	var first, second struct {
		System   json.RawMessage   `json:"system"`
		Messages []json.RawMessage `json:"messages"`
	}
	s.Require().NoError(json.Unmarshal(s.sent()[0].raw, &first))
	s.Require().NoError(json.Unmarshal(s.sent()[1].raw, &second))
	s.Equal(string(first.System), string(second.System))
	for i, message := range first.Messages {
		s.Equal(string(message), string(second.Messages[i]), "message %d", i)
	}
	s.True(strings.Contains(string(second.Messages[3]), `"system"`))
}

func (s *AnthropicSuite) TestASystemTurnBeforeAnEmptyAssistantIsToldOnTheUsersSide() {
	// The empty turn is sent as nothing, which would leave the system message followed by the
	// caller's: the API refuses that.
	_, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{
		{Role: llm.User, Content: "hello"},
		{Role: llm.System, Content: "Your colleague has come back."},
		{Role: llm.Assistant},
		{Role: llm.User, Content: "are you there?"},
	}})
	s.Require().NoError(err)

	s.Equal([]string{"user"}, roles(s.sent()[0].body))
}
