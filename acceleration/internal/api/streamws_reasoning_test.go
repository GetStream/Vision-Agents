package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// RespondReasoningSuite follows an assistant turn's thinking through the language-model
// socket: out on complete, back in on respond, and on to Claude as signed blocks.
type RespondReasoningSuite struct {
	suite.Suite
}

func TestRespondReasoningSuite(t *testing.T) {
	suite.Run(t, new(RespondReasoningSuite))
}

// claudeRequest is the Messages API body the anthropic driver sends for a respond frame.
func (s *RespondReasoningSuite) claudeRequest(frame string) map[string]any {
	var decoded respond
	s.Require().NoError(json.Unmarshal([]byte(frame), &decoded))
	params, err := decoded.params(options.LLM{})
	s.Require().NoError(err)

	sent := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		sent <- raw
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer server.Close()
	provider, err := anthropic.New(anthropic.Options{APIKey: "test", BaseURL: server.URL, Model: "claude-opus-5-5"})
	s.Require().NoError(err)
	stream, err := provider.Create(s.T().Context(), params)
	s.Require().NoError(err)
	_, _ = llm.Collect(stream)

	var body map[string]any
	s.Require().NoError(json.Unmarshal(<-sent, &body))
	return body
}

func (s *RespondReasoningSuite) TestReplayedReasoningReachesClaudeAsSignedBlocksInPlace() {
	body := s.claudeRequest(`{"type":"respond","id":"r2","messages":[
		{"role":"user","content":"find roses"},
		{"role":"assistant","content":"Looking.",
		 "tool_calls":[{"id":"toolu_1","name":"lookup","arguments":"{}"},{"id":"toolu_2","name":"clock","arguments":"{}"}],
		 "reasoning":[{"text":"Plan.","signature":"sig-1"},{"redacted":"enc-1"},{"signature":"sig-2","after_tool_calls":1}]},
		{"role":"tool","tool_call_id":"toolu_1","content":"roses"},
		{"role":"tool","tool_call_id":"toolu_2","content":"noon"}
	]}`)

	turn, err := json.Marshal(body["messages"].([]any)[1])
	s.Require().NoError(err)
	s.JSONEq(`{"role": "assistant", "content": [
		{"type": "thinking", "thinking": "Plan.", "signature": "sig-1"},
		{"type": "redacted_thinking", "data": "enc-1"},
		{"type": "text", "text": "Looking."},
		{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
		{"type": "thinking", "thinking": "", "signature": "sig-2"},
		{"type": "tool_use", "id": "toolu_2", "name": "clock", "input": {}}
	]}`, string(turn))
}

func (s *RespondReasoningSuite) TestReasoningOnAUserMessageIsRefused() {
	var decoded respond
	s.Require().NoError(json.Unmarshal([]byte(`{"messages":[{"role":"user","content":"hi","reasoning":[{"signature":"sig"}]}]}`), &decoded))

	_, err := decoded.params(options.LLM{})

	s.ErrorContains(err, "reasoning belongs to an assistant message")
}

func (s *RespondReasoningSuite) TestCompleteCarriesTheSignedThinking() {
	sent, ok := llmFrame(llm.ResponseCompleted{Response: llm.Response{ID: "r1", ReasoningBlocks: []llm.ReasoningBlock{
		{Text: "Plan.", Signature: "sig-1"},
		{Redacted: "enc-1"},
		{Signature: "sig-2", AfterToolCalls: 1},
	}}})
	s.Require().True(ok)

	raw, err := json.Marshal(sent["reasoning"])
	s.Require().NoError(err)
	s.JSONEq(`[
		{"text": "Plan.", "signature": "sig-1"},
		{"redacted": "enc-1"},
		{"signature": "sig-2", "after_tool_calls": 1}
	]`, string(raw))
}

func (s *RespondReasoningSuite) TestCompleteWithoutThinkingHasNoReasoningField() {
	sent, ok := llmFrame(llm.ResponseCompleted{Response: llm.Response{ID: "r1", ReasoningText: "unsigned"}})
	s.Require().True(ok)

	s.NotContains(sent, "reasoning")
}
