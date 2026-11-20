//go:build integration

package api

import (
	"crypto/rand"
	"net/http"
	"testing"
	"time"
)

// ToolOutputRetentionSuite covers what a recorded session keeps of its tools once the
// app's policy or the agent's config says how long to keep it.
type ToolOutputRetentionSuite struct {
	RouterSuite
}

func TestToolOutputRetentionSuite(t *testing.T) {
	runSuite(t, new(ToolOutputRetentionSuite))
}

func (s *ToolOutputRetentionSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ToolOutputRetentionSuite) TestAnAppPolicyOfNoneRecordsNoneOfAToolsOutput() {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/app",
		map[string]any{"tool_output_retention": "none"}, nil))
	// The config would keep it a week. The app's policy is the longest any of its agents keeps.
	config := s.createConfig(map[string]any{"name": "orders", "llm": "tooling/tool-model",
		"tools": []any{map[string]any{"name": "orders", "output_retention": "7d", "tools": []any{lookupOrder}}}})

	result := s.ranTheTool(config.Id, map[string]any{"output": "it ships tomorrow"})

	s.Empty(value(result.Text))
	s.Equal(map[string]any{"call_id": result.callID, "redacted": true, "tokens": float64(12)}, value(result.Payload))
}

func (s *ToolOutputRetentionSuite) TestAGroupThatKeepsNoneRecordsNoneOfItsToolsOutput() {
	config := s.createConfig(map[string]any{"name": "orders", "llm": "tooling/tool-model",
		"tools": []any{map[string]any{"name": "orders", "output_retention": "none", "tools": []any{lookupOrder}}}})

	result := s.ranTheTool(config.Id, map[string]any{"output": "it ships tomorrow"})

	s.Empty(value(result.Text))
	s.Equal(map[string]any{"call_id": result.callID, "redacted": true, "tokens": float64(12)}, value(result.Payload))
}

func (s *ToolOutputRetentionSuite) TestASessionDeclaringAToolNoGroupNamesIsRefused() {
	config := s.createConfig(map[string]any{"name": "orders", "llm": "tooling/tool-model", "tools": []any{
		map[string]any{"name": "orders", "tools": []any{lookupOrder}},
	}})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", map[string]any{
		"config_id": config.Id,
		"tools":     []any{map[string]any{"name": "cancel_order", "description": "cancel an order"}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "no tool group of the agent's config names the tool: cancel_order")
}

func (s *ToolOutputRetentionSuite) TestWithoutARetentionAToolsOutputIsRecordedWhole() {
	config := s.createConfig(map[string]any{"name": "orders", "llm": "tooling/tool-model"})

	result := s.ranTheTool(config.Id, map[string]any{"output": "it ships tomorrow"})

	s.Equal("it ships tomorrow", value(result.Text))
	s.Equal(`{"order":"12"}`, value(result.Payload)["arguments"])
}

// The tool_ran frame of a failed tool says its error without the key it echoed; the item the
// client lists later says the same, and an error with no key in it as it was sent.
func (s *ToolOutputRetentionSuite) TestAToolsErrorIsListedWithoutTheKeyItEchoed() {
	config := s.createConfig(map[string]any{"name": "orders", "llm": "tooling/tool-model"})
	key := "sk-test-" + rand.Text()
	for failure, listed := range map[string]string{
		"orders: 401 Unauthorized\nAuthorization: Bearer " + key: "orders: 401 Unauthorized\n[redacted]",
		"orders: the order service timed out":                    "orders: the order service timed out",
	} {
		result := s.ranTheTool(config.Id, map[string]any{"error": failure})

		s.Equal(listed, value(result.Payload)["error"])
	}
}

// recordedResult is a tool result item as the items endpoint lists it, with the id of the
// call the model made.
type recordedResult struct {
	AgentResponseItem
	callID string
}

// ranTheTool opens a recorded conversation under the config, has the model ask for the
// caller's own tool and answers it with the fields of answer, and returns the tool result item
// it recorded.
func (s *ToolOutputRetentionSuite) ranTheTool(configID string, answer map[string]any) recordedResult {
	opened := s.serverClient.createSession(CreateSessionRequest{
		ConfigId: &configID,
		Tools: &[]SessionTool{{
			Name: lookupOrder, Description: "find an order by its number",
			Parameters: &map[string]any{"type": "object"},
		}},
	})
	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", SayRequest{Text: "where is my order"}, nil))
	asked := s.await(watching, "tool_call")
	answer["type"], answer["tool_call_id"] = "tool_result", asked["id"]
	s.Require().NoError(watching.WriteJSON(answer))
	s.await(watching, "tool_ran")

	var found *AgentResponseItem
	s.Require().Eventually(func() bool {
		var page AgentResponseItemPage
		if s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+opened.Id+"/responses/items", nil, &page) != http.StatusOK {
			return false
		}
		for _, item := range page.Items {
			if item.Kind == AgentResponseItemKind("tool_result") {
				found = &item
				return true
			}
		}
		return false
	}, settleFor, 50*time.Millisecond, "the tool result was never recorded")
	callID, _ := asked["id"].(string)
	return recordedResult{AgentResponseItem: *found, callID: callID}
}

// createConfig creates the config body describes, naming the caller's own tool in a group
// when body names none.
func (s *ToolOutputRetentionSuite) createConfig(body map[string]any) AgentConfig {
	if _, named := body["tools"]; !named {
		body["tools"] = []any{map[string]any{"name": "orders", "tools": []any{lookupOrder}}}
	}
	var created AgentConfig
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/configs", body, &created))
	return created
}
