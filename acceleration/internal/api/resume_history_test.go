//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ResumeHistorySuite runs on an app of each test's own, since a test sets the app's policy.
type ResumeHistorySuite struct {
	RouterSuite
}

func TestResumeHistorySuite(t *testing.T) {
	runSuite(t, new(ResumeHistorySuite))
}

func (s *ResumeHistorySuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// recalledTool is the tool result resumable records behind the first answer.
const recalledTool = "tool: invoice 7: paid on 2026-09-01"

// resumable opens a text session on a config of the recites model, which answers with what it
// was handed, asks it one question, and records one tool call behind that answer as a turn
// that used a tool would have. It returns the session's id once the session has stopped.
func (s *ResumeHistorySuite) resumable(config store.ResumeHistory, session *ResumeHistory) string {
	ctx := context.Background()
	agent := store.AgentConfig{
		CustomerID: s.customerID(), Name: "resume-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "recites/recites-model", ResumeHistory: config,
	}
	s.Require().NoError(s.store.CreateAgentConfig(ctx, &agent))
	opened := s.serverClient.createSession(CreateSessionRequest{ConfigId: &agent.ID, ResumeHistory: session})
	s.answerTo(opened.Id, "Is invoice 7 paid?")

	var answered store.AgentResponseItem
	s.Require().Eventually(func() bool {
		items, err := s.store.SessionItems(ctx, s.customerID(), opened.Id, "", 0, nil)
		for _, item := range items {
			if item.Kind == store.ItemAnswer {
				answered = item
			}
		}
		return err == nil && answered.ResponseID != ""
	}, settleFor, 20*time.Millisecond, "the answer was never recorded")
	s.Require().NoError(s.store.AppendResponseItems(ctx, []store.AgentResponseItem{{
		ResponseID: answered.ResponseID, Ordinal: answered.Ordinal + 1, SessionID: opened.Id,
		Kind: store.ItemToolResult, ToolName: "invoices", Text: "invoice 7: paid on 2026-09-01",
		Payload: map[string]any{"call_id": "call-7", "arguments": `{"id":7}`},
	}}))
	s.serverClient.stopSession(opened.Id)
	return opened.Id
}

func (s *ResumeHistorySuite) answerTo(id, question string) string {
	watching := s.serverClient.opens("/v1/agents/sessions/" + id + "/events")
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/responses", CreateResponseRequest{Text: question}, nil))
	answered, _ := s.await(watching, "responded")["text"].(string)
	return answered
}

func (s *ResumeHistorySuite) policy(scope string, resume map[string]any) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/"+scope,
		map[string]any{"resume_history": resume}, nil))
}

func (s *ResumeHistorySuite) TestAResumedSessionIsHandedTheToolResultsBehindItsReplies() {
	id := s.resumable(store.ResumeHistory{IncludeTools: pointerTo(true)}, nil)

	recited := s.answerTo(id, "When?")

	s.Contains(recited, recalledTool)
	s.Contains(recited, "user: Is invoice 7 paid?")
}

func (s *ResumeHistorySuite) TestAResumedSessionIsHandedTextAloneByDefault() {
	id := s.resumable(store.ResumeHistory{}, nil)

	recited := s.answerTo(id, "When?")

	s.Contains(recited, "user: Is invoice 7 paid?")
	s.NotContains(recited, "invoice 7: paid on")
}

func (s *ResumeHistorySuite) TestAnAppPolicyTurnsToolsOnForAConfigThatSaysNothing() {
	s.policy("app", map[string]any{"include_tools": true})
	id := s.resumable(store.ResumeHistory{}, nil)

	s.Contains(s.answerTo(id, "When?"), recalledTool)
}

func (s *ResumeHistorySuite) TestAnOrganizationPolicyTurnsToolsOnForItsApps() {
	s.policy("organization", map[string]any{"include_tools": true})
	id := s.resumable(store.ResumeHistory{}, nil)

	s.Contains(s.answerTo(id, "When?"), recalledTool)
}

func (s *ResumeHistorySuite) TestAnAppPolicyWinsOverItsOrganizations() {
	s.policy("organization", map[string]any{"include_tools": true})
	s.policy("app", map[string]any{"include_tools": false})
	id := s.resumable(store.ResumeHistory{}, nil)

	s.NotContains(s.answerTo(id, "When?"), "invoice 7: paid on")
}

func (s *ResumeHistorySuite) TestAConfigWinsOverItsAppsPolicy() {
	s.policy("app", map[string]any{"include_tools": true})
	id := s.resumable(store.ResumeHistory{IncludeTools: pointerTo(false)}, nil)

	s.NotContains(s.answerTo(id, "When?"), "invoice 7: paid on")
}

func (s *ResumeHistorySuite) TestASessionRequestWinsOverItsConfig() {
	id := s.resumable(store.ResumeHistory{IncludeTools: pointerTo(false)}, &ResumeHistory{IncludeTools: pointerTo(true)})

	s.Contains(s.answerTo(id, "When?"), recalledTool)
}

func (s *ResumeHistorySuite) TestAnAppPolicyReadsBackWhatWasSet() {
	s.policy("app", map[string]any{"include_tools": true, "max_messages": 40})

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))

	s.Equal(&ResumeHistory{IncludeTools: pointerTo(true), MaxMessages: pointerTo(40)}, read.ResumeHistory)
}

func (s *ResumeHistorySuite) TestADeviceCannotChooseHowMuchHistoryIsResumed() {
	request := textSession(nil)
	request.ResumeHistory = &ResumeHistory{IncludeTools: pointerTo(true)}

	status, _ := s.client.failure(http.MethodPost, "/v1/agents/sessions", request)

	s.Equal(http.StatusForbidden, status)
}

func (s *ResumeHistorySuite) TestASessionKeepsWhatItWasOpenedWithOnItsRow() {
	s.policy("app", map[string]any{"max_characters": 90000})
	request := textSession(nil)
	request.ResumeHistory = &ResumeHistory{IncludeTools: pointerTo(true), MaxMessages: pointerTo(40)}

	opened := s.serverClient.createSession(request)

	s.Eventually(func() bool {
		row, err := s.store.StoredSession(context.Background(), s.customerID(), opened.Id)
		return err == nil && row.ResumeHistory.Tools() && row.ResumeHistory.MaxMessages == 40 &&
			row.ResumeHistory.MaxCharacters == 90000
	}, settleFor, 20*time.Millisecond)
}

func (s *ResumeHistorySuite) TestAResumeHistoryCapOutsideItsRangeIsRefused() {
	for _, body := range []map[string]any{
		{"name": "too-many-" + s.utils.uuid(), "resume_history": map[string]any{"max_messages": 101}},
		{"name": "too-few-" + s.utils.uuid(), "resume_history": map[string]any{"max_characters": 999}},
	} {
		status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", body)
		s.Equal(http.StatusBadRequest, status)
	}
	status, _ := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"resume_history": map[string]any{"max_messages": 0}})
	s.Equal(http.StatusBadRequest, status)
}

func (s *ResumeHistorySuite) TestAConfigsResumeHistoryReadsBackAndPatchesFieldByField() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "resume-" + s.utils.uuid(), "resume_history": map[string]any{"include_tools": true, "max_characters": 120000},
	}, &created))
	s.Require().NotNil(created.ResumeHistory)

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id, map[string]any{
		"resume_history": map[string]any{"max_messages": 30},
	}, &patched))

	s.Equal(&ResumeHistory{IncludeTools: pointerTo(true), MaxMessages: pointerTo(30), MaxCharacters: pointerTo(120000)}, patched.ResumeHistory)
}
