//go:build integration

package api

import (
	"context"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// resumable opens a text session on a config of the recites model, which answers with what it
// was handed, asks it one question, and records one tool call behind that answer as a turn
// that used a tool would have. It returns the session's id.
func (s *SessionCreateSuite) resumable(resume store.ResumeHistory) string {
	ctx := context.Background()
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "resume-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "recites/recites-model", ResumeHistory: resume,
	}
	s.Require().NoError(s.store.CreateAgentConfig(ctx, &config))
	opened := s.serverClient.createSession(CreateSessionRequest{ConfigId: &config.ID})
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

func (s *SessionCreateSuite) TestAResumedSessionIsHandedTheToolResultsBehindItsReplies() {
	id := s.resumable(store.ResumeHistory{IncludeTools: true})

	recited := s.answerTo(id, "When?")

	s.Contains(recited, "tool: invoice 7: paid on 2026-09-01")
	s.Contains(recited, "user: Is invoice 7 paid?")
}

func (s *SessionCreateSuite) TestAResumedSessionIsHandedTextAloneByDefault() {
	id := s.resumable(store.ResumeHistory{})

	recited := s.answerTo(id, "When?")

	s.Contains(recited, "user: Is invoice 7 paid?")
	s.NotContains(recited, "invoice 7: paid on")
}

func (s *SessionCreateSuite) TestADeviceCannotChooseHowMuchHistoryIsResumed() {
	request := textSession(nil)
	request.ResumeHistory = &ResumeHistory{IncludeTools: pointerTo(true)}

	status, _ := s.client.failure(http.MethodPost, "/v1/agents/sessions", request)

	s.Equal(http.StatusForbidden, status)
}

func (s *SessionCreateSuite) TestASessionsResumeHistoryIsKeptOnItsRow() {
	request := textSession(nil)
	request.ResumeHistory = &ResumeHistory{IncludeTools: pointerTo(true), MaxMessages: pointerTo(40)}

	opened := s.serverClient.createSession(request)

	s.Eventually(func() bool {
		row, err := s.store.StoredSession(context.Background(), s.customerID(), opened.Id)
		return err == nil && row.ResumeHistory == store.ResumeHistory{IncludeTools: true, MaxMessages: 40}
	}, settleFor, 20*time.Millisecond)
}

func (s *SessionCreateSuite) TestAResumeHistoryCapOutsideItsRangeIsRefused() {
	for _, body := range []map[string]any{
		{"name": "too-many-" + s.utils.uuid(), "resume_history": map[string]any{"max_messages": 101}},
		{"name": "too-few-" + s.utils.uuid(), "resume_history": map[string]any{"max_characters": 999}},
	} {
		status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", body)
		s.Equal(http.StatusBadRequest, status)
	}
}

func (s *SessionCreateSuite) TestAConfigsResumeHistoryReadsBackAndPatchesFieldByField() {
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
