//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"testing"
)

// SessionCapSuite is an app held to one session or model stream open at once.
type SessionCapSuite struct {
	RouterSuite
}

func TestSessionCapSuite(t *testing.T) {
	runSuite(t, new(SessionCapSuite))
}

func (s *SessionCapSuite) SetupSuite() {
	// One, so a test holds the whole of an app's allowance with a single session.
	s.sessionsPerApp = 1
	s.RouterSuite.SetupSuite()
}

func (s *SessionCapSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionCapSuite) TestASessionPastTheAppsSessionsOpenIsToldWhenToRetry() {
	var opened Session
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions",
		map[string]any{"text": true, "llm": "en-low-latency"}, &opened))

	response := s.serverClient.raw(http.MethodPost, "/v1/agents/sessions", map[string]any{"text": true, "llm": "en-low-latency"})

	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Equal("5", response.Header.Get("Retry-After"))
	var answered ErrorResponse
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answered))
	s.Equal(codeConcurrencyLimited, answered.Error.Code)
}

func (s *SessionCapSuite) TestAModelStreamCountsAgainstTheAppsSessionsOpen() {
	var opened Session
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions",
		map[string]any{"text": true, "llm": "en-low-latency"}, &opened))

	status, message := s.serverClient.refusal("/v1/stt/stream")

	s.Equal(http.StatusTooManyRequests, status)
	s.Contains(message, "as many sessions open as it may")
}

func (s *SessionCapSuite) TestAStoppedSessionFreesItsAppsPlace() {
	var opened Session
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions",
		map[string]any{"text": true, "llm": "en-low-latency"}, &opened))
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/stop", nil, nil))

	s.Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions",
		map[string]any{"text": true, "llm": "en-low-latency"}, &opened))
}
