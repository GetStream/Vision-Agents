package stream

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// ConnectorScopeRequiredSuite is how a connector_scope_required frame reads.
type ConnectorScopeRequiredSuite struct {
	suite.Suite
}

func TestConnectorScopeRequiredSuite(t *testing.T) {
	suite.Run(t, new(ConnectorScopeRequiredSuite))
}

func (s *ConnectorScopeRequiredSuite) TestAStepUpFrameReadsAsTheStepUp() {
	event := Event{Kind: "connector_scope_required", Frame: Frame{
		"type": "connector_scope_required", "name": "crm", "connector_id": "custom_crm", "connection_id": "conn-1",
		"scopes": []any{"files:read", "files:write"}, "authorization_id": "attempt-1",
		"launch_url":    "https://router.example/v1/agents/connectors/oauth/launch/attempt-1",
		"handoff_token": "handoff", "expires_at": "2026-10-07T12:00:00Z",
	}}

	asked, ok := event.ConnectorScopeRequired()

	s.True(ok)
	s.Equal(ConnectorScopeRequired{Name: "crm", ConnectorID: "custom_crm", ConnectionID: "conn-1",
		Scopes: []string{"files:read", "files:write"}, AuthorizationID: "attempt-1",
		LaunchURL: "https://router.example/v1/agents/connectors/oauth/launch/attempt-1", HandoffToken: "handoff",
		ExpiresAt: time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)}, asked)
}

func (s *ConnectorScopeRequiredSuite) TestAnotherKindIsNotAStepUp() {
	_, ok := Event{Kind: "tool_ran", Frame: Frame{"type": "tool_ran"}}.ConnectorScopeRequired()

	s.False(ok)
}
