//go:build integration

package api

import (
	"context"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The approve (AI-1305) runs on ConnectionToolsSuite: a bearer connection to the fake, whose
// validate lists echo and fail.

func (s *ConnectionToolsSuite) TestOnlyTheAppsBackendMayApproveTools() {
	id := s.connected(bearer.Name)
	s.validate(id)
	approval := s.approval(id, "echo")

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/connections/"+id+"/tools/approve", approval, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/connections/"+id+"/tools/approve", approval, nil)
	})
}

// TestApprovingAChangedToolPinsItAndValidateAnswersConnected: the way out of needs_reapproval
// for a bearer connection, which no consent ever moves to a new grant.
func (s *ConnectionToolsSuite) TestApprovingAChangedToolPinsItAndValidateAnswersConnected() {
	id := s.connected(bearer.Name)
	s.pin(id, map[string]string{"echo": "pinned-before-it-changed"})
	s.Require().Equal(validationNeedsReapproval, string(s.validate(id).Status))
	listed := s.listedDigests(id)

	var approved ApprovedConnectionTools
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/tools/approve",
		s.approval(id, "echo"), &approved))

	s.Equal(ApprovedConnectionTools{ConnectionID: id, Tools: []ApprovedConnectionTool{
		{Name: "echo", SchemaDigest: listed["echo"], PreviousSchemaDigest: "pinned-before-it-changed"}}}, approved)
	s.Equal(map[string]string{"echo": listed["echo"]}, s.pins(id))
	validation := s.validate(id)
	s.Equal(validationConnected, string(validation.Status))
	s.Empty(validation.ChangedTools)
	var audit ConnectorAuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connector-audit?connection_id="+id, nil, &audit))
	var reapproved []ConnectorAuditEvent
	for _, event := range audit.Items {
		if event.Action == store.AuditToolReapproved {
			reapproved = append(reapproved, event)
		}
	}
	s.Require().Len(reapproved, 1)
	s.Equal(approved.Tools, reapproved[0].Tools)
	s.Nil(reapproved[0].Credential)
}

// TestADigestOtherThanTheListedOneIsAConflictAndNothingIsApproved: echo's digest is right and
// fail's is not, so neither moves.
func (s *ConnectionToolsSuite) TestADigestOtherThanTheListedOneIsAConflictAndNothingIsApproved() {
	id := s.connected(bearer.Name)
	before := map[string]string{"echo": "pinned-before-it-changed", "fail": "pinned-before-it-changed"}
	s.pin(id, before)
	s.validate(id)
	approval := s.approval(id, "echo", "fail")
	approval["tools"].([]map[string]string)[1]["schema_digest"] = "read-before-the-provider-changed-it-again"

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections/"+id+"/tools/approve", approval)

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "the connection lists fail at another schema_digest")
	s.NotContains(failure, "echo")
	s.Equal(before, s.pins(id))
	s.Equal(validationNeedsReapproval, string(s.validate(id).Status))
}

func (s *ConnectionToolsSuite) TestAToolTheConnectionDoesNotListIsRefusedAndNothingIsApproved() {
	id := s.connected(bearer.Name)
	before := map[string]string{"echo": "pinned-before-it-changed"}
	s.pin(id, before)
	s.validate(id)
	approval := s.approval(id, "echo")
	approval["tools"] = append(approval["tools"].([]map[string]string),
		map[string]string{"name": "delete_workspace", "schema_digest": "anything"})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections/"+id+"/tools/approve", approval)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `the connection lists no tool named "delete_workspace"`)
	s.Equal(before, s.pins(id))
}

func (s *ConnectionToolsSuite) TestAToolNamedTwiceIsRefused() {
	id := s.connected(bearer.Name)
	s.validate(id)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections/"+id+"/tools/approve",
		s.approval(id, "echo", "echo"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `tool "echo" is named twice`)
	s.Empty(s.pins(id))
}

// approval is an approve's body naming tools at the digests the connection's last validate
// listed.
func (s *ConnectionToolsSuite) approval(id string, tools ...string) map[string]any {
	listed := s.listedDigests(id)
	named := []map[string]string{}
	for _, tool := range tools {
		named = append(named, map[string]string{"name": tool, "schema_digest": listed[tool]})
	}
	return map[string]any{"tools": named}
}

// pins are the digests the connection's tools are pinned at for its current grant.
func (s *ConnectionToolsSuite) pins(id string) map[string]string {
	pins, err := s.store.ConnectorToolPins(context.Background(), id, s.connectedAt(id))
	s.Require().NoError(err)
	return pins
}

// TestAToolApprovedAgainIsOfferedToTheNextSession, end to end: Alice's bearer connection had
// echo pinned at a schema the fake no longer lists, so validate says needs_reapproval and her
// session leaves echo out. Her backend approves echo at what GET .../tools lists; validate
// answers connected, and her next session's model calls echo.
func (s *SessionConnectorsSuite) TestAToolApprovedAgainIsOfferedToTheNextSession() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "session", "", map[string]any{"name": "echo"}))
	connection, err := s.store.ConnectorConnection(context.Background(), s.customerID(), mine)
	s.Require().NoError(err)
	_, err = s.store.PinConnectorTools(context.Background(), mine, connection.ConnectedAt,
		map[string]string{"echo": "pinned-before-it-changed"})
	s.Require().NoError(err)
	backend := s.serverClient.actingFor(s.client)
	validate := func() ConnectionValidation {
		var validation ConnectionValidation
		s.Require().Equal(http.StatusOK, backend.do(http.MethodPost, "/v1/agents/connections/"+mine+"/validate", nil, &validation))
		return validation
	}
	s.Require().Equal(validationNeedsReapproval, string(validate().Status))
	before := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	left := s.await(s.client.opens("/v1/agents/sessions/"+before.Id+"/events"), "connector_unavailable")
	s.Require().Equal("tool_unavailable", left["reason"], "the changed echo is left out until it is approved")

	s.Require().Equal(http.StatusOK, backend.do(http.MethodPost, "/v1/agents/connections/"+mine+"/tools/approve",
		map[string]any{"tools": []map[string]any{echo}}, nil))

	s.Equal(validationConnected, string(validate().Status))
	after := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	ran := s.toolRanOn(backend, s.client.opens("/v1/agents/sessions/"+after.Id+"/events"), after.Id)
	s.Equal(connectorEcho, ran["tool"])
	s.Equal(connectorEchoText, ran["result"])
}
