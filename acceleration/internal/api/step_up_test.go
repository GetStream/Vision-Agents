//go:build integration

package api

import (
	"context"
	"net/http"
	"net/url"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The step-up tests run in ChatLoginsSuite, whose router has oauth2_code against the fake and
// a model that calls crm's echo whenever a person asks: the fake refuses that call with a 403
// insufficient_scope until the token carries fakeprovider.RequiredScope (AI-854).

// TestAScopeChallengeAsksForOneStepUpAndKeepsTheGrant: a 403 insufficient_scope during a
// session produces one connector_scope_required event, asked again it produces none, and the
// connection keeps its grant while the step-up is open.
func (s *ChatLoginsSuite) TestAScopeChallengeAsksForOneStepUpAndKeepsTheGrant() {
	mine, events, opened := s.steppingUp()
	before := s.connectionOf(s.client, mine)

	s.ask(opened)
	asked := s.stepUpOn(events)
	s.ask(opened)
	again := s.ranWithoutStepUp(events)

	s.Equal("crm", asked["name"])
	s.Equal(mine, asked["connection_id"])
	s.Equal([]any{"files:read", fakeprovider.RequiredScope}, asked["scopes"], "the fake's challenge: granted plus missing")
	s.Equal(consentPublicURL+connectorLaunchPath+asked["authorization_id"].(string), asked["launch_url"])
	s.Regexp(handoffToken, asked["handoff_token"])
	s.Equal(store.AttemptStepUp, s.attemptKind(asked["authorization_id"].(string)))
	s.Contains(again, `"status":"scope_required"`, "the second call reads the same step-up")
	s.Equal(2, s.attemptsOn(mine), "the first consent and one step-up")
	after := s.connectionOf(s.client, mine)
	s.Equal(ConnectionStatus(store.ConnectionConnected), after.Status, "the old grant keeps working")
	s.Equal(before.Revision, after.Revision)
	s.Equal(before.GrantedScopes, after.GrantedScopes)
}

// TestAFinishedStepUpLetsTheSameCallRunInTheSameSession: the person consents to the scope the
// provider asked for, and the same session's next call of the same tool runs.
func (s *ChatLoginsSuite) TestAFinishedStepUpLetsTheSameCallRunInTheSameSession() {
	mine, events, opened := s.steppingUp()
	s.ask(opened)
	asked := s.stepUpOn(events)

	b := newBrowser(&s.RouterSuite, s.provider)
	authorize := b.handOff(s.started(asked))
	finished := b.finish(s.consent(authorize))
	s.ask(opened)
	ran := s.echoRanOn(events)

	parsed, err := url.Parse(authorize)
	s.Require().NoError(err)
	s.Equal("files:read "+fakeprovider.RequiredScope, parsed.Query().Get("scope"))
	s.Equal(s.landing(mine), finished.Header.Get("Location"))
	s.Equal(connectorEchoText, ran["result"], "the same call runs in the same session")
	s.Empty(ran["error"])
	s.Equal([]string{"files:read", fakeprovider.RequiredScope}, s.connectionOf(s.client, mine).GrantedScopes)
}

// TestADeniedStepUpLeavesTheOldGrantAsItWas: the person declines at the provider. Nothing is
// written, and the next refused call begins a new step-up rather than show the used one.
func (s *ChatLoginsSuite) TestADeniedStepUpLeavesTheOldGrantAsItWas() {
	mine, events, opened := s.steppingUp()
	before := s.connectionOf(s.client, mine)
	s.ask(opened)
	asked := s.stepUpOn(events)

	s.provider.Use(fakeprovider.ClientCredentials, fakeprovider.InsufficientScope, fakeprovider.ConsentDenied)
	b := newBrowser(&s.RouterSuite, s.provider)
	finished := b.finish(s.consent(b.handOff(s.started(asked))))
	s.provider.Use(fakeprovider.ClientCredentials, fakeprovider.InsufficientScope)
	s.ask(opened)
	again := s.stepUpOn(events)

	s.Equal(consentDashboard+"?"+url.Values{"connection_id": {mine}, "status": {consentDenied}}.Encode(), finished.Header.Get("Location"))
	after := s.connectionOf(s.client, mine)
	s.Equal(ConnectionStatus(store.ConnectionConnected), after.Status)
	s.Equal(before.Revision, after.Revision, "no credential was written")
	s.Equal(before.GrantedScopes, after.GrantedScopes)
	s.NotEqual(asked["authorization_id"], again["authorization_id"])
}

// TestAClaimsStepUpSendsTheChallengeBack: Microsoft's claims challenge on a 401 keeps the old
// grant, and a step-up that carries the claims back to the authorize request gets a token the
// provider takes.
func (s *ChatLoginsSuite) TestAClaimsStepUpSendsTheChallengeBack() {
	connector, _ := s.connectorWith(`
scopes:
  list: [files:read]
`)
	mine := s.connected(s.client, connector)
	s.provider.Use(fakeprovider.ClientCredentials, fakeprovider.ClaimsChallenge)
	refused := s.validate(mine)
	consents := ConnectorConsents(s.store, s.connectors, s.sealer, s.publicURL)

	consent, err := consents(context.Background(), session.ConsentRequest{
		CustomerID: s.customerID(), ConnectorID: connector, UserID: s.client.userID, ConnectionID: mine,
		StepUp: &core.Outcome{Kind: core.OutcomeScopeRequired, Claims: fakeprovider.ClaimsChallengeJSON},
	})
	s.Require().NoError(err)
	b := newBrowser(&s.RouterSuite, s.provider)
	authorize := b.handOff(Authorization{ID: consent.AuthorizationID, LaunchURL: consent.LaunchURL, HandoffToken: consent.HandoffToken})
	finished := b.finish(s.consent(authorize))

	s.NotEqual(ConnectionValidationStatus(validationConnected), refused.Status)
	s.Equal(store.AttemptStepUp, s.attemptKind(consent.AuthorizationID))
	parsed, err := url.Parse(authorize)
	s.Require().NoError(err)
	s.JSONEq(fakeprovider.ClaimsChallengeJSON, parsed.Query().Get("claims"))
	s.Equal("files:read", parsed.Query().Get("scope"), "no scope asked, so the manifest's own")
	s.Equal(s.landing(mine), finished.Header.Get("Location"))
	s.Equal(ConnectionValidationStatus(validationConnected), s.validate(mine).Status, "the new token passes the challenge")
}

// steppingUp is a connection of the caller's to a connector asking for files:read, a config
// binding it as crm with echo granted, and a session on that config choosing it, with its
// events socket. The fake refuses echo for want of fakeprovider.RequiredScope.
func (s *ChatLoginsSuite) steppingUp() (string, *websocket.Conn, string) {
	connector, grant := s.connectorWith(`
scopes:
  list: [files:read]
`)
	mine := s.connected(s.client, connector)
	s.provider.Use(fakeprovider.ClientCredentials, fakeprovider.InsufficientScope)
	opened := s.client.createSession(s.session(s.config(connector, grant), map[string]string{"crm": mine}))
	return mine, s.client.opens("/v1/agents/sessions/" + opened.Id + "/events"), opened.Id
}

// stepUpOn is the connector_scope_required frame events carries, read through the tool_ran of
// the echo that asked for it. What the model read of that call is the step-up's status and
// names neither the launch URL nor the handoff token.
func (s *ChatLoginsSuite) stepUpOn(events *websocket.Conn) map[string]any {
	var found, ran map[string]any
	s.Require().NoError(events.SetReadDeadline(time.Now().Add(settleFor)))
	for found == nil || ran == nil {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		switch frame["type"] {
		case "connector_scope_required":
			s.Nil(found, "one event")
			found = frame
		case "tool_ran":
			if frame["tool"] == connectorEcho {
				ran = frame
			}
		}
	}
	read, _ := ran["result"].(string)
	s.Contains(read, `"status":"scope_required"`)
	s.NotContains(read, found["launch_url"])
	s.NotContains(read, found["handoff_token"])
	s.settle(events)
	return found
}

// settle reads events until the reply is done, so the next command is not refused for one
// still running.
func (s *ChatLoginsSuite) settle(events *websocket.Conn) {
	for {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		s.NotEqual("connector_scope_required", frame["type"], "no second event")
		if frame["type"] == "responded" && frame["pending_work"] == false {
			return
		}
	}
}

// ranWithoutStepUp is what the model read of the next echo on events, which sent no
// connector_scope_required before it.
func (s *ChatLoginsSuite) ranWithoutStepUp(events *websocket.Conn) string {
	ran := s.echoRanOn(events)
	read, _ := ran["result"].(string)
	return read
}

// echoRanOn is the next tool_ran of crm's echo on events. A connector_scope_required before it
// fails the test.
func (s *ChatLoginsSuite) echoRanOn(events *websocket.Conn) map[string]any {
	s.Require().NoError(events.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		s.NotEqual("connector_scope_required", frame["type"], "no second event")
		if frame["type"] == "tool_ran" && frame["tool"] == connectorEcho {
			s.settle(events)
			return frame
		}
	}
}

// attemptsOn is how many consents were begun on connection.
func (s *ChatLoginsSuite) attemptsOn(connection string) int {
	var attempts int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM connector_authorization_attempts WHERE connection_id = ?", connection).Scan(&attempts))
	return attempts
}

// validate lists the connection's tools at the provider, as its owner's backend asks.
func (s *ChatLoginsSuite) validate(connection string) ConnectionValidation {
	var validation ConnectionValidation
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/connections/"+connection+"/validate", nil, &validation))
	return validation
}
