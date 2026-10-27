//go:build integration

package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The suite's own built-ins are delivered to by no other suite, and its tests run one at a
// time, so a count of one of them moves only with the test that delivers.

func (s *ConnectorEventsSuite) TestAMessageAnAgentAnswersIsCountedHandled() {
	s.countedOnce("events_test_bot", eventHandled, http.StatusOK, s.signed(s.messageEvent(""), time.Now()))
}

func (s *ConnectorEventsSuite) TestASignalIsCountedHandled() {
	s.countedOnce("events_test_bot", eventHandled, http.StatusOK, s.signed(uninstall(s.team()), time.Now()))
}

// The manifest skips a bot's own message, so the delivery carries nothing the router acts on.
func (s *ConnectorEventsSuite) TestASkippedMessageIsCountedUnhandled() {
	s.countedOnce("events_test_bot", eventUnhandled, http.StatusOK, s.signed(s.messageEvent("B0000BOT"), time.Now()))
}

func (s *ConnectorEventsSuite) TestAnUnsignedEventIsCountedRefused() {
	s.countedOnce("events_test_bot", eventRefused, http.StatusUnauthorized,
		s.signedWith(s.messageEvent(""), time.Now(), "another-secret"))
}

// events_test_unverifiable's verifier, ed25519, is not registered in this suite's router.
func (s *ConnectorEventsSuite) TestAnEventNothingHereVerifiesIsCountedRefused() {
	s.countedOnce("events_test_unverifiable", eventRefused, http.StatusNotFound, s.signed(s.messageEvent(""), time.Now()))
}

// No EVENTS_TEST_SECRETLESS_MCP_SIGNING_SECRET is in the suite's environment.
func (s *ConnectorEventsSuite) TestAnEventOfABuiltInWithoutASecretIsCountedRefused() {
	s.countedOnce("events_test_secretless", eventRefused, http.StatusNotFound, s.signed(s.messageEvent(""), time.Now()))
}

// Linear's manifest has no channel block. No other suite delivers to Linear's events route.
func (s *ConnectorEventsSuite) TestAnEventOfABuiltInWithoutAChannelIsCountedRefused() {
	s.countedOnce("linear", eventRefused, http.StatusNotFound, s.signed(uninstall(s.team()), time.Now()))
}

func (s *ConnectorEventsSuite) TestAnEventTheBridgeFailsOnIsCountedFailed() {
	s.delivered.failing(s.T(), errors.New("synthetic bridge failure"))
	s.countedOnce("events_test_bot", eventFailed, http.StatusInternalServerError, s.signed(s.messageEvent(""), time.Now()))
}

func (s *ConnectorEventsSuite) TestAHandshakeIsCountedAsAChallenge() {
	s.countedOnce("events_test_bot", eventChallenge, http.StatusOK,
		s.signed([]byte(`{"challenge":"synthetic-challenge-value","type":"url_verification"}`), time.Now()))
}

// An id under no definition is the sender's choice, so it is not counted at all.
func (s *ConnectorEventsSuite) TestAnEventOfAConnectorNobodyDefinedIsNotCounted() {
	id := "no_such_connector_" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	s.countedOnce(id, "", http.StatusNotFound, s.signed(uninstall(s.team()), time.Now()))
}

// A customer's own connector is counted as custom, never under the id the customer chose.
// Other suites may count custom events at the same time, so the count only has to rise.
func (s *ConnectorEventsSuite) TestACustomConnectorsEventIsCountedAsCustom() {
	connector := store.CustomPrefix + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte("id: " + connector + "\nrevision: 1\nname: Metrics probe\n" +
		"endpoints:\n  mcp: https://crm.example/mcp\nschemes: [oauth2_code]\nclient:\n  registration: [customer]\n"))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	appID, secret := "synthetic-custom-"+s.utils.uuid(), "synthetic-custom-signing-"+s.utils.uuid()
	record := &store.ConnectorOAuthClient{CustomerID: s.customerID(), ConnectorID: connector,
		Registration: core.ClientCustomer, ClientID: "synthetic-client", ProviderAppID: appID}
	record.SigningSecretSealed, err = s.sealer.SealWithAAD(secret, providerAppAAD(record.CustomerID, connector, appID))
	s.Require().NoError(err)
	record.SigningKEKVersion = s.sealer.CurrentVersion()
	_, err = s.store.PutConnectorOAuthClient(context.Background(), record)
	s.Require().NoError(err)
	before := eventsOf("custom", eventRefused)

	response, err := http.Post(s.server.URL+providerAppEventsPath+connector+"/"+appID, "application/json",
		strings.NewReader(`{"synthetic":"custom"}`))
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Equal(http.StatusNotFound, response.StatusCode, "its manifest has no channel")
	s.Greater(eventsOf("custom", eventRefused), before)
	s.Zero(eventsOf(connector, eventRefused))
}

// countedOnce delivers request to connector's events route, requires status, and that the
// connector's events counted outcome once more and no other outcome ("" for none at all).
func (s *ConnectorEventsSuite) countedOnce(connector, outcome string, status int, request *http.Request) {
	outcomes := []string{eventHandled, eventUnhandled, eventRefused, eventFailed, eventChallenge}
	before := map[string]float64{}
	for _, each := range outcomes {
		before[each] = eventsOf(connector, each)
	}

	answered, _, _ := s.deliver(connector, request)

	s.Equal(status, answered)
	for _, each := range outcomes {
		want := before[each]
		if each == outcome {
			want++
		}
		s.Equal(want, eventsOf(connector, each), each)
	}
}

// messageEvent is a Slack message in a team of the test's own, from bot when it is not "".
func (s *ConnectorEventsSuite) messageEvent(bot string) []byte {
	from := `"user":"U0000USER"`
	if bot != "" {
		from = `"bot_id":"` + bot + `"`
	}
	return []byte(`{"team_id":"` + s.team() + `","type":"event_callback","event":{"type":"message","channel":"C0000CHAN",` +
		from + `,"text":"hello","ts":"1759740000.000300"}}`)
}

// eventsBotAs is eventsBot as another built-in, id, whose operator secret is env's.
func eventsBotAs(id, env string) string {
	return strings.NewReplacer("id: events_test_bot", "id: "+id, "env: EVENTS_TEST_BOT", "env: "+env).Replace(eventsBot)
}

// eventsOf is how many events of connector were counted as outcome.
func eventsOf(connector, outcome string) float64 {
	return testutil.ToFloat64(connectorEvents.WithLabelValues(connector, outcome))
}
