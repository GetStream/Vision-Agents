//go:build integration

package api

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
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

// The provider app's route carries Slack's, WhatsApp's, Telnyx's and Linq's events, and a
// customer's own: an event on it is counted under the connector the route names.
func (s *SlackChannelSuite) TestAMessageOnTheProviderAppRouteIsCountedHandledUnderItsConnector() {
	before := eventsOf("slack_bot", eventHandled)

	s.messaged("U0000ALICE", "synthetic counted text", "1759740000.009100", "")

	s.Equal(before+1, eventsOf("slack_bot", eventHandled))
}

// The ids in the path of a provider app's route are the sender's choice: a request naming a
// provider app nobody holds, or one whose connector has no definition for its customer, leaves
// no series under the id it names. The second is a race the store's cascade makes rare: the
// definition went after the client was read.
func (s *SlackChannelSuite) TestAnEventOnTheProviderAppRouteOfAnIdNobodyDefinedIsNotCounted() {
	unknown := "no_such_connector_" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	status, _ := s.slack.Deliver(s.server.URL+providerAppEventsPath+unknown+"/A"+s.utils.uuid(), s.secret,
		[]byte(`{"synthetic":"unknown"}`), 0)
	s.Equal(http.StatusNotFound, status)
	s.Zero(eventSeries(unknown), "a provider app nobody holds")

	connector := store.CustomPrefix + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte("id: " + connector + "\nrevision: 1\nname: Metrics probe\n" +
		"endpoints:\n  mcp: https://crm.example/mcp\nschemes: [oauth2_code]\nclient:\n  registration: [customer]\n"))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	appID, secret := "synthetic-gone-"+s.utils.uuid(), "synthetic-gone-signing-"+s.utils.uuid()
	record := &store.ConnectorOAuthClient{CustomerID: s.customerID(), ConnectorID: connector,
		Registration: core.ClientCustomer, ClientID: "synthetic-client", ProviderAppID: appID}
	record.SigningSecretSealed, err = s.sealer.SealWithAAD(secret, providerAppAAD(record.CustomerID, connector, appID))
	s.Require().NoError(err)
	record.SigningKEKVersion = s.sealer.CurrentVersion()
	_, err = s.store.PutConnectorOAuthClient(context.Background(), record)
	s.Require().NoError(err)
	_, err = s.store.DB().NewRaw(`DELETE FROM connector_definitions WHERE customer_id = ? AND id = ?`,
		s.customerID(), connector).Exec(context.Background())
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorOAuthClient(context.Background(), s.customerID(), connector, core.ClientCustomer))
	})
	custom := eventsOf("custom", eventRefused)

	status, _ = s.slack.Deliver(s.server.URL+providerAppEventsPath+connector+"/"+appID, secret, []byte(`{"synthetic":"gone"}`), 0)

	s.Equal(http.StatusNotFound, status)
	s.Equal(custom, eventsOf("custom", eventRefused), "an event no definition reads is not counted")
	s.Zero(eventSeries(connector))
}

// An event the forwarder fails to queue is answered 500 so the provider delivers it again, and
// counted failed, not as the unhandled event it would be.
func (s *EventForwardingSuite) TestAnEventTheForwarderFailsOnIsCountedFailed() {
	closed, err := store.OpenShards(store.Options{DSN: testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN"))})
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	broken, err := eventforward.New(eventforward.Options{Store: closed, Secrets: s.sealer, HTTP: s.forwardHTTP, Logger: s.router.logger})
	s.Require().NoError(err)
	working := s.router.eventForwarder
	s.router.eventForwarder = broken
	s.T().Cleanup(func() { s.router.eventForwarder = working })
	failed, unhandled := eventsOf("slack_bot", eventFailed), eventsOf("slack_bot", eventUnhandled)

	status, _ := s.deliver(s.reaction(), 0)

	s.Equal(http.StatusInternalServerError, status)
	s.Equal(failed+1, eventsOf("slack_bot", eventFailed))
	s.Equal(unhandled, eventsOf("slack_bot", eventUnhandled))
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

// eventSeries is how many series of connectorEvents are under connector, whatever the outcome.
func eventSeries(connector string) int {
	registry := prometheus.NewPedanticRegistry()
	registry.MustRegister(connectorEvents)
	families, err := registry.Gather()
	if err != nil {
		panic(err)
	}
	found := 0
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() == "connector" && label.GetValue() == connector {
					found++
				}
			}
		}
	}
	return found
}
