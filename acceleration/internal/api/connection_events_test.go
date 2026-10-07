//go:build integration

package api

import (
	"context"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcpevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// issueCreated is the event the suite's bindings declare: the draft's own kind of example
// («incident.created»), named for an issue tracker.
const issueCreated = "issue.created"

// ConnectionEventsSuite is MCP Events over connections end to end (T60, AI-899): a connector
// at the fake provider, which offers events (fakeprovider.MCPEvents), an app connection holding
// the fake's token, validated through the API, and an agent config whose fixed binding declares
// an event. The validate subscribes; the fake checks the callback, then delivers to it.
type ConnectionEventsSuite struct {
	RouterSuite
	provider *fakeprovider.Server
	// token is an access token the fake issued. Synthetic, fresh per suite.
	token string
}

func TestConnectionEventsSuite(t *testing.T) {
	runSuite(t, new(ConnectionEventsSuite))
}

func (s *ConnectionEventsSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.MCPEvents)
	s.connectors = core.Registry{
		Schemes:     map[string]core.Scheme{bearer.Name: bearer.New()},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	s.connectorHTTP = s.provider.Client()
	s.mcpEventsOn = true
	s.RouterSuite.SetupSuite()
	s.token = issuedToken(&s.RouterSuite, s.provider)
}

func (s *ConnectionEventsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// TestAnEventFromAConnectedServerReachesTheAgent: the whole path. The event's data is in what
// the agent's model was asked.
func (s *ConnectionEventsSuite) TestAnEventFromAConnectedServerReachesTheAgent() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	marker := "TypeError in checkout " + s.utils.uuid()

	answered := s.provider.Emit(issueCreated, map[string]any{"title": marker})

	s.Equal([]int{http.StatusAccepted}, s.statuses(answered, connection))
	s.Eventually(func() bool { return s.modelWasAsked(marker) }, settleFor, 50*time.Millisecond)
}

// TestTheServerCheckedTheCallbackBeforeSubscribing: the fake subscribes only after the router
// echoed its signed challenge, and the subscription is active with the fake's grant.
func (s *ConnectionEventsSuite) TestTheServerCheckedTheCallbackBeforeSubscribing() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))

	held := s.held(connection)
	s.Require().Len(held, 1)
	s.Equal(store.ConnectionEventActive, held[0].Status)
	s.NotEmpty(held[0].RemoteID)
	s.Require().NotNil(held[0].RefreshBefore)
	s.Len(s.atTheFake(connection), 1)
}

// TestADeliverySignedWithAnotherSubscriptionsSecretIsRefused: two subscriptions of one
// connection, each with its own secret. A delivery to one signed with the other's opens nothing.
func (s *ConnectionEventsSuite) TestADeliverySignedWithAnotherSubscriptionsSecretIsRefused() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	binding["events"] = append(binding["events"].([]map[string]any), map[string]any{"event": "issue.closed"})
	s.subscribed(binding)
	subs := s.atTheFake(connection)
	s.Require().Len(subs, 2)
	marker := s.utils.uuid()

	status := s.provider.DeliverEvent(subs[0].URL, subs[1].Secret, subs[0].ID, "evt_"+s.utils.uuid(), subs[0].Name,
		map[string]any{"title": marker})

	s.Equal(http.StatusUnauthorized, status)
	s.NotEqual(subs[0].Secret, subs[1].Secret)
	s.Never(func() bool { return s.modelWasAsked(marker) }, time.Second, 50*time.Millisecond)
}

func (s *ConnectionEventsSuite) TestARetriedDeliveryOpensNoSecondConversation() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	sub := s.atTheFake(connection)[0]
	id := "evt_" + s.utils.uuid()

	first := s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, id, issueCreated, map[string]any{})
	again := s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, id, issueCreated, map[string]any{})

	s.Equal(http.StatusAccepted, first)
	s.Equal(http.StatusOK, again)
}

// TestADeletedConnectionStopsItsSubscription: the delete drops the rows at once, and the
// server's next delivery is told 410, which the draft says not to retry.
func (s *ConnectionEventsSuite) TestADeletedConnectionStopsItsSubscription() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	sub := s.atTheFake(connection)[0]

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+connection+"?force=true", nil, nil))

	s.Empty(s.held(connection))
	s.Equal(http.StatusGone, s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{}))
}

// TestADisconnectedConnectionStopsItsSubscription: a connection whose status is
// disconnected, as a delete writes it, ends the subscription: the delivery is a 410 and the row
// goes.
func (s *ConnectionEventsSuite) TestADisconnectedConnectionStopsItsSubscription() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	sub := s.atTheFake(connection)[0]
	s.setStatus(connection, store.ConnectionDisconnected)
	marker := s.utils.uuid()

	status := s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{"title": marker})

	s.Equal(http.StatusGone, status)
	s.Empty(s.held(connection))
	s.Never(func() bool { return s.modelWasAsked(marker) }, time.Second, 50*time.Millisecond)
}

// TestARenewalInFlightKeepsTheSubscription: the resolver writes needs_reauthorization at its
// checkpoint before every OAuth refresh and connected after it. A delivery in between is a
// 503, which the server sends again, and opens nothing; the subscription stays, so the next
// delivery after the refresh is taken.
func (s *ConnectionEventsSuite) TestARenewalInFlightKeepsTheSubscription() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	sub := s.atTheFake(connection)[0]
	s.setStatus(connection, store.ConnectionNeedsReauthorization)
	during, after := s.utils.uuid(), s.utils.uuid()

	waiting := s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{"title": during})
	s.setStatus(connection, store.ConnectionConnected)
	taken := s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{"title": after})

	s.Equal(http.StatusServiceUnavailable, waiting)
	s.Equal(http.StatusAccepted, taken)
	s.Len(s.held(connection), 1)
	s.Eventually(func() bool { return s.modelWasAsked(after) }, settleFor, 50*time.Millisecond)
	s.False(s.modelWasAsked(during))
}

// TestAStaleCopyOfARenewalKeepsTheSubscriptions: a validate reads the connection without the
// credential lock, so its copy can be needs_reauthorization from another router's refresh in
// flight while the stored row is connected again. Reconcile with that copy changes nothing.
func (s *ConnectionEventsSuite) TestAStaleCopyOfARenewalKeepsTheSubscriptions() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	stale, err := s.store.ConnectorConnection(context.Background(), s.customerID(), connection)
	s.Require().NoError(err)
	stale.Status = store.ConnectionNeedsReauthorization

	s.Require().NoError(s.mcpEvents.Reconcile(context.Background(), stale))

	held := s.held(connection)
	s.Require().Len(held, 1)
	s.Equal(store.ConnectionEventActive, held[0].Status)
}

// TestAWorkerThatMeetsARenewalAsksAgainBeforeTheGrantEnds: the worker's look comes while the
// connection is needs_reauthorization, 30 s before the server's grant ends. It keeps the row
// and looks again before refresh_before, not 15 minutes later, after the server stopped
// delivering. An idle worker with a 200 ms lease does the look.
func (s *ConnectionEventsSuite) TestAWorkerThatMeetsARenewalAsksAgainBeforeTheGrantEnds() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	s.setStatus(connection, store.ConnectionNeedsReauthorization)
	looked := time.Now().UTC()
	refreshBefore := looked.Add(30 * time.Second)
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE connection_event_subscriptions SET refresh_before = ?, next_attempt_at = ? WHERE connection_id = ?",
		refreshBefore, looked, connection)
	s.Require().NoError(err)

	s.idleWorker(idleLease)

	// Past either worker's claim lease (200 ms here, 1 s for the suite's own), so only the wait
	// a look saved satisfies it, whichever worker looked: both save the same.
	s.Eventually(func() bool {
		held := s.held(connection)
		return len(held) == 1 && held[0].NextAttemptAt != nil &&
			held[0].NextAttemptAt.After(looked.Add(2*time.Second)) && held[0].NextAttemptAt.Before(refreshBefore)
	}, settleFor, 10*time.Millisecond)
	s.Equal(store.ConnectionEventActive, s.held(connection)[0].Status)
}

// TestAGrantAlreadyEndedIsNotAskedForInALoop: the server answers a refreshBefore already past
// (a clock behind, or a bug). The router asks again once a lease (1 s here), not at once in a
// loop: at most a handful of events/subscribe in a second, not dozens.
func (s *ConnectionEventsSuite) TestAGrantAlreadyEndedIsNotAskedForInALoop() {
	s.provider.GrantEventsFor(-time.Minute)
	s.T().Cleanup(func() { s.provider.GrantEventsFor(0) })

	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	time.Sleep(time.Second)

	subs := s.atTheFake(connection)
	s.Require().Len(subs, 1)
	s.LessOrEqual(subs[0].Subscribes, 3)
}

// TestARevokedGrantPausesItsSubscriptionUntilTheConnectionIsConnectedAgain: the provider ended
// the grant (a revoke signal, as Slack's tokens_revoked). Nothing reaches the agent while the
// connection waits for a reconnect, and new credentials bring the subscription back with no
// validate.
func (s *ConnectionEventsSuite) TestARevokedGrantPausesItsSubscriptionUntilTheConnectionIsConnectedAgain() {
	connection := s.subscribed(s.binding(s.connection(), issueCreated))
	sub := s.atTheFake(connection)[0]
	s.Require().NoError(s.resolver.Revoke(context.Background(),
		core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: connection}, core.SignalRevoked, time.Time{}))
	revoked, reconnected := s.utils.uuid(), s.utils.uuid()

	paused := s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{"title": revoked})
	var read Connection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+connection, nil, &read))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+connection+"/credentials",
		map[string]any{"expected_revision": read.Revision, "values": map[string]string{bearer.SuppliedToken: s.token}}, nil))
	resumed := s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{"title": reconnected})

	s.Equal(http.StatusServiceUnavailable, paused)
	s.Equal(http.StatusAccepted, resumed)
	s.Eventually(func() bool { return s.modelWasAsked(reconnected) }, settleFor, 50*time.Millisecond)
	s.False(s.modelWasAsked(revoked))
}

// TestAnEventNoLongerDeclaredIsUnsubscribedAndGone: the config drops the event, and the next
// validate has the worker tell the server to stop and drop the row.
func (s *ConnectionEventsSuite) TestAnEventNoLongerDeclaredIsUnsubscribedAndGone() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	config := s.config(binding)
	s.validate(connection)
	s.Require().Eventually(func() bool { return len(s.atTheFake(connection)) == 1 }, settleFor, 20*time.Millisecond)
	sub := s.atTheFake(connection)[0]
	delete(binding, "events")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+config,
		map[string]any{"connectors": []map[string]any{binding}}, nil))

	s.validate(connection)

	s.Eventually(func() bool { return len(s.atTheFake(connection)) == 0 && len(s.held(connection)) == 0 }, settleFor, 20*time.Millisecond)
	s.Equal(http.StatusGone, s.provider.DeliverEvent(sub.URL, sub.Secret, sub.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{}))
}

// TestABindingWithNoEventsSubscribesToNothing: a validate of a connection bound without
// events adds no row and asks the server nothing about events, as before T60.
func (s *ConnectionEventsSuite) TestABindingWithNoEventsSubscribesToNothing() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	delete(binding, "events")
	s.config(binding)

	s.validate(connection)

	s.Never(func() bool { return len(s.held(connection)) > 0 || len(s.atTheFake(connection)) > 0 }, time.Second, 50*time.Millisecond)
}

func (s *ConnectionEventsSuite) TestASessionBindingCannotDeclareEvents() {
	connector := s.connector()
	binding := map[string]any{"name": "crm", "connector_id": connector, "connection": map[string]any{"type": "session"},
		"tools": []map[string]any{}, "events": []map[string]any{{"event": issueCreated}}}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "watcher-" + s.utils.uuid(), "connectors": []map[string]any{binding},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "crm" declares events, and only a fixed binding may`)
}

func (s *ConnectionEventsSuite) TestABindingsEventsAreReadBackAsWritten() {
	binding := s.binding(s.connection(), issueCreated)
	config := s.config(binding)

	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+config, nil, &read))

	s.Require().NotNil(read.Connectors)
	s.Require().Len(*read.Connectors, 1)
	events := value((*read.Connectors)[0].Events)
	s.Require().Len(events, 1)
	s.Equal(issueCreated, events[0].Event)
	s.Equal(map[string]any{"project": "web"}, value(events[0].Arguments))
	s.Equal("Say what broke.", value(events[0].Instructions))
}

// TestWithNoSubscriptionsAnIdleWorkerLooksTwiceALease: a router with connectors on and no
// subscription looks at start and then once a lease, a claim and when the next one is due
// each time: about 2 queries a lease, as eventforward's idle worker (#778), and never a busy
// loop. Its own service over its own pool, so the suite's router's queries are not counted;
// every subscription row is dropped first, as a deployment with no connections has none.
func (s *ConnectionEventsSuite) TestWithNoSubscriptionsAnIdleWorkerLooksTwiceALease() {
	_, err := s.store.DB().ExecContext(context.Background(), "DELETE FROM connection_event_subscriptions")
	s.Require().NoError(err)
	queries := s.idleWorker(idleLease)

	s.Require().Eventually(func() bool { return queries.Load() >= 2 }, settleFor, 10*time.Millisecond)
	time.Sleep(idleWindow)

	looks := idleWindow / idleLease
	s.GreaterOrEqual(queries.Load(), int64(2*looks-2), "it looked again once a lease")
	s.LessOrEqual(queries.Load(), int64(2*looks+4), "and no more often")
}

// TestAnIdleRouterTakesARowAnotherRouterAddsWithinALease: router A adds a subscription due now
// and stops before its worker asks for it; idle router B finds it at its next look, a lease
// later, and acts on it. Its connection does not exist, so B drops it.
func (s *ConnectionEventsSuite) TestAnIdleRouterTakesARowAnotherRouterAddsWithinALease() {
	queries := s.idleWorker(idleLease)
	s.Require().Eventually(func() bool { return queries.Load() >= 2 }, settleFor, 10*time.Millisecond)
	due := time.Now().UTC()
	left := store.ConnectionEventSubscription{CustomerID: s.customerID(), ConnectionID: "gone-" + s.utils.uuid(),
		ConfigID: s.utils.uuid(), Binding: "crm", Event: issueCreated, Key: "key", Token: s.utils.uuid(),
		SecretSealed: []byte("sealed"), KEKVersion: 1, Status: store.ConnectionEventPending, NextAttemptAt: &due}
	added, err := s.store.AddConnectionEventSubscription(context.Background(), &left)
	s.Require().NoError(err)
	s.Require().True(added)

	s.Eventually(func() bool { return len(s.held(left.ConnectionID)) == 0 }, 4*idleLease, 10*time.Millisecond)
}

// idleLease and idleWindow are the lease of an idle worker under test and how long it is
// watched: short enough for a test, long enough for several looks.
const (
	idleLease  = 200 * time.Millisecond
	idleWindow = time.Second
)

// idleWorker starts an MCP events service of its own over its own pool, with lease, and
// counts the queries it sends.
func (s *ConnectionEventsSuite) idleWorker(lease time.Duration) *countedQueries {
	own, err := store.Open(s.dsn())
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(own.Close()) })
	queries := &countedQueries{}
	own.DB().AddQueryHook(queries)
	events, err := mcpevents.New(mcpevents.Options{Store: own, Sessions: s.manager, Registry: s.connectors,
		Transports: s.transports, Secrets: s.sealer, PublicURL: s.server.URL, Lease: lease})
	s.Require().NoError(err)
	events.Start()
	s.T().Cleanup(events.Close)
	return queries
}

// setStatus writes a connection's status as the resolver and the delete write it.
func (s *ConnectionEventsSuite) setStatus(connection, status string) {
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE connector_connections SET status = ? WHERE id = ?", status, connection)
	s.Require().NoError(err)
}

// subscribed stores a config with binding, validates the binding's connection, and waits for
// the router to hold every subscription the binding declares as active.
func (s *ConnectionEventsSuite) subscribed(binding map[string]any) string {
	connection := binding["connection"].(map[string]any)["connection_id"].(string)
	s.config(binding)
	s.validate(connection)
	declared := len(binding["events"].([]map[string]any))
	s.Require().Eventually(func() bool {
		held := s.held(connection)
		for _, sub := range held {
			if sub.Status != store.ConnectionEventActive {
				return false
			}
		}
		return len(held) == declared
	}, settleFor, 20*time.Millisecond)
	return connection
}

// connector stores a connector of the app at the fake, taking a bearer token.
func (s *ConnectionEventsSuite) connector() string {
	id := "custom_crm" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: CRM
endpoints:
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [bearer]
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

// connection is a validated app connection to a new connector holding the fake's token.
func (s *ConnectionEventsSuite) connection() string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(s.connector()), bearer.Name), &created))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+created.ID+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{bearer.SuppliedToken: s.token}}, nil))
	s.validate(created.ID)
	return created.ID
}

// validate validates a connection through the API and requires it connected.
func (s *ConnectionEventsSuite) validate(id string) {
	var validation ConnectionValidation
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/validate", nil, &validation))
	s.Require().Equal(validationConnected, string(validation.Status), validation.Error)
}

// binding is a fixed binding, called crm, of the connection, granting no tool and declaring
// event with a filter and instructions.
func (s *ConnectionEventsSuite) binding(connection, event string) map[string]any {
	var read Connection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+connection, nil, &read))
	return map[string]any{"name": "crm", "connector_id": read.ConnectorID,
		"connection": map[string]any{"type": "fixed", "connection_id": connection},
		"tools":      []map[string]any{},
		"events": []map[string]any{{"event": event, "arguments": map[string]any{"project": "web"},
			"instructions": "Say what broke."}}}
}

// config is a new text agent config of the app, on a model each session notes, binding binding.
func (s *ConnectionEventsSuite) config(binding map[string]any) string {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "watcher-" + s.utils.uuid(), "mode": "text", "llm": "noted/noted-model",
		"connectors": []map[string]any{binding},
	}, &created))
	return created.Id
}

// held are the router's subscriptions of a connection.
func (s *ConnectionEventsSuite) held(connection string) []store.ConnectionEventSubscription {
	held := []store.ConnectionEventSubscription{}
	s.Require().NoError(s.store.DB().NewSelect().Model(&held).Where("customer_id = ?", s.customerID()).
		Where("connection_id = ?", connection).Order("created_at", "id").Scan(context.Background()))
	return held
}

// atTheFake are the fake's subscriptions whose callback is one of the connection's tokens, in
// the router's order.
func (s *ConnectionEventsSuite) atTheFake(connection string) []fakeprovider.EventSubscription {
	var found []fakeprovider.EventSubscription
	for _, sub := range s.held(connection) {
		for _, remote := range s.provider.EventSubscriptions() {
			if strings.HasSuffix(remote.URL, mcpevents.Path+sub.Token) {
				found = append(found, remote)
			}
		}
	}
	return found
}

// statuses are what the connection's subscriptions answered an Emit.
func (s *ConnectionEventsSuite) statuses(answered map[string]int, connection string) []int {
	var statuses []int
	for _, sub := range s.atTheFake(connection) {
		statuses = append(statuses, answered[sub.ID])
	}
	return statuses
}

// modelWasAsked reports whether any session's model was handed marker.
func (s *ConnectionEventsSuite) modelWasAsked(marker string) bool {
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	for _, model := range s.noted {
		for _, asked := range model.requests() {
			for _, message := range asked.Input {
				if strings.Contains(message.Content+llm.TextOf(message.Parts), marker) {
					return true
				}
			}
		}
	}
	return false
}

// dsn is the suite's own database, as SetupSuite opened it.
func (s *ConnectionEventsSuite) dsn() string {
	return testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN"))
}

// countedQueries counts the queries a pool sends.
type countedQueries struct{ atomic.Int64 }

func (*countedQueries) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context {
	return ctx
}

func (c *countedQueries) AfterQuery(context.Context, *bun.QueryEvent) { c.Add(1) }
