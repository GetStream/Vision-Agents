//go:build integration

package eventforward

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ForwarderSuite is the worker against Postgres and local TLS destinations: what it sends
// again, what it gives up, and that its own client is egress's.
type ForwarderSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	sealer *auth.Sealer
	// dsn is the suite's database, for a store of a test's own.
	dsn string
}

func TestForwarderSuite(t *testing.T) {
	suite.Run(t, new(ForwarderSuite))
}

func (s *ForwarderSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	// A database of this suite's own, as the store suite has, since it drops the schema.
	s.dsn = testenv.Database(dsn, "eventforward")
	db, err := store.Open(s.dsn)
	s.Require().NoError(err)
	s.store = db
	var database string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.sealer, err = auth.NewSealerWithKeyring(1, map[int]string{1: "test key one"})
	s.Require().NoError(err)
}

func (s *ForwarderSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

// The router's own client is egress's, which dials public addresses alone: a destination
// stored on loopback, as no create would let through, is never reached. It is plain HTTP, so
// any client that is not egress's reaches it.
func (s *ForwarderSuite) TestTheRoutersClientNeverReachesALoopbackDestination() {
	var hits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { hits.Add(1) }))
	s.T().Cleanup(target.Close)
	forwarder := s.forwarder(nil)
	customer := s.destinationOf(forwarder, target.URL)

	s.Require().NoError(forwarder.Forward(s.ctx, Event{CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"a":1}`)}))

	s.Never(func() bool { return hits.Load() > 0 }, 300*time.Millisecond, 20*time.Millisecond)
}

// After the first attempt and one for each wait, a forward that never got through is dropped.
func (s *ForwarderSuite) TestAForwardThatKeepsFailingIsGivenUpAfterItsRetries() {
	var hits atomic.Int32
	target := s.destination(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	})
	forwarder := s.forwarder(target.Client())
	customer := s.destinationOf(forwarder, target.URL)

	s.Require().NoError(forwarder.Forward(s.ctx, Event{CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"b":2}`)}))

	s.Require().Eventually(func() bool { return s.pending(customer) == 0 }, 5*time.Second, 10*time.Millisecond)
	s.Equal(int32(3), hits.Load(), "the first attempt and the two waits")
}

// The spec: «3xx: Failure. Following redirects causes unnecessary load on both the sender and
// the receiver» (https://www.standardwebhooks.com/).
func (s *ForwarderSuite) TestARedirectIsNotFollowedNorSentAgain() {
	var hits atomic.Int32
	target := s.destination(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Redirect(w, r, "/elsewhere", http.StatusTemporaryRedirect)
	})
	forwarder := s.forwarder(target.Client())
	customer := s.destinationOf(forwarder, target.URL+"/hooks")

	s.Require().NoError(forwarder.Forward(s.ctx, Event{CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"c":3}`)}))

	s.Require().Eventually(func() bool { return s.pending(customer) == 0 }, 5*time.Second, 10*time.Millisecond)
	s.Equal(int32(1), hits.Load())
}

// The review of #774 (AI-924) sent 16 forwards to one customer's URL that never answers, then
// one to another customer's, which arrived after 14.9 s: every send slot waited out
// attemptTimeout. With perDestination the dead URL holds two slots, and the other customer's
// forward goes out at once.
func (s *ForwarderSuite) TestADestinationThatNeverAnswersDoesNotHoldUpAnother() {
	release := make(chan struct{})
	var hung atomic.Int32
	arrived := make(chan struct{}, 1)
	target := s.destination(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/dead" {
			hung.Add(1)
			select {
			case <-release:
			case <-r.Context().Done():
			}
			return
		}
		arrived <- struct{}{}
	})
	s.T().Cleanup(func() { close(release) })
	forwarder := s.forwarder(target.Client())
	dead := s.destinationOf(forwarder, target.URL+"/dead")
	s.T().Cleanup(func() { s.dropForwards(dead) })
	live := s.destinationOf(forwarder, target.URL+"/live")
	for n := range inFlight {
		s.Require().NoError(forwarder.Forward(s.ctx, Event{CustomerID: dead, ConnectorID: "slack_bot", Body: fmt.Appendf(nil, `{"n":%d}`, n)}))
	}
	s.Require().Eventually(func() bool { return hung.Load() >= perDestination }, 5*time.Second, 10*time.Millisecond)

	s.Require().NoError(forwarder.Forward(s.ctx, Event{CustomerID: live, ConnectorID: "slack_bot", Body: []byte(`{"live":1}`)}))

	// 2 s: far under the 15 s of attemptTimeout the live forward waited before the cap, and
	// far over the 10 ms this suite's forwarder polls at, so a slow machine does not fail it.
	s.Eventually(func() bool { return len(arrived) == 1 }, 2*time.Second, 10*time.Millisecond,
		"another customer's forward waits for no slot the dead URL holds")
	s.Equal(int32(perDestination), hung.Load(), "the dead URL holds no more than its own slots")
}

// Slack Bolt refuses a request whose X-Slack-Request-Timestamp is more than 5 minutes old
// («const requestTimestampMaxDeltaMin = 5;», bolt-js src/receivers/verify-request.ts), so an
// attempt sent after that carries none of Slack's signature headers, and is verified with
// webhook-signature.
func (s *ForwarderSuite) TestAnAttemptPastTheProvidersWindowCarriesOnlyWebhookSignature() {
	sent := s.sentHeaders(time.Now().Add(-time.Second))

	s.Equal("application/json", sent.Get("Content-Type"))
	s.Empty(sent.Get("X-Slack-Signature"))
	s.Empty(sent.Get("X-Slack-Request-Timestamp"))
	s.NotEmpty(sent.Get(headerSignature))
}

func (s *ForwarderSuite) TestAnAttemptWithinTheProvidersWindowCarriesItsSignatureAsItCame() {
	sent := s.sentHeaders(time.Now().Add(time.Minute))

	s.Equal("v0=synthetic", sent.Get("X-Slack-Signature"))
	s.Equal("1759740000", sent.Get("X-Slack-Request-Timestamp"))
	s.NotEmpty(sent.Get(headerSignature))
}

// AI-926: with connectors on and no forward queued, a router sends Postgres nothing for
// forwarding after its look at start until a lease (1 min) has passed. The worker before it
// claimed once each poll: at this suite's 10 ms, about 50 claims in the 500 ms watched here.
func (s *ForwarderSuite) TestAForwarderWithNothingQueuedSendsNoQuery() {
	_, err := s.store.DB().ExecContext(s.ctx, "DELETE FROM connector_event_deliveries")
	s.Require().NoError(err)
	own, err := store.Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(own.Close()) })
	queries := &queryCount{}
	own.DB().AddQueryHook(queries)
	forwarder, err := New(Options{Store: own, Secrets: s.sealer, Poll: 10 * time.Millisecond})
	s.Require().NoError(err)

	forwarder.Start()
	s.T().Cleanup(forwarder.Close)

	// The look at start: one claim, and when the next forward is due, which is never.
	s.Require().Eventually(func() bool { return queries.Load() == 2 }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return queries.Load() > 2 }, 500*time.Millisecond, 10*time.Millisecond)
}

// A router that stopped with forwards queued has them sent by the next one to start: the look
// at start (AI-926). Here the first forwarder queues and is never started, as a router that
// stopped right after the ack.
func (s *ForwarderSuite) TestAForwardQueuedBeforeTheForwarderStartedIsSent() {
	var hits atomic.Int32
	target := s.destination(func(w http.ResponseWriter, _ *http.Request) { hits.Add(1) })
	stopped, err := New(Options{Store: s.store, Secrets: s.sealer, HTTP: target.Client()})
	s.Require().NoError(err)
	customer := s.destinationOf(stopped, target.URL)
	s.Require().NoError(stopped.Forward(s.ctx, Event{CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"queued":1}`)}))

	s.forwarder(target.Client())

	s.Require().Eventually(func() bool { return s.pending(customer) == 0 }, 5*time.Second, 10*time.Millisecond)
	s.Equal(int32(1), hits.Load())
}

// The review of #778: router B starts and finds nothing queued; then router A queues a
// forward and stops before it sends it, so no Forward and no finished send wakes B. B looks
// again a lease after its last look and sends it. Here the lease is 200 ms; the bound, 2 s, is
// far over that and the 10 ms poll, and far under forever, which is what B waited before.
func (s *ForwarderSuite) TestAnIdleRouterSendsAForwardAnotherRouterLeftQueued() {
	_, err := s.store.DB().ExecContext(s.ctx, "DELETE FROM connector_event_deliveries")
	s.Require().NoError(err)
	var hits atomic.Int32
	target := s.destination(func(w http.ResponseWriter, _ *http.Request) { hits.Add(1) })
	s.routerAfterItsFirstLook(target.Client())
	stopped, err := New(Options{Store: s.store, Secrets: s.sealer, HTTP: target.Client()})
	s.Require().NoError(err)
	customer := s.destinationOf(stopped, target.URL)

	s.Require().NoError(stopped.Forward(s.ctx, Event{CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"left":1}`)}))

	s.Require().Eventually(func() bool { return hits.Load() == 1 }, 2*time.Second, 10*time.Millisecond)
	s.Require().Eventually(func() bool { return s.pending(customer) == 0 }, 2*time.Second, 10*time.Millisecond)
}

// The same with a retry queued an hour out: B waits a lease, not the hour, before it looks
// again and finds what router A left.
func (s *ForwarderSuite) TestARouterWaitingOnALaterRetryStillSendsAForwardAnotherRouterLeft() {
	_, err := s.store.DB().ExecContext(s.ctx, "DELETE FROM connector_event_deliveries")
	s.Require().NoError(err)
	var hits atomic.Int32
	target := s.destination(func(w http.ResponseWriter, _ *http.Request) { hits.Add(1) })
	stopped, err := New(Options{Store: s.store, Secrets: s.sealer, HTTP: target.Client()})
	s.Require().NoError(err)
	later := s.destinationOf(stopped, target.URL)
	_, err = s.store.QueueEventDeliveries(s.ctx, later, "slack_bot", []string{store.ForwardAll},
		store.EventDelivery{ID: "msg_later", Body: []byte(`{"later":1}`), NextAttemptAt: time.Now().Add(time.Hour)})
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.dropForwards(later) })
	s.routerAfterItsFirstLook(target.Client())
	customer := s.destinationOf(stopped, target.URL)

	s.Require().NoError(stopped.Forward(s.ctx, Event{CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"left":2}`)}))

	s.Require().Eventually(func() bool { return s.pending(customer) == 0 }, 2*time.Second, 10*time.Millisecond)
	s.Equal(int32(1), hits.Load())
}

// A forward for a customer without destinations queues nothing.
func (s *ForwarderSuite) TestACustomerWithoutDestinationsQueuesNothing() {
	forwarder := s.forwarder(nil)

	s.Require().NoError(forwarder.Forward(s.ctx, Event{CustomerID: "nobody-" + uuid.NewString(), ConnectorID: "slack_bot", Body: []byte(`{}`)}))

	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_event_deliveries d "+
		"JOIN connector_event_destinations t ON t.id = d.destination_id WHERE t.customer_id LIKE 'nobody-%'").Scan(&rows))
	s.Zero(rows)
}

// forwarder is a started forwarder whose retries wait a millisecond, sending through client;
// nil is egress's.
func (s *ForwarderSuite) forwarder(client *http.Client) *Forwarder {
	forwarder, err := New(Options{
		Store: s.store, Secrets: s.sealer, HTTP: client,
		Retries: []time.Duration{time.Millisecond, time.Millisecond}, Poll: 10 * time.Millisecond,
	})
	s.Require().NoError(err)
	forwarder.Start()
	s.T().Cleanup(forwarder.Close)
	return forwarder
}

// destinationOf stores a destination of a new customer at url, past the create check, and
// returns the customer.
func (s *ForwarderSuite) destinationOf(forwarder *Forwarder, url string) string {
	destination := store.EventDestination{
		ID: uuid.NewString(), CustomerID: "customer-" + uuid.NewString(), ConnectorID: "slack_bot", URL: url, Forward: store.ForwardAll,
	}
	secret, err := forwarder.NewSecret(destination.CustomerID, destination.ConnectorID, destination.ID)
	s.Require().NoError(err)
	destination.SecretSealed, destination.KEKVersion = secret.Sealed, secret.Version
	s.Require().NoError(s.store.CreateEventDestination(s.ctx, &destination))
	return destination.CustomerID
}

// destination is a TLS server on loopback answering with handler. Its Client trusts its
// certificate.
func (s *ForwarderSuite) destination(handler http.HandlerFunc) *httptest.Server {
	server := httptest.NewTLSServer(handler)
	s.T().Cleanup(server.Close)
	return server
}

// sentHeaders forwards a delivery with Slack's headers, which verify until headersUntil, and
// returns the headers the destination got.
func (s *ForwarderSuite) sentHeaders(headersUntil time.Time) http.Header {
	got := make(chan http.Header, 1)
	target := s.destination(func(w http.ResponseWriter, r *http.Request) { got <- r.Header.Clone() })
	forwarder := s.forwarder(target.Client())
	customer := s.destinationOf(forwarder, target.URL)

	s.Require().NoError(forwarder.Forward(s.ctx, Event{
		CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"type":"event_callback"}`), HeadersUntil: headersUntil,
		Headers: map[string]string{
			"Content-Type": "application/json", "X-Slack-Signature": "v0=synthetic", "X-Slack-Request-Timestamp": "1759740000",
		},
	}))

	select {
	case header := <-got:
		return header
	case <-time.After(5 * time.Second):
		s.FailNow("the destination got no forward")
		return nil
	}
}

// routerAfterItsFirstLook starts router B, on a pool of its own, with a 200 ms lease, and
// returns once B's look at start is over (a claim and when the next forward is due), so what
// is queued after that only a later look of B's finds.
func (s *ForwarderSuite) routerAfterItsFirstLook(client *http.Client) {
	own, err := store.Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(own.Close()) })
	queries := &queryCount{}
	own.DB().AddQueryHook(queries)
	router, err := New(Options{Store: own, Secrets: s.sealer, HTTP: client, Poll: 10 * time.Millisecond, Lease: 200 * time.Millisecond})
	s.Require().NoError(err)
	router.Start()
	s.T().Cleanup(router.Close)
	s.Require().Eventually(func() bool { return queries.Load() >= 2 }, 5*time.Second, 10*time.Millisecond, "router B's look at start")
}

// dropForwards deletes the forwards of a customer not yet done with.
func (s *ForwarderSuite) dropForwards(customer string) {
	_, err := s.store.DB().ExecContext(s.ctx, "DELETE FROM connector_event_deliveries d USING connector_event_destinations t "+
		"WHERE t.id = d.destination_id AND t.customer_id = ?", customer)
	s.Require().NoError(err)
}

// queryCount counts the queries a store sent Postgres.
type queryCount struct{ atomic.Int32 }

func (q *queryCount) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context {
	return ctx
}

func (q *queryCount) AfterQuery(context.Context, *bun.QueryEvent) { q.Add(1) }

// pending is how many forwards of a customer are not done with.
func (s *ForwarderSuite) pending(customer string) int {
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_event_deliveries d "+
		"JOIN connector_event_destinations t ON t.id = d.destination_id WHERE t.customer_id = ?", customer).Scan(&rows))
	return rows
}
