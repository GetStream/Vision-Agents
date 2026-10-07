//go:build integration

package eventforward

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

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
	db, err := store.Open(testenv.Database(dsn, "eventforward"))
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

// pending is how many forwards of a customer are not done with.
func (s *ForwarderSuite) pending(customer string) int {
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_event_deliveries d "+
		"JOIN connector_event_destinations t ON t.id = d.destination_id WHERE t.customer_id = ?", customer).Scan(&rows))
	return rows
}
