//go:build integration

package eventforward

import (
	"crypto/rand"
	"database/sql"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// forwardsCounted is a customer's webhooks_out row in the activity record.
type forwardsCounted struct {
	requests, refused int64
	code              sql.NullString
	organization      string
}

// countingForwarder is a started forwarder, as forwarder makes, that counts in an activity
// record of its own, which writes only when the test closes it.
func (s *ForwarderSuite) countingForwarder(client *http.Client) (*Forwarder, *activity.Recorder) {
	record, err := activity.New(activity.Options{Store: s.store, Interval: time.Hour, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	forwarder, err := New(Options{
		Store: s.store, Secrets: s.sealer, HTTP: client, Activity: record,
		Retries: []time.Duration{time.Millisecond, time.Millisecond}, Poll: 10 * time.Millisecond,
	})
	s.Require().NoError(err)
	forwarder.Start()
	s.T().Cleanup(forwarder.Close)
	return forwarder, record
}

// forwardedOnce forwards one event of customer, waits until it is done with, and returns the
// customer's webhooks_out row once the record is written.
func (s *ForwarderSuite) forwardedOnce(forwarder *Forwarder, record *activity.Recorder, customer string) forwardsCounted {
	s.Require().NoError(forwarder.Forward(s.ctx, Event{CustomerID: customer, ConnectorID: "slack_bot", Body: []byte(`{"d":4}`)}))
	s.Require().Eventually(func() bool { return s.pending(customer) == 0 }, 5*time.Second, 10*time.Millisecond)
	record.Close()

	var row forwardsCounted
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, `
SELECT requests, refused, last_refusal_code, organization_id FROM customer_activity
WHERE app_id = ? AND area = ? AND client = ?`, customer, activity.AreaWebhooksOut, activity.ClientRouter).
		Scan(&row.requests, &row.refused, &row.code, &row.organization))
	return row
}

// Defect: a destination answering 410 counted as delivered, without why, or under no
// organization, so the day's digest of the app's organization leaves it out.
func (s *ForwarderSuite) TestAForwardTheDestinationRefusesIsCountedRefusedUnderItsOrganization() {
	target := s.destination(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusGone) })
	forwarder, record := s.countingForwarder(target.Client())
	customer := s.destinationOf(forwarder, target.URL)
	organization := "org-" + rand.Text()
	s.Require().NoError(s.store.JoinOrganization(s.ctx, customer, organization))

	row := s.forwardedOnce(forwarder, record, customer)

	s.Equal(int64(1), row.requests)
	s.Equal(int64(1), row.refused)
	s.Equal(codeRefused, row.code.String)
	s.Equal(organization, row.organization)
}

// Defect: each failed attempt counted as a refusal, so a forward that got through on its
// second attempt reads as refused.
func (s *ForwarderSuite) TestAForwardSentOnItsSecondAttemptIsCountedOnceAndNotRefused() {
	var hits atomic.Int32
	target := s.destination(func(w http.ResponseWriter, _ *http.Request) {
		if hits.Add(1) == 1 {
			w.WriteHeader(http.StatusBadGateway)
		}
	})
	forwarder, record := s.countingForwarder(target.Client())
	customer := s.destinationOf(forwarder, target.URL)

	row := s.forwardedOnce(forwarder, record, customer)

	s.Equal(int32(2), hits.Load())
	s.Equal(int64(1), row.requests)
	s.Equal(int64(0), row.refused)
	s.False(row.code.Valid, "code %q", row.code.String)
}

// Defect: a forward given up counted once per attempt, or under the code of a destination that
// said no, when it never answered at all.
func (s *ForwarderSuite) TestAForwardGivenUpAfterItsRetriesIsCountedRefusedOnce() {
	var hits atomic.Int32
	target := s.destination(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	})
	forwarder, record := s.countingForwarder(target.Client())
	customer := s.destinationOf(forwarder, target.URL)

	row := s.forwardedOnce(forwarder, record, customer)

	s.Equal(int32(3), hits.Load(), "the first attempt and the two waits")
	s.Equal(int64(1), row.requests)
	s.Equal(int64(1), row.refused)
	s.Equal(codeUnavailable, row.code.String)
}

// Defect: a forward the router never sent, its signing secret not opening, counted under a
// code that blames the destination.
func (s *ForwarderSuite) TestAForwardThatCouldNotBeSignedIsCountedAsNotSent() {
	var hits atomic.Int32
	target := s.destination(func(http.ResponseWriter, *http.Request) { hits.Add(1) })
	forwarder, record := s.countingForwarder(target.Client())
	destination := store.EventDestination{
		ID: uuid.NewString(), CustomerID: "customer-" + uuid.NewString(), ConnectorID: "slack_bot", URL: target.URL,
		Forward: store.ForwardAll, SecretSealed: []byte("sealed by nobody"), KEKVersion: 1,
	}
	s.Require().NoError(s.store.CreateEventDestination(s.ctx, &destination))

	row := s.forwardedOnce(forwarder, record, destination.CustomerID)

	s.Equal(int32(0), hits.Load())
	s.Equal(int64(1), row.refused)
	s.Equal(codeNotSent, row.code.String)
}
