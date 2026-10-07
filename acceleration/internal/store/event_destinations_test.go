//go:build integration

package store

import (
	"time"
)

// eventDestination stores a destination of customer's for connector, forwarding forward.
func (s *StoreSuite) eventDestination(customerID, connectorID, id, forward string) EventDestination {
	destination := EventDestination{
		ID: id, CustomerID: customerID, ConnectorID: connectorID, URL: "https://hooks.example.com/" + id,
		Forward: forward, SecretSealed: []byte("sealed-" + id), KEKVersion: 1,
	}
	s.Require().NoError(s.store.CreateEventDestination(s.ctx, &destination))
	return destination
}

// queued are the destinations a delivery id is queued for.
func (s *StoreSuite) queued(id string) []string {
	var destinations []string
	rows, err := s.store.DB().QueryContext(s.ctx, "SELECT destination_id FROM connector_event_deliveries WHERE id = ? ORDER BY destination_id", id)
	s.Require().NoError(err)
	defer rows.Close() //nolint:errcheck // read to the end below
	for rows.Next() {
		var destination string
		s.Require().NoError(rows.Scan(&destination))
		destinations = append(destinations, destination)
	}
	s.Require().NoError(rows.Err())
	return destinations
}

func (s *StoreSuite) TestAFourthEventDestinationOfAConnectorIsRefused() {
	for _, id := range []string{"one", "two", "three"} {
		s.eventDestination("acme-app", "slack_bot", id, ForwardAll)
	}
	s.eventDestination("acme-app", "linear", "another-connector", ForwardAll)

	fourth := EventDestination{
		ID: "four", CustomerID: "acme-app", ConnectorID: "slack_bot", URL: "https://hooks.example.com/four",
		Forward: ForwardAll, SecretSealed: []byte("sealed"), KEKVersion: 1,
	}
	s.ErrorIs(s.store.CreateEventDestination(s.ctx, &fourth), ErrEventDestinationsFull)
}

func (s *StoreSuite) TestAnEventDestinationIsOnlyItsOwnCustomers() {
	s.eventDestination("acme-app", "slack_bot", "acme-hook", ForwardAll)

	listed, err := s.store.EventDestinations(s.ctx, "other-app", "slack_bot", 0, nil)
	s.Require().NoError(err)
	s.Empty(listed)
	s.ErrorIs(s.store.DeleteEventDestination(s.ctx, "other-app", "slack_bot", "acme-hook"), ErrNoEventDestination)
	_, err = s.store.RotateEventDestinationSecret(s.ctx, "other-app", "slack_bot", "acme-hook", []byte("theirs"), 1, s.base)
	s.ErrorIs(err, ErrNoEventDestination)
}

func (s *StoreSuite) TestAHandledDeliveryIsQueuedOnlyForDestinationsOfEveryEvent() {
	s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)
	s.eventDestination("acme-app", "slack_bot", "unhandled", ForwardUnhandled)
	s.eventDestination("other-app", "slack_bot", "theirs", ForwardAll)

	queued, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll}, EventDelivery{ID: "msg_handled", Body: []byte("{}"), NextAttemptAt: s.base})
	s.Require().NoError(err)
	s.Equal(1, queued)
	s.Equal([]string{"every"}, s.queued("msg_handled"))

	queued, err = s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll, ForwardUnhandled}, EventDelivery{ID: "msg_unhandled", Body: []byte("{}"), NextAttemptAt: s.base})
	s.Require().NoError(err)
	s.Equal(2, queued)
	s.Equal([]string{"every", "unhandled"}, s.queued("msg_unhandled"))
}

func (s *StoreSuite) TestADeliveryQueuedTwiceUnderOneIDIsOneRow() {
	s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)
	delivery := EventDelivery{ID: "msg_same", Headers: map[string]string{"X-Slack-Signature": "v0=abc"}, Body: []byte("{}"), NextAttemptAt: s.base}

	_, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll, ForwardUnhandled}, delivery)
	s.Require().NoError(err)
	queued, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll, ForwardUnhandled}, delivery)

	s.Require().NoError(err)
	s.Zero(queued)
	s.Len(s.queued("msg_same"), 1)
}

func (s *StoreSuite) TestAClaimedDeliveryIsNotClaimedAgainUntilItsLeaseRunsOut() {
	destination := s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)
	_, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll, ForwardUnhandled}, EventDelivery{
		ID: "msg_one", Headers: map[string]string{"Content-Type": "application/json"}, Body: []byte(`{"a":1}`), NextAttemptAt: s.base,
	})
	s.Require().NoError(err)

	claimed, err := s.store.ClaimEventDeliveries(s.ctx, s.base, 10, 10, nil, s.base.Add(time.Minute))
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Equal(destination.URL, claimed[0].URL)
	s.Equal([]byte(`{"a":1}`), claimed[0].Body)
	s.Equal(map[string]string{"Content-Type": "application/json"}, claimed[0].Headers)
	s.Equal(destination.SecretSealed, claimed[0].SecretSealed)

	again, err := s.store.ClaimEventDeliveries(s.ctx, s.base.Add(59*time.Second), 10, 10, nil, s.base.Add(2*time.Minute))
	s.Require().NoError(err)
	s.Empty(again, "another worker leaves it while the lease runs")
	after, err := s.store.ClaimEventDeliveries(s.ctx, s.base.Add(time.Minute), 10, 10, nil, s.base.Add(2*time.Minute))
	s.Require().NoError(err)
	s.Len(after, 1, "a stopped worker's delivery is taken once its lease ran out")
}

// AI-924: a destination already sending one of its two takes one more, and another
// destination's delivery is taken past the rest of its own.
func (s *StoreSuite) TestAClaimTakesAtMostPerDestinationLessWhatIsBeingSent() {
	s.eventDestination("acme-app", "slack_bot", "slow", ForwardAll)
	s.eventDestination("globex-app", "slack_bot", "other", ForwardAll)
	for _, id := range []string{"msg_1", "msg_2", "msg_3"} {
		_, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll}, EventDelivery{ID: id, Body: []byte("{}"), NextAttemptAt: s.base})
		s.Require().NoError(err)
	}
	_, err := s.store.QueueEventDeliveries(s.ctx, "globex-app", "slack_bot", []string{ForwardAll},
		EventDelivery{ID: "msg_other", Body: []byte("{}"), NextAttemptAt: s.base.Add(time.Second)})
	s.Require().NoError(err)

	claimed, err := s.store.ClaimEventDeliveries(s.ctx, s.base.Add(time.Second), 10, 2, map[string]int{"slow": 1}, s.base.Add(time.Minute))

	s.Require().NoError(err)
	taken := map[string]int{}
	for _, delivery := range claimed {
		taken[delivery.DestinationID]++
	}
	s.Equal(map[string]int{"slow": 1, "other": 1}, taken)
}

func (s *StoreSuite) TestTheNextDeliveryIsTheOneDueFirstAndNoneWhenNothingIsQueued() {
	_, queued, err := s.store.NextEventDeliveryAt(s.ctx)
	s.Require().NoError(err)
	s.False(queued)
	s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)
	for i, id := range []string{"msg_late", "msg_early"} {
		_, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll},
			EventDelivery{ID: id, Body: []byte("{}"), NextAttemptAt: s.base.Add(time.Duration(1-i) * time.Minute)})
		s.Require().NoError(err)
	}

	next, queued, err := s.store.NextEventDeliveryAt(s.ctx)

	s.Require().NoError(err)
	s.True(queued)
	s.True(next.Equal(s.base), "%s is not %s", next, s.base)
}

// AI-924: the store keeps until when the provider's own headers verify, for the worker to read.
func (s *StoreSuite) TestAClaimedDeliveryKeepsUntilWhenItsProviderHeadersVerify() {
	s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)
	until := s.base.Add(5 * time.Minute)
	_, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll},
		EventDelivery{ID: "msg_one", Body: []byte("{}"), NextAttemptAt: s.base, ProviderHeadersUntil: &until})
	s.Require().NoError(err)

	claimed, err := s.store.ClaimEventDeliveries(s.ctx, s.base, 10, 2, nil, s.base.Add(time.Minute))

	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Require().NotNil(claimed[0].ProviderHeadersUntil)
	s.True(claimed[0].ProviderHeadersUntil.Equal(until))
}

func (s *StoreSuite) TestARetriedDeliveryIsDueAtItsNextAttempt() {
	s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)
	_, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll, ForwardUnhandled}, EventDelivery{ID: "msg_one", Body: []byte("{}"), NextAttemptAt: s.base})
	s.Require().NoError(err)

	s.Require().NoError(s.store.RetryEventDelivery(s.ctx, "every", "msg_one", 1, s.base.Add(5*time.Second)))

	early, err := s.store.ClaimEventDeliveries(s.ctx, s.base.Add(4*time.Second), 10, 10, nil, s.base.Add(time.Minute))
	s.Require().NoError(err)
	s.Empty(early)
	due, err := s.store.ClaimEventDeliveries(s.ctx, s.base.Add(5*time.Second), 10, 10, nil, s.base.Add(time.Minute))
	s.Require().NoError(err)
	s.Require().Len(due, 1)
	s.Equal(1, due[0].Attempts)
}

func (s *StoreSuite) TestDeletingADestinationDropsItsPendingDeliveries() {
	s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)
	_, err := s.store.QueueEventDeliveries(s.ctx, "acme-app", "slack_bot", []string{ForwardAll, ForwardUnhandled}, EventDelivery{ID: "msg_one", Body: []byte("{}"), NextAttemptAt: s.base})
	s.Require().NoError(err)

	s.Require().NoError(s.store.DeleteEventDestination(s.ctx, "acme-app", "slack_bot", "every"))

	s.Empty(s.queued("msg_one"))
}

func (s *StoreSuite) TestARotationKeepsTheReplacedSecretUntilItsTime() {
	s.eventDestination("acme-app", "slack_bot", "every", ForwardAll)

	rotated, err := s.store.RotateEventDestinationSecret(s.ctx, "acme-app", "slack_bot", "every", []byte("sealed-new"), 2, s.base)

	s.Require().NoError(err)
	s.Equal([]byte("sealed-new"), rotated.SecretSealed)
	s.Equal(2, rotated.KEKVersion)
	s.Equal([]byte("sealed-every"), rotated.PreviousSecretSealed)
	s.Equal(1, rotated.PreviousKEKVersion)
	s.Require().NotNil(rotated.PreviousUntil)
	s.True(rotated.PreviousUntil.Equal(s.base))
}

func (s *StoreSuite) TestEventDestinationsArePagedNewestFirst() {
	s.eventDestination("acme-app", "slack_bot", "older", ForwardAll)
	s.eventDestination("acme-app", "slack_bot", "newer", ForwardAll)

	first, err := s.store.EventDestinations(s.ctx, "acme-app", "slack_bot", 1, nil)
	s.Require().NoError(err)
	s.Require().Len(first, 2, "one more than the page, to tell there is another")
	s.Equal("newer", first[0].ID)
	next, err := s.store.EventDestinations(s.ctx, "acme-app", "slack_bot", 1, &EventDestinationPosition{CreatedAt: first[0].CreatedAt, ID: first[0].ID})
	s.Require().NoError(err)
	s.Require().Len(next, 1)
	s.Equal("older", next[0].ID)
}
