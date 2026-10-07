//go:build integration

package store

import (
	"strings"
	"time"
)

// eventSubscription stores a subscription of customer's connection, due at s.base, for config
// and binding crm, keyed by key.
func (s *StoreSuite) eventSubscription(customerID, connectionID, configID, key string) ConnectionEventSubscription {
	due := s.base
	sub := ConnectionEventSubscription{
		CustomerID: customerID, ConnectionID: connectionID, ConfigID: configID, Binding: "crm",
		Event: "issue.created", Key: key, Token: "token-" + connectionID + "-" + configID + "-" + key,
		SecretSealed: []byte("sealed"), KEKVersion: 1, Status: ConnectionEventPending, NextAttemptAt: &due,
	}
	added, err := s.store.AddConnectionEventSubscription(s.ctx, &sub)
	s.Require().NoError(err)
	s.Require().True(added)
	return sub
}

// connectionEventSubscriptions are a customer's connection's subscriptions, oldest first.
func (s *StoreSuite) connectionEventSubscriptions(customerID, connectionID string) ([]ConnectionEventSubscription, error) {
	subs := []ConnectionEventSubscription{}
	err := s.store.DB().NewSelect().Model(&subs).Where("customer_id = ?", customerID).
		Where("connection_id = ?", connectionID).Order("created_at", "id").Scan(s.ctx)
	return subs, err
}

func (s *StoreSuite) TestTheSameSubscriptionAddedTwiceIsStoredOnce() {
	s.eventSubscription("acme-app", "conn-1", "config-1", "key-1")
	again := ConnectionEventSubscription{
		CustomerID: "acme-app", ConnectionID: "conn-1", ConfigID: "config-1", Binding: "crm",
		Event: "issue.created", Key: "key-1", Token: "another-token", SecretSealed: []byte("sealed"), KEKVersion: 1,
		Status: ConnectionEventPending,
	}

	added, err := s.store.AddConnectionEventSubscription(s.ctx, &again)

	s.Require().NoError(err)
	s.False(added)
	held, err := s.connectionEventSubscriptions("acme-app", "conn-1")
	s.Require().NoError(err)
	s.Require().Len(held, 1)
	s.Equal("token-conn-1-config-1-key-1", held[0].Token)
}

func (s *StoreSuite) TestAConnectionsSubscriptionsAreOnlyItsCustomers() {
	s.eventSubscription("acme-app", "conn-1", "config-1", "key-1")

	held, err := s.connectionEventSubscriptions("other-app", "conn-1")
	s.Require().NoError(err)
	s.Empty(held)
	s.Require().NoError(s.store.DeleteConnectionEventSubscriptions(s.ctx, "other-app", "conn-1"))
	held, err = s.connectionEventSubscriptions("acme-app", "conn-1")
	s.Require().NoError(err)
	s.Len(held, 1, "another customer's delete leaves it")
}

func (s *StoreSuite) TestAClaimLeasesWhatIsDueAndNotWhatIsNot() {
	due := s.eventSubscription("acme-app", "conn-1", "config-1", "key-1")
	later := s.eventSubscription("acme-app", "conn-1", "config-1", "key-2")
	future := s.base.Add(time.Hour)
	later.NextAttemptAt = &future
	s.Require().NoError(s.store.SaveConnectionEventSubscription(s.ctx, &later))

	claimed, err := s.store.ClaimConnectionEventSubscriptions(s.ctx, s.base, 10, s.base.Add(time.Minute))
	s.Require().NoError(err)
	again, err := s.store.ClaimConnectionEventSubscriptions(s.ctx, s.base, 10, s.base.Add(time.Minute))
	s.Require().NoError(err)

	s.Require().Len(claimed, 1)
	s.Equal(due.ID, claimed[0].ID)
	s.True(claimed[0].NextAttemptAt.Equal(s.base.Add(time.Minute)), "pushed to the lease's end")
	s.Empty(again, "a leased one is not taken again before its lease ends")
}

// TestTwoRoutersClaimingAtOnceTakeEachSubscriptionOnce: router A's claim pauses after it read
// the due rows; router B claims them meanwhile. A then takes none of them, so the server is
// never asked for one subscription by two routers at once.
func (s *StoreSuite) TestTwoRoutersClaimingAtOnceTakeEachSubscriptionOnce() {
	for _, key := range []string{"one", "two", "three"} {
		s.eventSubscription("acme-app", "conn-1", "config-1", key)
	}
	routerA, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(routerA.Close()) })
	read := "    WHERE next_attempt_at <= ?\n"
	s.Require().Contains(claimConnectionEventSubscriptionsQuery, read)
	paused := strings.Replace(claimConnectionEventSubscriptionsQuery, read,
		"    WHERE next_attempt_at <= ? AND (SELECT 1 FROM pg_sleep(0.5)) = 1\n", 1)
	type claim struct {
		claimed []ConnectionEventSubscription
		err     error
	}
	byA := make(chan claim, 1)

	go func() {
		claimed, err := routerA.claimConnectionEventSubscriptions(s.ctx, paused, s.base, 10, s.base.Add(time.Minute))
		byA <- claim{claimed, err}
	}()
	s.Require().Eventually(func() bool {
		var sleeping int
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT count(*) FROM pg_stat_activity WHERE wait_event = 'PgSleep' AND datname = current_database()").Scan(&sleeping))
		return sleeping == 1
	}, 5*time.Second, 10*time.Millisecond, "router A's claim is in its pause")
	byB, err := s.store.ClaimConnectionEventSubscriptions(s.ctx, s.base, 10, s.base.Add(time.Minute))
	s.Require().NoError(err)
	a := <-byA
	s.Require().NoError(a.err)

	s.Len(byB, 3)
	s.Empty(a.claimed, "router A takes none of what router B leased")
}

func (s *StoreSuite) TestTheNextSubscriptionDueIsTheFirstAndNoneWhenNoneIsEverDue() {
	_, found, err := s.store.NextConnectionEventSubscriptionAt(s.ctx)
	s.Require().NoError(err)
	s.False(found, "no subscriptions")
	sub := s.eventSubscription("acme-app", "conn-1", "config-1", "key-1")
	sub.NextAttemptAt = nil
	s.Require().NoError(s.store.SaveConnectionEventSubscription(s.ctx, &sub))

	_, found, err = s.store.NextConnectionEventSubscriptionAt(s.ctx)
	s.Require().NoError(err)
	s.False(found, "a grant that does not expire is never due")
	s.eventSubscription("acme-app", "conn-1", "config-1", "key-2")
	next, found, err := s.store.NextConnectionEventSubscriptionAt(s.ctx)
	s.Require().NoError(err)
	s.True(found)
	s.True(next.Equal(s.base))
}

func (s *StoreSuite) TestAnEventIsClaimedOnceAndGoesWithItsSubscription() {
	sub := s.eventSubscription("acme-app", "conn-1", "config-1", "key-1")

	first, err := s.store.ClaimConnectionEvent(s.ctx, sub.ID, "evt-1")
	s.Require().NoError(err)
	again, err := s.store.ClaimConnectionEvent(s.ctx, sub.ID, "evt-1")
	s.Require().NoError(err)
	s.Require().NoError(s.store.DeleteConnectionEventSubscription(s.ctx, sub.ID))

	s.True(first)
	s.False(again)
	var left int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connection_event_deliveries").Scan(&left))
	s.Zero(left)
	_, err = s.store.ConnectionEventSubscriptionByToken(s.ctx, sub.Token)
	s.ErrorIs(err, ErrNoConnectionEventSubscription)
}

func (s *StoreSuite) TestMakingAConnectionDueMakesEveryOneOfItsSubscriptionsDue() {
	sub := s.eventSubscription("acme-app", "conn-1", "config-1", "key-1")
	other := s.eventSubscription("acme-app", "conn-2", "config-1", "key-1")
	for _, held := range []*ConnectionEventSubscription{&sub, &other} {
		held.NextAttemptAt = nil
		s.Require().NoError(s.store.SaveConnectionEventSubscription(s.ctx, held))
	}

	s.Require().NoError(s.store.DueConnectionEventSubscriptions(s.ctx, "acme-app", "conn-1", s.base))

	claimed, err := s.store.ClaimConnectionEventSubscriptions(s.ctx, s.base, 10, s.base.Add(time.Minute))
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Equal(sub.ID, claimed[0].ID)
}
