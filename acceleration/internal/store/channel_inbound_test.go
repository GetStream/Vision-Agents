//go:build integration

package store

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// inbound queues a message of the thread channel channelID, linking the channel first, due at
// s.base; change sets the fields a test cares about.
func (s *StoreSuite) inbound(channelID, providerMessageID string, change ...func(*ChannelInbound)) ChannelInbound {
	message := s.inboundMessage(channelID, providerMessageID, change...)
	queued, err := s.store.QueueChannelInbound(s.ctx, message)
	s.Require().NoError(err)
	s.Require().True(queued)
	return message
}

// inboundMessage is the message inbound queues, its thread channel linked, not yet queued.
func (s *StoreSuite) inboundMessage(channelID, providerMessageID string, change ...func(*ChannelInbound)) ChannelInbound {
	_, err := s.store.LinkChannelThread(s.ctx, thread(channelID, "C0000CHAN:"+channelID))
	s.Require().NoError(err)
	message := ChannelInbound{
		ChannelID: channelID, ProviderMessageID: providerMessageID, CustomerID: "acme-app", ConnectorID: "acme",
		ProviderUnitID: "T0000TEAM", ThreadKey: "C0000CHAN:" + channelID, ConfigID: "config-one", ConfigName: "Ada",
		AuthorID: "U0000AUTHOR", Text: "hello " + providerMessageID, Raw: []byte(`{"event":"` + providerMessageID + `"}`),
		NextAttemptAt: s.base,
	}
	for _, apply := range change {
		apply(&message)
	}
	return message
}

// claim claims every message due at at, under a lease of a minute.
func (s *StoreSuite) claim(at time.Time) []ChannelInbound {
	claimed, err := s.store.ClaimChannelInbound(s.ctx, at, 10, at.Add(time.Minute))
	s.Require().NoError(err)
	return claimed
}

// claimedOne claims at at and requires the one message it takes to be providerMessageID.
func (s *StoreSuite) claimedOne(at time.Time, providerMessageID string) ChannelInbound {
	claimed := s.claim(at)
	s.Require().Len(claimed, 1)
	s.Require().Equal(providerMessageID, claimed[0].ProviderMessageID)
	return claimed[0]
}

// inboundState is the state and attempts of a message's row.
func (s *StoreSuite) inboundState(channelID, providerMessageID string) (state string, attempts int) {
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT state, attempts FROM channel_thread_inbound WHERE channel_id = ? AND provider_message_id = ?",
		channelID, providerMessageID).Scan(&state, &attempts))
	return state, attempts
}

func (s *StoreSuite) TestAClaimedInboundMessageHasEverythingItWasQueuedWith() {
	until := s.base.Add(5 * time.Minute)
	s.inbound("thread-one", "1759740000.000100", func(message *ChannelInbound) {
		message.Media = []core.Media{{URL: "https://files.example.com/a.png", MIMEType: "image/png"}}
		message.Forward = &ChannelInboundForward{ID: "Ev0001", Headers: map[string]string{"X-Slack-Signature": "v0=abc"}, HeadersUntil: &until}
	})

	claimed := s.claimedOne(s.base, "1759740000.000100")

	s.Equal(core.InboundMessage{
		ConnectorID: "acme", ProviderUnitID: "T0000TEAM", ThreadKey: "C0000CHAN:thread-one", AuthorID: "U0000AUTHOR",
		Text: "hello 1759740000.000100", Media: []core.Media{{URL: "https://files.example.com/a.png", MIMEType: "image/png"}},
		ProviderMessageID: "1759740000.000100", Raw: []byte(`{"event":"1759740000.000100"}`),
	}, claimed.Message())
	s.Equal("thread-one", claimed.ChannelID)
	s.Equal("acme-app", claimed.CustomerID)
	s.Equal("config-one", claimed.ConfigID)
	s.Equal("Ada", claimed.ConfigName)
	s.Equal(ChannelInboundQueued, claimed.State)
	s.Zero(claimed.Attempts)
	s.True(claimed.NextAttemptAt.Equal(s.base.Add(time.Minute)), "a claimed message is due at the end of its lease")
	s.Require().NotNil(claimed.Forward)
	s.Equal("Ev0001", claimed.Forward.ID)
	s.Equal(map[string]string{"X-Slack-Signature": "v0=abc"}, claimed.Forward.Headers)
	s.Require().NotNil(claimed.Forward.HeadersUntil)
	s.True(claimed.Forward.HeadersUntil.Equal(until))
}

// The row keys a file by its own names, not by core.Media's field names, which a rename would
// change under the rows queued before it.
func (s *StoreSuite) TestAnInboundMessagesMediaIsStoredUnderItsOwnKeys() {
	s.inbound("thread-one", "1759740000.000100", func(message *ChannelInbound) {
		message.Media = []core.Media{{URL: "https://files.example.com/a.png", MIMEType: "image/png"}}
	})

	var media string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT media::text FROM channel_thread_inbound WHERE channel_id = 'thread-one'").Scan(&media))

	s.JSONEq(`[{"url": "https://files.example.com/a.png", "mime_type": "image/png"}]`, media)
	s.Equal([]core.Media{{URL: "https://files.example.com/a.png", MIMEType: "image/png"}},
		s.claimedOne(s.base, "1759740000.000100").Message().Media)
}

func (s *StoreSuite) TestAnInboundMessageWithNoMediaAndNoForwardIsClaimedWithNeither() {
	s.inbound("thread-one", "1759740000.000100")

	claimed := s.claimedOne(s.base, "1759740000.000100")

	s.Empty(claimed.Media)
	s.Nil(claimed.Forward)
	var media int
	var forwarded bool
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT jsonb_array_length(media), forward IS NOT NULL FROM channel_thread_inbound WHERE channel_id = 'thread-one'").Scan(&media, &forwarded))
	s.Zero(media, "no media is an empty array, not JSON null")
	s.False(forwarded, "no forward is NULL, not JSON null")
}

func (s *StoreSuite) TestAnInboundMessageQueuedTwiceInOneChannelIsOneRow() {
	first := s.inbound("thread-one", "1759740000.000100")

	again, err := s.store.QueueChannelInbound(s.ctx, first)
	s.Require().NoError(err)
	s.False(again, "a provider's retry of a message queues nothing")
	s.inbound("thread-two", "1759740000.000100")

	claimed := s.claim(s.base)
	s.Len(claimed, 2, "the same provider id in another thread channel is another message")
}

func (s *StoreSuite) TestAWrittenInboundMessageIsNotQueuedAgain() {
	first := s.inbound("thread-one", "1759740000.000100")
	_, err := s.store.FinishChannelInbound(s.ctx, s.claimedOne(s.base, "1759740000.000100"), s.base)
	s.Require().NoError(err)

	again, err := s.store.QueueChannelInbound(s.ctx, first)

	s.Require().NoError(err)
	s.False(again)
	s.Empty(s.claim(s.base.Add(time.Hour)))
}

func (s *StoreSuite) TestAClaimedInboundMessageIsNotClaimedAgainUntilItsLeaseRunsOut() {
	s.inbound("thread-one", "1759740000.000100")
	s.claimedOne(s.base, "1759740000.000100")

	s.Empty(s.claim(s.base.Add(59*time.Second)), "another worker leaves it while the lease runs")
	s.claimedOne(s.base.Add(time.Minute), "1759740000.000100")
}

func (s *StoreSuite) TestAnInboundMessageIsNotClaimedBeforeItIsDue() {
	s.inbound("thread-one", "1759740000.000100", func(message *ChannelInbound) { message.NextAttemptAt = s.base.Add(time.Second) })

	s.Empty(s.claim(s.base))
	s.claimedOne(s.base.Add(time.Second), "1759740000.000100")
}

func (s *StoreSuite) TestAThreadsSecondMessageWaitsUntilItsFirstIsWritten() {
	s.inbound("thread-one", "1759740000.000100")
	s.inbound("thread-one", "1759740000.000200")
	s.inbound("thread-two", "1759740000.000300")

	claimed := s.claim(s.base)
	s.Require().Len(claimed, 2)
	taken := []string{claimed[0].ProviderMessageID, claimed[1].ProviderMessageID}
	s.ElementsMatch([]string{"1759740000.000100", "1759740000.000300"}, taken, "one message of a thread at a time, the oldest")
	first := claimed[0]
	if first.ProviderMessageID != "1759740000.000100" {
		first = claimed[1]
	}
	s.Empty(s.claim(s.base.Add(time.Second)))

	changed, err := s.store.FinishChannelInbound(s.ctx, first, s.base.Add(time.Second))
	s.Require().NoError(err)
	s.True(changed)
	s.claimedOne(s.base.Add(time.Second), "1759740000.000200")
}

// Message A of a thread is queued in a transaction that is still open when message B is queued:
// B waits for A's commit, so A, queued first, is written first and B only after it. Were B
// queued at once, it would be claimed first, and A, committed with the smaller seq while B is
// being written, would be claimed too: two of one thread's messages at once, out of order.
func (s *StoreSuite) TestAThreadsMessageQueuedWhileAnEarlierOneCommitsWaitsForIt() {
	a := s.inboundMessage("thread-one", "1759740000.000100")
	b := s.inboundMessage("thread-one", "1759740000.000200")
	tx, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer tx.Rollback() //nolint:errcheck // a no-op once committed
	queued, err := queueChannelInbound(s.ctx, tx, a)
	s.Require().NoError(err)
	s.Require().True(queued)
	byB := make(chan error, 1)
	go func() {
		_, err := s.store.QueueChannelInbound(s.ctx, b)
		byB <- err
	}()
	s.Require().Eventually(func() bool {
		var waiting int
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' AND datname = current_database() AND query LIKE '%FROM channel_threads%FOR NO KEY UPDATE%'").Scan(&waiting))
		return waiting == 1
	}, 5*time.Second, 10*time.Millisecond, "queueing B waits for A's transaction")
	s.Empty(s.claim(s.base), "neither is committed")

	s.Require().NoError(tx.Commit())
	s.Require().NoError(<-byB)

	first := s.claimedOne(s.base, "1759740000.000100")
	s.Empty(s.claim(s.base), "B waits while A is written")
	changed, err := s.store.FinishChannelInbound(s.ctx, first, s.base)
	s.Require().NoError(err)
	s.True(changed)
	s.claimedOne(s.base, "1759740000.000200")
}

func (s *StoreSuite) TestAThreadsMessagesAreTakenInTheOrderTheyWereQueuedNotByWhenTheyAreDue() {
	s.inbound("thread-one", "1759740000.000100", func(message *ChannelInbound) { message.NextAttemptAt = s.base.Add(time.Second) })
	s.inbound("thread-one", "1759740000.000200")

	s.Empty(s.claim(s.base), "the second is due, but the first is not yet")
	s.claimedOne(s.base.Add(time.Second), "1759740000.000100")
}

func (s *StoreSuite) TestAClaimTakesAtMostLimitOldestDueFirst() {
	s.inbound("thread-one", "1759740000.000100", func(message *ChannelInbound) { message.NextAttemptAt = s.base.Add(time.Second) })
	s.inbound("thread-two", "1759740000.000200")

	claimed, err := s.store.ClaimChannelInbound(s.ctx, s.base.Add(time.Second), 1, s.base.Add(time.Minute))
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Equal("1759740000.000200", claimed[0].ProviderMessageID)

	for _, limit := range []int{0, -1} {
		none, err := s.store.ClaimChannelInbound(s.ctx, s.base.Add(time.Second), limit, s.base.Add(time.Minute))
		s.Require().NoError(err, "limit %d", limit)
		s.Empty(none, "limit %d", limit)
	}
}

func (s *StoreSuite) TestARetriedInboundMessageHoldsItsThreadUntilItsNextAttempt() {
	s.inbound("thread-one", "1759740000.000100")
	s.inbound("thread-one", "1759740000.000200")
	first := s.claimedOne(s.base, "1759740000.000100")

	changed, err := s.store.RetryChannelInbound(s.ctx, first, s.base.Add(5*time.Second))
	s.Require().NoError(err)
	s.True(changed)

	s.Empty(s.claim(s.base.Add(4*time.Second)), "the second waits for the first's retry")
	again := s.claimedOne(s.base.Add(5*time.Second), "1759740000.000100")
	s.Equal(1, again.Attempts)
}

func (s *StoreSuite) TestAFailedInboundMessageLetsItsThreadGoOn() {
	s.inbound("thread-one", "1759740000.000100")
	s.inbound("thread-one", "1759740000.000200")
	first := s.claimedOne(s.base, "1759740000.000100")
	_, err := s.store.RetryChannelInbound(s.ctx, first, s.base.Add(time.Second))
	s.Require().NoError(err)
	last := s.claimedOne(s.base.Add(time.Second), "1759740000.000100")

	changed, _, err := s.store.FailChannelInbound(s.ctx, last, s.base.Add(2*time.Second))
	s.Require().NoError(err)
	s.True(changed)

	state, attempts := s.inboundState("thread-one", "1759740000.000100")
	s.Equal(ChannelInboundFailed, state)
	s.Equal(2, attempts, "the attempt that gave it up counts")
	s.claimedOne(s.base.Add(2*time.Second), "1759740000.000200")
}

func (s *StoreSuite) TestAWrittenInboundMessageKeepsItsAttempts() {
	s.inbound("thread-one", "1759740000.000100")
	first := s.claimedOne(s.base, "1759740000.000100")
	_, err := s.store.RetryChannelInbound(s.ctx, first, s.base.Add(time.Second))
	s.Require().NoError(err)

	_, err = s.store.FinishChannelInbound(s.ctx, s.claimedOne(s.base.Add(time.Second), "1759740000.000100"), s.base.Add(2*time.Second))
	s.Require().NoError(err)

	state, attempts := s.inboundState("thread-one", "1759740000.000100")
	s.Equal(ChannelInboundDone, state)
	s.Equal(1, attempts)
}

// A worker whose lease ran out while it wrote, and whose message another worker took since,
// must not finish, retry or give up the other worker's attempt.
func (s *StoreSuite) TestAWorkerWhoseLeaseWasTakenOverChangesNothing() {
	s.inbound("thread-one", "1759740000.000100")
	stale := s.claimedOne(s.base, "1759740000.000100")
	current := s.claimedOne(s.base.Add(time.Minute), "1759740000.000100")

	retried, err := s.store.RetryChannelInbound(s.ctx, stale, s.base.Add(time.Minute))
	s.Require().NoError(err)
	failed, _, err := s.store.FailChannelInbound(s.ctx, stale, s.base.Add(time.Minute))
	s.Require().NoError(err)
	finished, err := s.store.FinishChannelInbound(s.ctx, stale, s.base.Add(time.Minute))
	s.Require().NoError(err)

	s.False(retried)
	s.False(failed)
	s.False(finished)
	state, attempts := s.inboundState("thread-one", "1759740000.000100")
	s.Equal(ChannelInboundQueued, state)
	s.Zero(attempts)
	s.Empty(s.claim(s.base.Add(90*time.Second)), "the current worker's lease still holds it")
	changed, err := s.store.FinishChannelInbound(s.ctx, current, s.base.Add(90*time.Second))
	s.Require().NoError(err)
	s.True(changed)
}

func (s *StoreSuite) TestAFinishedInboundMessageIsNotFinishedAgain() {
	s.inbound("thread-one", "1759740000.000100")
	claimed := s.claimedOne(s.base, "1759740000.000100")
	_, err := s.store.FinishChannelInbound(s.ctx, claimed, s.base)
	s.Require().NoError(err)

	failed, _, err := s.store.FailChannelInbound(s.ctx, claimed, s.base)

	s.Require().NoError(err)
	s.False(failed)
	state, _ := s.inboundState("thread-one", "1759740000.000100")
	s.Equal(ChannelInboundDone, state)
}

// pausedInboundClaim is ClaimChannelInbound's statement with a pause after head read the rows,
// so another router's claim commits inside it every run.
func (s *StoreSuite) pausedInboundClaim() string {
	head := "    WHERE cti.state = 'queued'\n    ORDER BY cti.channel_id, cti.seq\n), due AS ("
	s.Require().Contains(claimChannelInboundQuery, head)
	return strings.Replace(claimChannelInboundQuery, head,
		"    WHERE cti.state = 'queued' AND (SELECT 1 FROM pg_sleep(0.5)) = 1\n    ORDER BY cti.channel_id, cti.seq\n), due AS (", 1)
}

// claimWhileAnotherRouterActs runs router A's paused claim at at and, while A is in its pause,
// meanwhile on this suite's store, and returns what A took.
func (s *StoreSuite) claimWhileAnotherRouterActs(at time.Time, meanwhile func()) []ChannelInbound {
	routerA, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(routerA.Close()) })
	type claim struct {
		claimed []ChannelInbound
		err     error
	}
	byA := make(chan claim, 1)
	go func() {
		claimed, err := routerA.claimChannelInbound(s.ctx, s.pausedInboundClaim(), at, 10, at.Add(time.Minute))
		byA <- claim{claimed, err}
	}()
	s.Require().Eventually(func() bool {
		var sleeping int
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT count(*) FROM pg_stat_activity WHERE wait_event = 'PgSleep' AND datname = current_database()").Scan(&sleeping))
		return sleeping == 1
	}, 5*time.Second, 10*time.Millisecond, "router A's claim is in its pause")
	meanwhile()
	a := <-byA
	s.Require().NoError(a.err)
	return a.claimed
}

func (s *StoreSuite) TestTwoRoutersClaimingAtOnceTakeEachInboundMessageOnce() {
	s.inbound("thread-one", "1759740000.000100")
	s.inbound("thread-two", "1759740000.000200")
	var byB []ChannelInbound

	byA := s.claimWhileAnotherRouterActs(s.base, func() { byB = s.claim(s.base) })

	s.Len(byB, 2)
	s.Empty(byA, "router A takes none of what router B leased")
}

// Router B claims a thread's message and finishes it while router A's claim, which read it as
// queued, is paused: A must not take the written message again, even once B's lease is over.
func (s *StoreSuite) TestARouterDoesNotClaimAnInboundMessageAnotherFinishedDuringItsClaim() {
	s.inbound("thread-one", "1759740000.000100")

	byA := s.claimWhileAnotherRouterActs(s.base.Add(2*time.Minute), func() {
		_, err := s.store.FinishChannelInbound(s.ctx, s.claimedOne(s.base, "1759740000.000100"), s.base)
		s.Require().NoError(err)
	})

	s.Empty(byA)
}

// A row another router holds locked, mid-claim or mid-finish, is skipped: the claim takes the
// other threads' messages at once rather than waiting for that router's commit.
func (s *StoreSuite) TestAClaimSkipsAnInboundMessageAnotherRouterHoldsLocked() {
	s.inbound("thread-one", "1759740000.000100")
	s.inbound("thread-two", "1759740000.000200")
	held, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer held.Rollback() //nolint:errcheck // only releases the lock
	_, err = held.ExecContext(s.ctx, "SELECT 1 FROM channel_thread_inbound WHERE channel_id = 'thread-one' FOR UPDATE")
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(s.ctx, 2*time.Second)
	defer cancel()
	claimed, err := s.store.ClaimChannelInbound(ctx, s.base, 10, s.base.Add(time.Minute))

	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Equal("1759740000.000200", claimed[0].ProviderMessageID)
}

func (s *StoreSuite) TestTheNextInboundMessageIsTheOneAClaimMayTakeFirst() {
	_, found, err := s.store.NextChannelInboundAt(s.ctx)
	s.Require().NoError(err)
	s.False(found, "none when nothing is queued")

	s.inbound("thread-one", "1759740000.000100")
	s.inbound("thread-one", "1759740000.000200")
	s.inbound("thread-two", "1759740000.000300", func(message *ChannelInbound) { message.NextAttemptAt = s.base.Add(time.Hour) })
	s.claimedOne(s.base, "1759740000.000100")

	next, found, err := s.store.NextChannelInboundAt(s.ctx)
	s.Require().NoError(err)
	s.True(found)
	s.True(next.Equal(s.base.Add(time.Minute)), "the second message of thread-one waits for the first's lease, not due at %s", next)
}

func (s *StoreSuite) TestNoInboundMessageIsNextOnceAllAreWrittenOrGivenUp() {
	s.inbound("thread-one", "1759740000.000100")
	s.inbound("thread-two", "1759740000.000200")
	for _, claimed := range s.claim(s.base) {
		if claimed.ChannelID == "thread-one" {
			_, err := s.store.FinishChannelInbound(s.ctx, claimed, s.base)
			s.Require().NoError(err)
			continue
		}
		_, _, err := s.store.FailChannelInbound(s.ctx, claimed, s.base)
		s.Require().NoError(err)
	}

	_, found, err := s.store.NextChannelInboundAt(s.ctx)

	s.Require().NoError(err)
	s.False(found)
}

func (s *StoreSuite) TestPruningKeepsAWrittenMessageADayAndAGivenUpOneThirtyDays() {
	for _, id := range []string{"done-old", "done-new", "failed-old", "failed-new", "queued"} {
		s.inbound("thread-"+id, id)
	}
	for _, claimed := range s.claim(s.base) {
		at := s.base
		switch claimed.ProviderMessageID {
		case "done-old":
			at = s.base.Add(-24*time.Hour - time.Second)
		case "failed-old":
			at = s.base.Add(-30*24*time.Hour - time.Second)
		case "failed-new":
			at = s.base.Add(-30*24*time.Hour + time.Second)
		case "done-new":
			at = s.base.Add(-24*time.Hour + time.Second)
		case "queued":
			continue
		}
		var err error
		if strings.HasPrefix(claimed.ProviderMessageID, "done") {
			_, err = s.store.FinishChannelInbound(s.ctx, claimed, at)
		} else {
			_, _, err = s.store.FailChannelInbound(s.ctx, claimed, at)
		}
		s.Require().NoError(err)
	}

	pruned, err := s.store.PruneChannelInbound(s.ctx, s.base)

	s.Require().NoError(err)
	s.Equal(2, pruned)
	var kept []string
	rows, err := s.store.DB().QueryContext(s.ctx, "SELECT provider_message_id FROM channel_thread_inbound ORDER BY provider_message_id")
	s.Require().NoError(err)
	defer rows.Close() //nolint:errcheck // read to the end below
	for rows.Next() {
		var id string
		s.Require().NoError(rows.Scan(&id))
		kept = append(kept, id)
	}
	s.Require().NoError(rows.Err())
	s.Equal([]string{"done-new", "failed-new", "queued"}, kept)
}

func (s *StoreSuite) TestAThreadChannelsInboundMessagesGoWithIt() {
	s.inbound("thread-one", "1759740000.000100")

	_, err := s.store.DB().ExecContext(s.ctx, "DELETE FROM channel_threads WHERE channel_id = 'thread-one'")
	s.Require().NoError(err)

	s.Empty(s.claim(s.base))
}

// givenUp queues a message in a thread channel of its own, claims it and gives it up, and
// returns whether its delivery is to be forwarded.
func (s *StoreSuite) givenUp(channelID string, change ...func(*ChannelInbound)) bool {
	s.inbound(channelID, "1759740000.000100", change...)
	changed, forward, err := s.store.FailChannelInbound(s.ctx, s.claimedOne(s.base, "1759740000.000100"), s.base)
	s.Require().NoError(err)
	s.Require().True(changed)
	return forward
}

// forwarding gives a message the delivery forward id, and the body raw.
func forwarding(id, raw string) func(*ChannelInbound) {
	return func(message *ChannelInbound) {
		message.Forward = &ChannelInboundForward{ID: id}
		message.Raw = []byte(raw)
	}
}

func (s *StoreSuite) TestOnlyTheFirstGivenUpMessageOfADeliveryForwardsIt() {
	s.True(s.givenUp("thread-one", forwarding("Ev0001", `{"one":1}`)))
	s.False(s.givenUp("thread-two", forwarding("Ev0001", `{"two":2}`)), "the provider's id keys the delivery")
	s.True(s.givenUp("thread-three", forwarding("Ev0002", `{"one":1}`)), "another delivery")
}

func (s *StoreSuite) TestADeliveryWithNoIdIsKeyedByItsBody() {
	s.True(s.givenUp("thread-one", forwarding("", `{"batch":1}`)))
	s.False(s.givenUp("thread-two", forwarding("", `{"batch":1}`)), "the same body")
	s.True(s.givenUp("thread-three", forwarding("", `{"batch":2}`)), "another body")
	s.True(s.givenUp("thread-four", forwarding("Ev0001", `{"batch":1}`)), "a body with an id is another delivery")
}

func (s *StoreSuite) TestAnotherCustomersOrConnectorsDeliveryIsForwardedOnItsOwn() {
	s.True(s.givenUp("thread-one", forwarding("Ev0001", `{}`)))
	s.True(s.givenUp("thread-two", forwarding("Ev0001", `{}`), func(message *ChannelInbound) { message.CustomerID = "globex-app" }))
	s.True(s.givenUp("thread-three", forwarding("Ev0001", `{}`), func(message *ChannelInbound) { message.ConnectorID = "other" }))
}

func (s *StoreSuite) TestAGivenUpMessageWithNoForwardForwardsNothing() {
	s.True(s.givenUp("thread-one", forwarding("Ev0001", `{}`)))
	s.False(s.givenUp("thread-two"))
}

// A message whose earlier attempt finished it is not given up, so it forwards nothing.
func (s *StoreSuite) TestAWorkerWhoseLeaseWasTakenOverForwardsNothing() {
	s.inbound("thread-one", "1759740000.000100", forwarding("Ev0001", `{}`))
	stale := s.claimedOne(s.base, "1759740000.000100")
	s.claimedOne(s.base.Add(time.Minute), "1759740000.000100")

	changed, forward, err := s.store.FailChannelInbound(s.ctx, stale, s.base.Add(time.Minute))

	s.Require().NoError(err)
	s.False(changed)
	s.False(forward)
}

// Two messages of one delivery given up at once, as two routers do, forward it once.
func (s *StoreSuite) TestTwoMessagesOfADeliveryGivenUpAtOnceForwardItOnce() {
	for round := range 20 {
		var claimed []ChannelInbound
		for _, channel := range []string{"thread-a", "thread-b"} {
			id := fmt.Sprintf("1759740000.%06d", round)
			s.inbound(channel, id, forwarding(fmt.Sprintf("Ev%04d", round), `{}`))
			claimed = append(claimed, s.claimedOne(s.base, id))
		}
		forwards := make(chan bool, len(claimed))
		var wg sync.WaitGroup
		for _, message := range claimed {
			wg.Go(func() {
				_, forward, err := s.store.FailChannelInbound(s.ctx, message, s.base)
				s.NoError(err)
				forwards <- forward
			})
		}
		wg.Wait()
		close(forwards)
		count := 0
		for forward := range forwards {
			if forward {
				count++
			}
		}
		s.Equal(1, count, "round %d", round)
	}
}
