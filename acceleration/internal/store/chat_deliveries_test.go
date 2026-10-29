//go:build integration

package store

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/uptrace/bun"
)

// chatLease is the lease the tests claim under: longer than an answer may take.
const chatLease = 3 * time.Minute

// queuedChat queues a message of the app 1257545, due at s.base; change sets the fields a test
// cares about.
func (s *StoreSuite) queuedChat(messageID string, change ...func(*ChatDelivery)) ChatDelivery {
	delivery := s.chatDelivery(messageID, change...)
	queued, err := s.store.QueueChatDelivery(s.ctx, delivery)
	s.Require().NoError(err)
	s.Require().True(queued)
	return delivery
}

// chatDelivery is the message queuedChat queues, not yet queued.
func (s *StoreSuite) chatDelivery(messageID string, change ...func(*ChatDelivery)) ChatDelivery {
	delivery := ChatDelivery{
		StreamApp: 1257545, MessageID: messageID, CustomerID: "acme-app", APIKey: "key-one",
		Body: []byte(`{"type":"message.new","message":{"id":"` + messageID + `"}}`), NextAttemptAt: s.base,
	}
	for _, apply := range change {
		apply(&delivery)
	}
	return delivery
}

// claimChats claims every message due at at, under chatLease.
func (s *StoreSuite) claimChats(at time.Time) []ChatDelivery {
	claimed, err := s.store.ClaimChatDeliveries(s.ctx, at, 10, at.Add(chatLease))
	s.Require().NoError(err)
	return claimed
}

// claimedChat claims at at and requires the one message it takes to be messageID.
func (s *StoreSuite) claimedChat(at time.Time, messageID string) ChatDelivery {
	claimed := s.claimChats(at)
	s.Require().Len(claimed, 1)
	s.Require().Equal(messageID, claimed[0].MessageID)
	return claimed[0]
}

// chatRow is the state, attempts and finish time of a message's row in the app 1257545.
func (s *StoreSuite) chatRow(messageID string) (state string, attempts int, finishedAt *time.Time) {
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT state, attempts, finished_at FROM chat_deliveries WHERE stream_app = 1257545 AND message_id = ?",
		messageID).Scan(&state, &attempts, &finishedAt))
	return state, attempts, finishedAt
}

// chatRows is how many rows the table holds.
func (s *StoreSuite) chatRows() int {
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM chat_deliveries").Scan(&rows))
	return rows
}

func (s *StoreSuite) TestAClaimedChatDeliveryHasEverythingItWasQueuedWith() {
	s.queuedChat("msg-one", func(delivery *ChatDelivery) { delivery.Deployment = true })

	claimed := s.claimedChat(s.base, "msg-one")

	s.NotZero(claimed.Seq)
	s.Equal(int64(1257545), claimed.StreamApp)
	s.Equal("acme-app", claimed.CustomerID)
	s.True(claimed.Deployment)
	s.Equal("key-one", claimed.APIKey)
	s.Equal(`{"type":"message.new","message":{"id":"msg-one"}}`, string(claimed.Body))
	s.Equal(ChatDeliveryQueued, claimed.State)
	s.Zero(claimed.Attempts)
	s.True(claimed.NextAttemptAt.Equal(s.base.Add(chatLease)), "a claimed message is due at the end of the lease its claim gave")
	s.WithinDuration(time.Now(), claimed.ReceivedAt, time.Minute)
	s.Nil(claimed.FinishedAt)
}

func (s *StoreSuite) TestAChatDeliveryQueuedTwiceIsOneRow() {
	s.queuedChat("msg-one")

	queued, err := s.store.QueueChatDelivery(s.ctx, s.chatDelivery("msg-one", func(delivery *ChatDelivery) {
		delivery.Body = []byte(`{"again":true}`)
	}))

	s.Require().NoError(err)
	s.False(queued)
	s.Equal(1, s.chatRows())
	s.Equal(`{"type":"message.new","message":{"id":"msg-one"}}`, string(s.claimedChat(s.base, "msg-one").Body), "the first delivery's row stays")
}

// Stream's message ids are unique within an app, so another app's message of the same id is a
// message of its own.
func (s *StoreSuite) TestAChatMessageOfTheSameIDInAnotherAppIsQueuedOnItsOwn() {
	s.queuedChat("msg-one")

	queued, err := s.store.QueueChatDelivery(s.ctx, s.chatDelivery("msg-one", func(delivery *ChatDelivery) { delivery.StreamApp = 1728173 }))

	s.Require().NoError(err)
	s.True(queued)
	s.Len(s.claimChats(s.base), 2)
}

func (s *StoreSuite) TestAnAnsweredChatMessageIsNotQueuedAgain() {
	s.queuedChat("msg-one")
	_, err := s.store.FinishChatDelivery(s.ctx, s.claimedChat(s.base, "msg-one"), s.base)
	s.Require().NoError(err)

	queued, err := s.store.QueueChatDelivery(s.ctx, s.chatDelivery("msg-one"))

	s.Require().NoError(err)
	s.False(queued)
	state, _, _ := s.chatRow("msg-one")
	s.Equal(ChatDeliveryDone, state)
}

// queueChatDelivery takes the caller's transaction: what it queues there is queued only when
// the transaction commits, and a second queue in it is still one row.
func (s *StoreSuite) TestAChatDeliveryQueuedInACallersTransactionIsQueuedOnlyWithIt() {
	rolledBack := errors.New("the caller's next step failed")
	err := s.store.DB().RunInTx(s.ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		queued, err := queueChatDelivery(ctx, tx, s.chatDelivery("msg-one"))
		s.Require().NoError(err)
		s.Require().True(queued)
		return rolledBack
	})
	s.Require().ErrorIs(err, rolledBack)
	s.Zero(s.chatRows(), "a rolled back transaction queues nothing")

	err = s.store.DB().RunInTx(s.ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		first, err := queueChatDelivery(ctx, tx, s.chatDelivery("msg-one"))
		s.Require().NoError(err)
		second, err := queueChatDelivery(ctx, tx, s.chatDelivery("msg-one"))
		s.Require().NoError(err)
		s.True(first)
		s.False(second)
		return nil
	})

	s.Require().NoError(err)
	s.Equal(1, s.chatRows())
}

func (s *StoreSuite) TestAChatMessageIsNotClaimedBeforeItIsDue() {
	s.queuedChat("msg-one", func(delivery *ChatDelivery) { delivery.NextAttemptAt = s.base.Add(time.Second) })

	s.Empty(s.claimChats(s.base))
	s.claimedChat(s.base.Add(time.Second), "msg-one")
}

// A worker that stopped mid-answer leaves its row leased: another takes it once the lease the
// claim gave runs out, and not before.
func (s *StoreSuite) TestAClaimedChatMessageIsClaimedAgainOnceItsLeaseRunsOut() {
	s.queuedChat("msg-one")
	s.claimedChat(s.base, "msg-one")

	s.Empty(s.claimChats(s.base.Add(chatLease-time.Second)), "still leased")
	again := s.claimedChat(s.base.Add(chatLease), "msg-one")

	s.Zero(again.Attempts, "a lease that ran out is no failed attempt")
	s.True(again.NextAttemptAt.Equal(s.base.Add(2 * chatLease)))
}

func (s *StoreSuite) TestAChatClaimTakesAtMostLimitOldestDueFirst() {
	s.queuedChat("due-last", func(delivery *ChatDelivery) { delivery.NextAttemptAt = s.base.Add(-time.Second) })
	s.queuedChat("due-first", func(delivery *ChatDelivery) { delivery.NextAttemptAt = s.base.Add(-time.Minute) })
	s.queuedChat("due-first-too", func(delivery *ChatDelivery) { delivery.NextAttemptAt = s.base.Add(-time.Minute) })

	claimed, err := s.store.ClaimChatDeliveries(s.ctx, s.base, 2, s.base.Add(chatLease))

	s.Require().NoError(err)
	s.Require().Len(claimed, 2)
	s.ElementsMatch([]string{"due-first", "due-first-too"}, []string{claimed[0].MessageID, claimed[1].MessageID})
	s.claimedChat(s.base, "due-last")
}

func (s *StoreSuite) TestAChatClaimTakesMessagesDueAtOnceInTheOrderTheyWereQueued() {
	for _, id := range []string{"msg-one", "msg-two", "msg-three"} {
		s.queuedChat(id)
	}

	for _, id := range []string{"msg-one", "msg-two", "msg-three"} {
		claimed, err := s.store.ClaimChatDeliveries(s.ctx, s.base, 1, s.base.Add(chatLease))
		s.Require().NoError(err)
		s.Require().Len(claimed, 1)
		s.Equal(id, claimed[0].MessageID)
	}
}

func (s *StoreSuite) TestAChatClaimWithNoRoomTakesNothing() {
	s.queuedChat("msg-one")

	claimed, err := s.store.ClaimChatDeliveries(s.ctx, s.base, -1, s.base.Add(chatLease))

	s.Require().NoError(err)
	s.Empty(claimed)
	s.claimedChat(s.base, "msg-one")
}

func (s *StoreSuite) TestARetriedChatMessageIsDueAtItsNextAttempt() {
	s.queuedChat("msg-one")
	claimed := s.claimedChat(s.base, "msg-one")

	retried, err := s.store.RetryChatDelivery(s.ctx, claimed, s.base.Add(10*time.Second))

	s.Require().NoError(err)
	s.True(retried)
	s.Empty(s.claimChats(s.base.Add(9 * time.Second)))
	again := s.claimedChat(s.base.Add(10*time.Second), "msg-one")
	s.Equal(1, again.Attempts)
	s.Equal(ChatDeliveryQueued, again.State)
}

func (s *StoreSuite) TestAnAnsweredChatMessageIsDoneWithItsAttempts() {
	s.queuedChat("msg-one")
	_, err := s.store.RetryChatDelivery(s.ctx, s.claimedChat(s.base, "msg-one"), s.base.Add(time.Second))
	s.Require().NoError(err)
	claimed := s.claimedChat(s.base.Add(time.Second), "msg-one")

	finished, err := s.store.FinishChatDelivery(s.ctx, claimed, s.base.Add(5*time.Second))

	s.Require().NoError(err)
	s.True(finished)
	state, attempts, finishedAt := s.chatRow("msg-one")
	s.Equal(ChatDeliveryDone, state)
	s.Equal(1, attempts, "only the failed attempt counts")
	s.Require().NotNil(finishedAt)
	s.True(finishedAt.Equal(s.base.Add(5 * time.Second)))
	s.Empty(s.claimChats(s.base.Add(time.Hour)))
}

func (s *StoreSuite) TestAGivenUpChatMessageIsFailedWithItsLastAttemptCounted() {
	s.queuedChat("msg-one")
	claimed := s.claimedChat(s.base, "msg-one")

	failed, err := s.store.FailChatDelivery(s.ctx, claimed, s.base.Add(5*time.Second))

	s.Require().NoError(err)
	s.True(failed)
	state, attempts, finishedAt := s.chatRow("msg-one")
	s.Equal(ChatDeliveryFailed, state)
	s.Equal(1, attempts)
	s.Require().NotNil(finishedAt)
	s.True(finishedAt.Equal(s.base.Add(5 * time.Second)))
	s.Empty(s.claimChats(s.base.Add(time.Hour)))
}

// A worker whose lease ran out while it answered, and whose message another worker took since,
// must not finish, retry or give up the other worker's attempt.
func (s *StoreSuite) TestAWorkerWhoseChatLeaseWasTakenOverChangesNothing() {
	s.queuedChat("msg-one")
	stale := s.claimedChat(s.base, "msg-one")
	current := s.claimedChat(s.base.Add(chatLease), "msg-one")

	retried, err := s.store.RetryChatDelivery(s.ctx, stale, s.base.Add(chatLease))
	s.Require().NoError(err)
	failed, err := s.store.FailChatDelivery(s.ctx, stale, s.base.Add(chatLease))
	s.Require().NoError(err)
	finished, err := s.store.FinishChatDelivery(s.ctx, stale, s.base.Add(chatLease))
	s.Require().NoError(err)

	s.False(retried)
	s.False(failed)
	s.False(finished)
	state, attempts, _ := s.chatRow("msg-one")
	s.Equal(ChatDeliveryQueued, state)
	s.Zero(attempts)
	s.Empty(s.claimChats(s.base.Add(2*chatLease-time.Second)), "the current worker's lease still holds it")
	changed, err := s.store.FinishChatDelivery(s.ctx, current, s.base.Add(chatLease))
	s.Require().NoError(err)
	s.True(changed)
}

func (s *StoreSuite) TestAnAnsweredChatMessageIsNotGivenUpOrRetriedAfter() {
	s.queuedChat("msg-one")
	claimed := s.claimedChat(s.base, "msg-one")
	_, err := s.store.FinishChatDelivery(s.ctx, claimed, s.base)
	s.Require().NoError(err)

	failed, err := s.store.FailChatDelivery(s.ctx, claimed, s.base)
	s.Require().NoError(err)
	retried, err := s.store.RetryChatDelivery(s.ctx, claimed, s.base)
	s.Require().NoError(err)

	s.False(failed)
	s.False(retried)
	state, attempts, _ := s.chatRow("msg-one")
	s.Equal(ChatDeliveryDone, state)
	s.Zero(attempts)
}

// pausedChatClaim is ClaimChatDeliveries' statement with a pause before it locks a row, so
// another router's claim commits inside it every run.
func (s *StoreSuite) pausedChatClaim() string {
	due := "    WHERE cd.state = 'queued' AND cd.next_attempt_at <= ?\n"
	s.Require().Contains(claimChatDeliveriesQuery, due)
	return strings.Replace(claimChatDeliveriesQuery, due,
		"    WHERE cd.state = 'queued' AND cd.next_attempt_at <= ? AND (SELECT 1 FROM pg_sleep(0.5)) = 1\n", 1)
}

// claimChatsWhileAnotherRouterActs runs router A's paused claim at at and, while A is in its
// pause, meanwhile on this suite's store, and returns what A took.
func (s *StoreSuite) claimChatsWhileAnotherRouterActs(at time.Time, meanwhile func()) []ChatDelivery {
	routerA, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(routerA.Close()) })
	type claim struct {
		claimed []ChatDelivery
		err     error
	}
	byA := make(chan claim, 1)
	go func() {
		claimed, err := routerA.claimChatDeliveries(s.ctx, s.pausedChatClaim(), at, 10, at.Add(chatLease))
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

// Router B leases both messages while router A's claim, which read them as due, is paused: A
// checks each row again as it locks it and takes none of what B leased.
func (s *StoreSuite) TestTwoRoutersClaimingChatMessagesAtOnceTakeEachOnce() {
	s.queuedChat("msg-one")
	s.queuedChat("msg-two")
	var byB []ChatDelivery

	byA := s.claimChatsWhileAnotherRouterActs(s.base, func() { byB = s.claimChats(s.base) })

	s.Len(byB, 2)
	s.Empty(byA, "router A takes none of what router B leased")
}

// Router B claims a message and answers it while router A's claim, which read it as queued and
// due once B's lease is over, is paused: A must not take the answered message again.
func (s *StoreSuite) TestARouterDoesNotClaimAChatMessageAnotherAnsweredDuringItsClaim() {
	s.queuedChat("msg-one")

	byA := s.claimChatsWhileAnotherRouterActs(s.base.Add(2*chatLease), func() {
		_, err := s.store.FinishChatDelivery(s.ctx, s.claimedChat(s.base, "msg-one"), s.base)
		s.Require().NoError(err)
	})

	s.Empty(byA)
}

// A row another router holds locked, mid-claim or mid-finish, is skipped: the claim takes the
// other messages at once rather than waiting for that router's commit.
func (s *StoreSuite) TestAChatClaimSkipsAMessageAnotherRouterHoldsLocked() {
	s.queuedChat("msg-one")
	s.queuedChat("msg-two")
	held, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer held.Rollback() //nolint:errcheck // only releases the lock
	_, err = held.ExecContext(s.ctx, "SELECT 1 FROM chat_deliveries WHERE message_id = 'msg-one' FOR UPDATE")
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(s.ctx, 2*time.Second)
	defer cancel()
	claimed, err := s.store.ClaimChatDeliveries(ctx, s.base, 10, s.base.Add(chatLease))

	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Equal("msg-two", claimed[0].MessageID)
}

// Two routers claiming over and over at the same moment, one row at a time, take each message
// exactly once between them.
func (s *StoreSuite) TestRoutersClaimingChatMessagesConcurrentlyTakeEachExactlyOnce() {
	const messages = 40
	for i := range messages {
		s.queuedChat(fmt.Sprintf("msg-%02d", i))
	}
	routerA, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(routerA.Close()) })

	var mu sync.Mutex
	taken := map[string]int{}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for _, router := range []*Store{s.store, routerA} {
		wg.Go(func() {
			// Bounded, so a claim that takes a message again fails the test rather than
			// spinning on it.
			for range 2 * messages {
				claimed, err := router.ClaimChatDeliveries(s.ctx, s.base, 1, s.base.Add(chatLease))
				if err != nil {
					errs <- err
					return
				}
				if len(claimed) == 0 {
					return
				}
				mu.Lock()
				taken[claimed[0].MessageID]++
				mu.Unlock()
			}
		})
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		s.Require().NoError(err)
	}
	s.Len(taken, messages)
	for id, times := range taken {
		s.Equal(1, times, "%s taken %d times", id, times)
	}
}

func (s *StoreSuite) TestTheNextChatMessageIsTheQueuedOneDueFirst() {
	_, found, err := s.store.NextChatDeliveryAt(s.ctx)
	s.Require().NoError(err)
	s.False(found, "none when nothing is queued")

	s.queuedChat("msg-one")
	s.queuedChat("msg-two", func(delivery *ChatDelivery) { delivery.NextAttemptAt = s.base.Add(time.Hour) })
	s.queuedChat("msg-three", func(delivery *ChatDelivery) { delivery.NextAttemptAt = s.base.Add(-time.Hour) })
	_, err = s.store.FinishChatDelivery(s.ctx, s.claimedChat(s.base.Add(-time.Hour), "msg-three"), s.base)
	s.Require().NoError(err)
	s.claimedChat(s.base, "msg-one")

	next, found, err := s.store.NextChatDeliveryAt(s.ctx)

	s.Require().NoError(err)
	s.True(found)
	s.True(next.Equal(s.base.Add(chatLease)), "msg-one is due at the end of its lease, not an answered row's time: %s", next)
}

func (s *StoreSuite) TestNoChatMessageIsNextOnceAllAreAnsweredOrGivenUp() {
	s.queuedChat("msg-one")
	s.queuedChat("msg-two")
	for _, claimed := range s.claimChats(s.base) {
		var err error
		if claimed.MessageID == "msg-one" {
			_, err = s.store.FinishChatDelivery(s.ctx, claimed, s.base)
		} else {
			_, err = s.store.FailChatDelivery(s.ctx, claimed, s.base)
		}
		s.Require().NoError(err)
	}

	_, found, err := s.store.NextChatDeliveryAt(s.ctx)

	s.Require().NoError(err)
	s.False(found)
}

func (s *StoreSuite) TestPruningKeepsAnAnsweredChatMessageADayAndAGivenUpOneThirtyDays() {
	for _, id := range []string{"done-old", "done-new", "failed-old", "failed-new"} {
		s.queuedChat(id)
	}
	s.queuedChat("queued", func(delivery *ChatDelivery) { delivery.NextAttemptAt = s.base.Add(-60 * 24 * time.Hour) })
	finishedAt := map[string]time.Time{
		"done-old":   s.base.Add(-24*time.Hour - time.Second),
		"done-new":   s.base.Add(-24*time.Hour + time.Second),
		"failed-old": s.base.Add(-30*24*time.Hour - time.Second),
		"failed-new": s.base.Add(-30*24*time.Hour + time.Second),
	}
	claimed, err := s.store.ClaimChatDeliveries(s.ctx, s.base.Add(-time.Hour), 10, s.base.Add(chatLease))
	s.Require().NoError(err)
	s.Require().Len(claimed, 1, "only the long-queued row is due an hour before base")
	_, err = s.store.RetryChatDelivery(s.ctx, claimed[0], s.base.Add(time.Hour))
	s.Require().NoError(err)
	for _, claimed := range s.claimChats(s.base) {
		at := finishedAt[claimed.MessageID]
		if strings.HasPrefix(claimed.MessageID, "done") {
			_, err = s.store.FinishChatDelivery(s.ctx, claimed, at)
		} else {
			_, err = s.store.FailChatDelivery(s.ctx, claimed, at)
		}
		s.Require().NoError(err)
	}

	pruned, err := s.store.PruneChatDeliveries(s.ctx, s.base)

	s.Require().NoError(err)
	s.Equal(2, pruned)
	var kept []string
	rows, err := s.store.DB().QueryContext(s.ctx, "SELECT message_id FROM chat_deliveries ORDER BY message_id")
	s.Require().NoError(err)
	defer rows.Close() //nolint:errcheck // read to the end below
	for rows.Next() {
		var id string
		s.Require().NoError(rows.Scan(&id))
		kept = append(kept, id)
	}
	s.Require().NoError(rows.Err())
	s.Equal([]string{"done-new", "failed-new", "queued"}, kept, "a queued row is never pruned, however old")
}

// A message's answer starts once: the row leaves the queue as it starts, so no later try, nor
// a later claim, answers it again, and the worker holding the lease still finishes it.
func (s *StoreSuite) TestAChatMessagesAnswerStartsOnce() {
	s.queuedChat("msg-one")
	claimed := s.claimedChat(s.base, "msg-one")

	started, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base.Add(time.Second))
	s.Require().NoError(err)
	again, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base.Add(2*time.Second))
	s.Require().NoError(err)

	s.True(started)
	s.False(again, "an answer that started is not started again")
	s.Empty(s.claimChats(s.base.Add(time.Hour)), "a message whose answer started is not claimed again")
	finished, err := s.store.FinishChatDelivery(s.ctx, claimed, s.base.Add(5*time.Second))
	s.Require().NoError(err)
	s.True(finished, "the worker holding the lease records its attempt")
	state, attempts, finishedAt := s.chatRow("msg-one")
	s.Equal(ChatDeliveryDone, state)
	s.Zero(attempts)
	s.Require().NotNil(finishedAt)
	s.True(finishedAt.Equal(s.base.Add(time.Second)), "it left the queue as its answer started: %s", finishedAt)
}

// The message is named by its app and its id together: either alone is another message.
func (s *StoreSuite) TestAChatAnswerStartsOnlyForItsOwnAppAndMessage() {
	s.queuedChat("msg-one")

	otherApp, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257546, "msg-one", s.base)
	s.Require().NoError(err)
	otherMessage, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-two", s.base)
	s.Require().NoError(err)

	s.False(otherApp)
	s.False(otherMessage)
	state, _, finishedAt := s.chatRow("msg-one")
	s.Equal(ChatDeliveryQueued, state)
	s.Nil(finishedAt)
}

func (s *StoreSuite) TestAGivenUpChatMessagesAnswerDoesNotStart() {
	s.queuedChat("msg-one")
	_, err := s.store.FailChatDelivery(s.ctx, s.claimedChat(s.base, "msg-one"), s.base)
	s.Require().NoError(err)

	started, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base)

	s.Require().NoError(err)
	s.False(started)
	state, _, _ := s.chatRow("msg-one")
	s.Equal(ChatDeliveryFailed, state)
}

// A worker whose try failed after the answer started elsewhere, such as a peer whose reply was
// lost, neither retries nor gives up the message: it was answered.
func (s *StoreSuite) TestAChatMessageWhoseAnswerStartedIsNotRetriedOrGivenUp() {
	s.queuedChat("msg-one")
	claimed := s.claimedChat(s.base, "msg-one")
	_, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base)
	s.Require().NoError(err)

	retried, err := s.store.RetryChatDelivery(s.ctx, claimed, s.base.Add(time.Second))
	s.Require().NoError(err)
	failed, err := s.store.FailChatDelivery(s.ctx, claimed, s.base.Add(time.Second))
	s.Require().NoError(err)

	s.False(retried)
	s.False(failed)
	state, attempts, _ := s.chatRow("msg-one")
	s.Equal(ChatDeliveryDone, state)
	s.Zero(attempts)
}

// A worker whose lease ran out does not finish the attempt of the worker that took the message
// since, even once that one's answer started.
func (s *StoreSuite) TestAStartedChatMessageIsNotFinishedByAWorkerWhoseLeaseWasTakenOver() {
	s.queuedChat("msg-one")
	stale := s.claimedChat(s.base, "msg-one")
	current := s.claimedChat(s.base.Add(chatLease), "msg-one")
	_, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base.Add(chatLease))
	s.Require().NoError(err)

	staleFinished, err := s.store.FinishChatDelivery(s.ctx, stale, s.base.Add(chatLease))
	s.Require().NoError(err)
	currentFinished, err := s.store.FinishChatDelivery(s.ctx, current, s.base.Add(chatLease))
	s.Require().NoError(err)

	s.False(staleFinished)
	s.True(currentFinished)
}

// A message whose answer started and that nothing then took is queued again under the same
// lease: the worker holding it retries it, and the next try starts its answer.
func (s *StoreSuite) TestAReleasedChatAnswerIsRetriedAndStartsAgain() {
	s.queuedChat("msg-one")
	claimed := s.claimedChat(s.base, "msg-one")
	_, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base)
	s.Require().NoError(err)

	s.Require().NoError(s.store.ReleaseChatDeliveryAnswer(s.ctx, 1257545, "msg-one"))

	state, _, finishedAt := s.chatRow("msg-one")
	s.Equal(ChatDeliveryQueued, state)
	s.Nil(finishedAt)
	retried, err := s.store.RetryChatDelivery(s.ctx, claimed, s.base.Add(time.Second))
	s.Require().NoError(err)
	s.True(retried)
	s.claimedChat(s.base.Add(time.Second), "msg-one")
	started, err := s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base.Add(time.Second))
	s.Require().NoError(err)
	s.True(started)
}

// Only an answer that started is given back: a message given up, or of another app, stays as
// it is.
func (s *StoreSuite) TestReleasingAChatAnswerLeavesAGivenUpMessageAndAnotherAppsAlone() {
	s.queuedChat("msg-one")
	_, err := s.store.FailChatDelivery(s.ctx, s.claimedChat(s.base, "msg-one"), s.base)
	s.Require().NoError(err)
	s.queuedChat("msg-two")
	_, err = s.store.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-two", s.base)
	s.Require().NoError(err)

	s.Require().NoError(s.store.ReleaseChatDeliveryAnswer(s.ctx, 1257545, "msg-one"))
	s.Require().NoError(s.store.ReleaseChatDeliveryAnswer(s.ctx, 1257546, "msg-two"))
	s.Require().NoError(s.store.ReleaseChatDeliveryAnswer(s.ctx, 1257545, "msg-three"))

	failed, _, _ := s.chatRow("msg-one")
	s.Equal(ChatDeliveryFailed, failed)
	started, _, _ := s.chatRow("msg-two")
	s.Equal(ChatDeliveryDone, started)
}

// Routers starting one message's answer at the same moment start it once between them.
func (s *StoreSuite) TestRoutersStartingAChatAnswerAtOnceStartItOnce() {
	s.queuedChat("msg-one")
	routerA, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(routerA.Close()) })

	var started atomic.Int32
	var wg sync.WaitGroup
	errs := make(chan error, 20)
	for i := range 20 {
		router := []*Store{s.store, routerA}[i%2]
		wg.Go(func() {
			ok, err := router.StartChatDeliveryAnswer(s.ctx, 1257545, "msg-one", s.base)
			if err != nil {
				errs <- err
				return
			}
			if ok {
				started.Add(1)
			}
		})
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		s.Require().NoError(err)
	}
	s.Equal(int32(1), started.Load())
}
