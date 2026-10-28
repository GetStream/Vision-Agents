//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/webhook"
)

// EventForwardingSuite is a customer's Slack app's events going on to the customer's event
// destinations (T46): the fake provider plays Slack, delivering to
// POST /v1/connectors/events/slack_bot/{app id} signed with the app's signing secret; the
// router acts on the event as it does without destinations, acks, and the suite's forwarder
// sends the raw event to local TLS servers that stand for the customer's URLs.
type EventForwardingSuite struct {
	RouterSuite

	slack     *fakeprovider.Server
	app       store.ConnectorOAuthClient
	secret    string
	workspace string
	// logged is what the router logged, for the line a skipped message leaves.
	logged *lockedLog
}

func TestEventForwardingSuite(t *testing.T) {
	runSuite(t, new(EventForwardingSuite))
}

// SetupSuite registers hmac_header, oauth2_code and the real channel bridge, as
// SlackChannelSuite does, and an event forwarder.
func (s *EventForwardingSuite) SetupSuite() {
	s.database = "event_forwarding"
	verifier := hmacheader.New()
	code, err := oauth2code.New(oauth2code.Config{HTTP: http.DefaultClient})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{oauth2code.Name: code},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.channelProvider = func() string { return strings.TrimPrefix(s.slack.URL, "https://") }
	s.forwardHTTP = destinationClient()
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *EventForwardingSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.slack = fakeprovider.New(s.T(), fakeprovider.SlackChannel)
	s.workspace = s.slack.TeamID
	s.secret = "synthetic-signing-" + s.utils.uuid()
	s.app = s.providerApp(s.secret)
}

// A Block Kit button click is an interaction the router does not act on. It reaches the
// customer as Slack sent it, so Slack Bolt verifies it with the app's own signing secret.
func (s *EventForwardingSuite) TestABlockActionsReachesTheDestinationWithSlacksBodyAndHeadersAsTheyCame() {
	target := newDestination(s.T())
	created := s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.interaction(`{"type":"block_actions","team":{"id":"` + s.workspace + `"},"user":{"id":"U0000ALICE"},` +
		`"trigger_id":"synthetic.trigger","actions":[{"action_id":"approve","block_id":"b1","type":"button","value":"yes"}]}`)

	status, _, sent := s.slack.DeliverInteraction(s.eventsURL(), s.secret, body)

	s.Require().Equal(http.StatusOK, status)
	got := s.forwardedTo(target, 1)[0]
	s.Equal(body, got.body, "the raw body, byte for byte")
	for _, name := range []string{"Content-Type", "X-Slack-Signature", "X-Slack-Request-Timestamp"} {
		s.Equal(sent.Get(name), got.header.Get(name), name)
	}
	s.NoError(webhook.Verify(created.Secret, got.header, got.body, time.Now()), "signed with the destination's secret")
}

func (s *EventForwardingSuite) TestAForwardSignedForOneCustomerDoesNotVerifyWithAnothersKey() {
	mine := newDestination(s.T())
	created := s.createDestination("slack_bot", mine.URL, store.ForwardUnhandled)
	s.useApp(s.data.createApp())
	theirs := s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardUnhandled)

	s.deliver(s.reaction(), 0)

	got := s.forwardedTo(mine, 1)[0]
	s.NoError(webhook.Verify(created.Secret, got.header, got.body, time.Now()))
	s.ErrorIs(webhook.Verify(theirs.Secret, got.header, got.body, time.Now()), webhook.ErrBadSignature)
}

// A person's message an agent answers is the router's: a destination of unhandled events is
// not sent it, and one of every event is.
func (s *EventForwardingSuite) TestAMessageAnAgentAnswersGoesOnlyToADestinationOfEveryEvent() {
	s.answering(s.connectedBot())
	unhandled := newDestination(s.T())
	s.createDestination("slack_bot", unhandled.URL, store.ForwardUnhandled)
	every := newDestination(s.T())
	s.createDestination("slack_bot", every.URL, store.ForwardAll)
	body := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100")

	status, _ := s.deliver(body, 0)

	s.Equal(http.StatusOK, status)
	s.Equal(body, s.forwardedTo(every, 1)[0].body)
	s.Never(func() bool { return len(unhandled.requests()) > 0 }, dropped, 20*time.Millisecond)
	s.Require().Eventually(func() bool { return len(s.chat.Stored(s.threadChannel())) == 1 }, settleFor, 10*time.Millisecond,
		"the agent's thread channel still gets the message")
}

// An answer is counted before the bridge writes the message into its thread channel, after
// the ack. When that write fails no agent got the message, so a destination of unhandled
// events gets it then (AI-924), not neither of them.
func (s *EventForwardingSuite) TestAMessageWhoseThreadChannelWriteFailsGoesToADestinationOfUnhandledEvents() {
	s.answering(s.connectedBot())
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	// Stream refuses every write of the customer's: its app is read only, as app mode leaves
	// the deployment's app once the fallback is off.
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.readOnly[s.customerID()] = true })
	body := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100")

	status, _ := s.deliver(body, 0)

	s.Equal(http.StatusOK, status)
	s.Equal(body, s.forwardedTo(target, 1)[0].body)
	s.Empty(s.chat.Stored(s.threadChannel()), "the thread channel did not get it")
}

// AI-1001: a reply in a thread nobody linked yet is answered by no agent when it arrives, so a
// destination of unhandled events gets it then. When the mention Slack retries links the thread,
// the agent gets the reply as well; the forward is not taken back.
func (s *EventForwardingSuite) TestAReplyThatWaitedIsForwardedWhenItArrivesAndAnsweredOnceItsThreadIsLinked() {
	s.answering(s.connectedBot())
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	reply := s.reply("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100")

	s.deliver(reply, 0)
	s.Equal(reply, s.forwardedTo(target, 1)[0].body, "unhandled when it arrives")
	s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.000100"), 1)

	s.Require().Eventually(func() bool { return len(s.chat.Stored(s.threadChannel())) == 2 }, settleFor, 10*time.Millisecond,
		"the agent's thread channel gets the mention and the reply")
	s.Never(func() bool { return len(target.requests()) > 1 }, dropped, 20*time.Millisecond,
		"the mention an agent answers is not unhandled")
}

// AI-1001: a reply that waited for its thread's link reached the destinations of unhandled
// events when it arrived. When the mention that links the thread takes it and its write then
// fails, they already have it: the mention's delivery, which an agent answered, is not sent
// to them in its place.
func (s *EventForwardingSuite) TestAReplyThatWaitedAndIsNotWrittenDoesNotForwardTheMentionInItsPlace() {
	bot := s.connectedBot()
	s.answering(bot)
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	reply := s.reply("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100")
	s.deliver(reply, 0)
	s.Require().Equal(reply, s.forwardedTo(target, 1)[0].body)
	// The mention's first delivery linked the thread and claimed the mention, then failed before
	// it took the reply, so Slack's retry writes the reply alone.
	thread := store.ChannelThread{
		ChannelID: conversation.ThreadChannelPrefix + s.utils.uuid(), CustomerID: s.customerID(),
		ConnectorID: "slack_bot", ProviderUnitID: s.workspace, ThreadKey: "C0000CHAN:1759740000.000100",
		ConnectionID: bot, ThreadParts: map[string]string{"channel": "C0000CHAN", "thread_ts": "1759740000.000100"},
		StreamAppPK: s.app.StreamAppPK,
	}
	_, err := s.store.LinkChannelThread(context.Background(), &thread)
	s.Require().NoError(err)
	_, err = s.store.ClaimChannelThreadMessage(context.Background(), thread.ChannelID, store.ClaimInbound, "1759740000.000100")
	s.Require().NoError(err)
	// Stream refuses every write of the customer's, as in the test above.
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.readOnly[s.customerID()] = true })

	status, _ := s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.000100"), 1)

	s.Equal(http.StatusOK, status)
	s.Never(func() bool { return len(target.requests()) > 1 }, dropped, 20*time.Millisecond,
		"the destination got the mention's delivery as unhandled")
	s.Empty(s.chat.Stored(thread.ChannelID), "the reply was not written")
}

// AI-1001: only the messages a take brought go without the delivery's forward. In a delivery
// of original messages (WhatsApp batches them), each one never written carries it, and the
// destinations key it by the delivery: two given up are one forward. A message the provider
// retried was taken before and queues nothing.
func (s *EventForwardingSuite) TestABatchsMessagesThatAreNotWrittenAreForwardedOnce() {
	s.answering(s.connectedBot())
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	first, second, third := s.inbound("1759740000.000100"), s.inbound("1759740000.000300"), s.inbound("1759740000.000500")
	_, err := s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{first}, nil)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return len(s.chat.Stored(s.threadChannel())) == 1 }, settleFor, 10*time.Millisecond)
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.readOnly[s.customerID()] = true })
	delivery := eventforward.Event{CustomerID: s.customerID(), ConnectorID: "slack_bot", ID: "Ev" + s.utils.uuid(),
		Headers: map[string]string{"Content-Type": "application/json", "X-Slack-Signature": "v0=synthetic"}}

	answered, err := s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{first, second, third}, &delivery)

	s.Require().NoError(err)
	s.True(answered)
	got := s.forwardedTo(target, 1)[0]
	s.Contains([][]byte{second.Raw, third.Raw}, got.body, "the body of a message given up")
	s.Equal("application/json", got.header.Get("Content-Type"), "the delivery's headers")
	s.Equal("v0=synthetic", got.header.Get("X-Slack-Signature"), "a signature that does not age")
	s.Never(func() bool { return len(target.requests()) > 1 }, dropped, 20*time.Millisecond, "one forward for the delivery")
	var ageing int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_thread_inbound WHERE customer_id = ? AND forward ? 'headers_until'", s.customerID()).Scan(&ageing))
	s.Zero(ageing, "a delivery whose headers do not age is stored without headers_until")
}

// A message the store fails to queue releases its claim, so the provider's retry of the delivery
// takes it again. Postgres's jsonb refuses \u0000, and only the queue stores the delivery's
// headers, so a NUL in one fails the queue and nothing before it.
func (s *EventForwardingSuite) TestAMessageTheStoreFailsToQueueIsNotClaimed() {
	s.answering(s.connectedBot())
	delivery := eventforward.Event{CustomerID: s.customerID(), ConnectorID: "slack_bot", Headers: map[string]string{"X-Synthetic": "a\x00b"}}

	_, err := s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, &delivery)

	s.Require().Error(err)
	var claims int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_thread_messages WHERE channel_id = ? AND kind = ?", s.threadChannel(), store.ClaimInbound).Scan(&claims))
	s.Zero(claims)
}

// The mention that links a thread takes the replies that waited for it. When the mention then
// fails to queue, they wait again, so Slack's retry of the mention brings them as well.
func (s *EventForwardingSuite) TestAReplyThatWaitedIsWrittenWhenItsMentionFailsToQueueAndIsRetried() {
	s.answering(s.connectedBot())
	s.deliver(s.reply("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	// A NUL fails the queue alone, as in the test above.
	delivery := eventforward.Event{CustomerID: s.customerID(), ConnectorID: "slack_bot", Headers: map[string]string{"X-Synthetic": "a\x00b"}}
	_, err := s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, &delivery)
	s.Require().Error(err)

	_, err = s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)

	s.Require().NoError(err)
	channel := s.threadChannel()
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) == 2 }, settleFor, 10*time.Millisecond,
		"the mention and the reply that waited are written, not %d messages", len(s.chat.Stored(channel)))
	s.Equal([]any{channel + "-1759740000.000100", channel + "-1759740000.000200"},
		[]any{s.chat.Stored(channel)[0]["id"], s.chat.Stored(channel)[1]["id"]})
}

// A reply that waited and itself fails to queue waits again too, for the mention's retry.
func (s *EventForwardingSuite) TestAReplyThatWaitedAndFailsToQueueIsWrittenWhenItsMentionIsRetried() {
	s.answering(s.connectedBot())
	s.deliver(s.reply("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	drop := s.trigger("failing_waited_queue", "channel_thread_inbound", "INSERT",
		"NEW.customer_id = '"+s.customerID()+"' AND NEW.provider_message_id = '1759740000.000200'", "RAISE EXCEPTION 'synthetic queue failure';")
	_, err := s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)
	s.Require().ErrorContains(err, "synthetic queue failure")
	drop()

	_, err = s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)

	s.Require().NoError(err)
	channel := s.threadChannel()
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) == 2 }, settleFor, 10*time.Millisecond,
		"the mention and the reply that waited are written, not %d messages", len(s.chat.Stored(channel)))
}

// A provider that hangs up while its message is queued cancels the request's context
// (net/http). The message is queued all the same, so the provider's retry, dropped as one, loses
// nothing.
func (s *EventForwardingSuite) TestAMessageWhoseRequestEndsWhileItIsQueuedIsWritten() {
	s.answering(s.connectedBot())
	s.trigger("slow_inbound_queue", "channel_thread_inbound", "INSERT", "NEW.customer_id = '"+s.customerID()+"'", "PERFORM pg_sleep(1);")

	err := s.deliverHangingUp(s.inbound("1759740000.000100"))

	s.Require().NoError(err)
	_, err = s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)
	s.Require().NoError(err)
	channel := s.threadChannel()
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) == 1 }, settleFor, 10*time.Millisecond)
}

// A message the store fails to queue after its provider hung up still releases its claim, so the
// provider's retry queues it.
func (s *EventForwardingSuite) TestAMessageWhoseQueueFailsAfterItsRequestEndedIsNotClaimed() {
	s.answering(s.connectedBot())
	drop := s.trigger("failing_inbound_queue", "channel_thread_inbound", "INSERT", "NEW.customer_id = '"+s.customerID()+"'",
		"PERFORM pg_sleep(1); RAISE EXCEPTION 'synthetic queue failure';")

	err := s.deliverHangingUp(s.inbound("1759740000.000100"))

	s.Require().ErrorContains(err, "synthetic queue failure")
	s.Zero(s.inboundClaims())
	drop()
	_, err = s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)
	s.Require().NoError(err)
	channel := s.threadChannel()
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) == 1 }, settleFor, 10*time.Millisecond)
}

// A reply that waited and is put back after its mention failed to queue is put back off the
// request, so a provider that hung up does not lose it, and only the reply is put back.
func (s *EventForwardingSuite) TestAReplyThatWaitedIsPutBackWhenItsMentionFailsToQueueAfterTheProviderHungUp() {
	s.answering(s.connectedBot())
	s.deliver(s.reply("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	s.trigger("failing_inbound_queue", "channel_thread_inbound", "INSERT", "NEW.customer_id = '"+s.customerID()+"'",
		"PERFORM pg_sleep(1); RAISE EXCEPTION 'synthetic queue failure';")

	err := s.deliverHangingUp(s.inbound("1759740000.000100"))

	s.Require().ErrorContains(err, "synthetic queue failure")
	var waiting int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_thread_waiting WHERE customer_id = ?", s.customerID()).Scan(&waiting))
	s.Equal(1, waiting, "the reply waits again, and only the reply")
}

// A queue insert that hangs is cut short by its bound, and the claim is released on a bound of
// its own, so the provider gets its error in time and its retry queues the message.
func (s *EventForwardingSuite) TestAMessageWhoseQueueHangsIsAnsweredInTimeAndNotClaimed() {
	s.answering(s.connectedBot())
	drop := s.trigger("hanging_inbound_queue", "channel_thread_inbound", "INSERT", "NEW.customer_id = '"+s.customerID()+"'", "PERFORM pg_sleep(3);")
	bounded := s.boundedRouter(200 * time.Millisecond)

	started := time.Now()
	_, err := bounded.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)

	s.Require().Error(err)
	s.Less(time.Since(started), 2*time.Second, "the insert was not cut short")
	s.Zero(s.inboundClaims())
	drop()
	_, err = s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)
	s.Require().NoError(err)
	channel := s.threadChannel()
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) == 1 }, settleFor, 10*time.Millisecond)
}

// A release that hangs after the queue failed is cut short by its bound too: the provider gets
// its error in time. The claim stays then, until the claim and the queue are one transaction
// (AI-2150).
func (s *EventForwardingSuite) TestAMessageWhoseReleaseHangsIsAnsweredInTime() {
	s.answering(s.connectedBot())
	s.trigger("hanging_inbound_release", "channel_thread_messages", "DELETE",
		"OLD.channel_id IN (SELECT channel_id FROM channel_threads WHERE customer_id = '"+s.customerID()+"')", "PERFORM pg_sleep(3);")
	bounded := s.boundedRouter(200 * time.Millisecond)
	delivery := eventforward.Event{CustomerID: s.customerID(), ConnectorID: "slack_bot", Headers: map[string]string{"X-Synthetic": "a\x00b"}}

	started := time.Now()
	_, err := bounded.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, &delivery)

	s.Require().ErrorIs(err, os.ErrDeadlineExceeded, "the release's failure is answered")
	s.Less(time.Since(started), 2*time.Second, "the release was not cut short")
}

// deliverHangingUp delivers message on a request whose provider hangs up once the message is
// claimed, as net/http cancels the request's context, and answers what Deliver did.
func (s *EventForwardingSuite) deliverHangingUp(message core.InboundMessage) error {
	ctx, hangUp := context.WithCancel(context.Background())
	defer hangUp()
	done := make(chan error, 1)
	go func() {
		_, err := s.bridge.Deliver(ctx, s.app, []core.InboundMessage{message}, nil)
		done <- err
	}()
	s.Require().Eventually(func() bool { return s.inboundClaims() == 1 }, settleFor, 5*time.Millisecond)
	time.Sleep(300 * time.Millisecond) // the insert is in its trigger
	hangUp()
	return <-done
}

// inboundClaims is how many inbound messages the bridge claimed in the test customer's thread
// channels.
func (s *EventForwardingSuite) inboundClaims() int {
	var claims int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_thread_messages m JOIN channel_threads t USING (channel_id) WHERE t.customer_id = ? AND m.kind = ?",
		s.customerID(), store.ClaimInbound).Scan(&claims))
	return claims
}

// trigger runs body before each op on table for a row where when holds, until the test ends or
// drop. The table is in the suite's own database (database), and when names the test's
// customer, so no other test's rows meet it.
func (s *EventForwardingSuite) trigger(name, table, op, when, body string) (drop func()) {
	db := s.store.DB()
	_, err := db.ExecContext(context.Background(), "CREATE OR REPLACE FUNCTION "+name+"() RETURNS trigger AS $$ BEGIN IF "+when+" THEN "+body+
		" END IF; RETURN COALESCE(NEW, OLD); END $$ LANGUAGE plpgsql")
	s.Require().NoError(err)
	_, err = db.ExecContext(context.Background(), "DROP TRIGGER IF EXISTS "+name+" ON "+table)
	s.Require().NoError(err)
	_, err = db.ExecContext(context.Background(), "CREATE TRIGGER "+name+" BEFORE "+op+" ON "+table+" FOR EACH ROW EXECUTE FUNCTION "+name+"()")
	s.Require().NoError(err)
	var once sync.Once
	drop = func() {
		once.Do(func() {
			_, err := db.ExecContext(context.Background(), "DROP TRIGGER IF EXISTS "+name+" ON "+table)
			s.Require().NoError(err)
			_, err = db.ExecContext(context.Background(), "DROP FUNCTION IF EXISTS "+name+"()")
			s.Require().NoError(err)
		})
	}
	s.T().Cleanup(drop)
	return drop
}

// boundedRouter is another router's bridge whose store writes off the request are bounded by
// timeout, its worker not started.
func (s *EventForwardingSuite) boundedRouter(timeout time.Duration) *channelbridge.Bridge {
	options := s.channelBridgeOptions(slog.New(slog.DiscardHandler))
	options.QueueTimeout = timeout
	bridge, err := channelbridge.New(options)
	s.Require().NoError(err)
	return bridge
}

// One thread's messages are given up minutes apart, after the first one's forward was sent and
// its row deleted: the second given up still forwards nothing.
func (s *EventForwardingSuite) TestAMessageOfADeliveryGivenUpAfterItsForwardWasSentForwardsNothing() {
	s.answering(s.connectedBot())
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.readOnly[s.customerID()] = true })
	delivery := eventforward.Event{CustomerID: s.customerID(), ConnectorID: "slack_bot", ID: "Ev" + s.utils.uuid()}
	_, err := s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, &delivery)
	s.Require().NoError(err)
	s.forwardedTo(target, 1)
	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)

	_, err = s.bridge.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000300")}, &delivery)

	s.Require().NoError(err)
	s.Require().Eventually(func() bool {
		var failed int
		s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
			"SELECT count(*) FROM channel_thread_inbound WHERE customer_id = ? AND state = 'failed'", s.customerID()).Scan(&failed))
		return failed == 2
	}, settleFor, 10*time.Millisecond, "the second message was not given up")
	s.Never(func() bool { return len(target.requests()) > 1 }, dropped, 20*time.Millisecond)
}

// A router that stops right after it answered the provider leaves the message queued, and
// another router's worker writes it, under the provider's id for it.
func (s *EventForwardingSuite) TestAMessageARouterQueuedBeforeItStoppedIsWrittenByAnother() {
	s.answering(s.connectedBot())
	stopped := s.anotherRouter()

	answered, err := stopped.Deliver(context.Background(), s.app, []core.InboundMessage{s.inbound("1759740000.000100")}, nil)
	stopped.Close()

	s.Require().NoError(err)
	s.True(answered)
	next := s.anotherRouter()
	next.Start()
	defer next.Close()
	channel := s.threadChannel()
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) == 1 }, settleFor, 10*time.Millisecond)
	s.Equal(channel+"-1759740000.000100", s.chat.Stored(channel)[0]["id"])
}

// The delivery a message is forwarded with when it is never written is stored with it, so the
// router that gives it up forwards it, not only the one that took it.
func (s *EventForwardingSuite) TestAMessageARouterQueuedBeforeItStoppedIsForwardedByTheOneThatGivesItUp() {
	s.answering(s.connectedBot())
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.readOnly[s.customerID()] = true })
	message := s.inbound("1759740000.000100")
	stopped := s.anotherRouter()
	delivery := eventforward.Event{CustomerID: s.customerID(), ConnectorID: "slack_bot", ID: "Ev" + s.utils.uuid(),
		Headers:      map[string]string{"Content-Type": "application/json", "X-Slack-Signature": "v0=synthetic"},
		HeadersUntil: time.Now().Add(-time.Minute)}

	_, err := stopped.Deliver(context.Background(), s.app, []core.InboundMessage{message}, &delivery)
	stopped.Close()

	s.Require().NoError(err)
	next := s.anotherRouter()
	next.Start()
	defer next.Close()
	got := s.forwardedTo(target, 1)[0]
	s.Equal(message.Raw, got.body)
	s.Equal("application/json", got.header.Get("Content-Type"))
	s.Empty(got.header.Get("X-Slack-Signature"), "a signature past its window is not sent")
}

// anotherRouter is the bridge of another router on the suite's database and Stream Chat, its
// worker not started.
func (s *EventForwardingSuite) anotherRouter() *channelbridge.Bridge {
	bridge, err := channelbridge.New(s.channelBridgeOptions(slog.New(slog.DiscardHandler)))
	s.Require().NoError(err)
	return bridge
}

// inbound is a message by U0000ALICE that mentions the bot in C0000CHAN, as the verifier hands
// it over.
func (s *EventForwardingSuite) inbound(ts string) core.InboundMessage {
	return core.InboundMessage{ConnectorID: "slack_bot", ProviderUnitID: s.workspace, ThreadKey: "C0000CHAN:" + ts,
		AuthorID: "U0000ALICE", Text: "hi", ProviderMessageID: ts, Raw: s.message("U0000ALICE", "hi", ts)}
}

// Slack's event_id is «A unique identifier for this specific event»
// (https://docs.slack.dev/apis/events-api/), and slack_bot.yaml keys a forward by it: two
// deliveries of one event are one forward even when their bodies differ.
func (s *EventForwardingSuite) TestTwoDeliveriesOfOneSlackEventAreOneForward() {
	target := newDestination(s.T())
	release := target.holding()
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	first := s.reaction()
	again := []byte(strings.Replace(string(first), `"type":"event_callback"`, `"type":"event_callback","is_ext_shared_channel":false`, 1))
	s.Require().NotEqual(first, again)

	s.deliver(first, 0)
	s.Require().Eventually(func() bool { return len(target.requests()) == 1 }, settleFor, 10*time.Millisecond)
	s.deliver(again, 1)
	release()

	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
	s.Len(target.requests(), 1)
}

// Mode C: the customer runs its own agent, so no agent config binds the bot connection, and a
// message is one the router answers in no way.
func (s *EventForwardingSuite) TestAMessageNoAgentAnswersIsUnhandled() {
	s.connectedBot()
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100")

	s.deliver(body, 0)

	s.Equal(body, s.forwardedTo(target, 1)[0].body)
}

// TestAMessageAnAppPostedIsSkippedAndLoggedAtInfo (AI-1053 F67): the manifest's skip_if_present
// rule leaves a message out, and the line that says so names the connector, the provider app
// and the rule, never the text.
func (s *EventForwardingSuite) TestAMessageAnAppPostedIsSkippedAndLoggedAtInfo() {
	s.connectedBot()
	raw, err := json.Marshal(map[string]string{"type": "message", "channel": "C0000CHAN", "user": "U0000ALICE",
		"text": botMention + "synthetic-text-never-logged", "ts": "1759740000.000100", "channel_type": "channel", "bot_id": "B0000APP"})
	s.Require().NoError(err)

	status, _ := s.deliver(s.event(string(raw)), 0)

	s.Equal(http.StatusOK, status)
	var line string
	for _, l := range strings.Split(s.logged.String(), "\n") {
		if strings.Contains(l, "skipped a connector event's message") && strings.Contains(l, s.app.ProviderAppID) {
			line = l
		}
	}
	s.Contains(line, "level=INFO")
	s.Contains(line, "connector=slack_bot")
	s.Contains(line, "$.event.bot_id")
	s.NotContains(s.logged.String(), "synthetic-text-never-logged")
}

// https://docs.slack.dev/reference/events/url_verification: the handshake proves the router's
// URL to Slack; the customer's URL is not Slack's to call.
func (s *EventForwardingSuite) TestAURLVerificationIsNotForwarded() {
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardAll)

	status, answer := s.deliver([]byte(`{"token":"synthetic","challenge":"synthetic-challenge-value","type":"url_verification"}`), 0)

	s.Equal(http.StatusOK, status)
	s.Equal("synthetic-challenge-value", answer)
	s.Never(func() bool { return len(target.requests()) > 0 }, dropped, 20*time.Millisecond)
}

// Slack waits three seconds for its answer (https://docs.slack.dev/apis/events-api/); a
// customer's URL may take fifteen.
func (s *EventForwardingSuite) TestTheAckToSlackDoesNotWaitForTheDestination() {
	target := newDestination(s.T())
	release := target.holding()
	defer release()
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)

	started := time.Now()
	status, _ := s.deliver(s.reaction(), 0)

	s.Equal(http.StatusOK, status)
	s.Less(time.Since(started), time.Second, "answered while the destination still holds the forward")
	s.Require().Eventually(func() bool { return len(target.requests()) == 1 }, settleFor, 10*time.Millisecond,
		"the forward is on its way")
}

func (s *EventForwardingSuite) TestAForwardTheDestinationAnswers503IsSentAgainAndTakenOnce() {
	target := newDestination(s.T())
	target.answer(http.StatusServiceUnavailable)
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.reaction()

	s.deliver(body, 0)

	sent := s.forwardedTo(target, 2)
	s.Equal(body, sent[1].body)
	s.Equal(sent[0].header.Get("webhook-id"), sent[1].header.Get("webhook-id"), "one webhook-id on every attempt")
	s.Never(func() bool { return len(target.requests()) > 2 }, dropped, 20*time.Millisecond)
	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
}

func (s *EventForwardingSuite) TestAForwardTheDestinationAnswers400IsNotSentAgain() {
	target := newDestination(s.T())
	target.answer(http.StatusBadRequest)
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)

	s.deliver(s.reaction(), 0)

	s.forwardedTo(target, 1)
	s.Never(func() bool { return len(target.requests()) > 1 }, dropped, 20*time.Millisecond)
	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
}

// Slack retries a delivery it got no answer to in time, with X-Slack-Retry-Num
// (https://docs.slack.dev/apis/events-api/, «Retries»); the forward of the first one still
// pending is not queued twice.
func (s *EventForwardingSuite) TestADeliverySlackRetriesWhileItsForwardIsPendingIsQueuedOnce() {
	target := newDestination(s.T())
	release := target.holding()
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.reaction()
	s.deliver(body, 0)
	s.Require().Eventually(func() bool { return len(target.requests()) == 1 }, settleFor, 10*time.Millisecond)

	s.deliver(body, 1)
	release()

	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
	s.Len(target.requests(), 1)
}

// The Standard Webhooks rotation: «the webhook is signed both using the current key, and using
// an old key (for a set period of time)» (https://www.standardwebhooks.com/).
func (s *EventForwardingSuite) TestAfterARotationAForwardVerifiesWithTheOldSecretAndTheNew() {
	target := newDestination(s.T())
	created := s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	var rotated ConnectorEventDestinationSecret
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		destinationPath("slack_bot", created.Destination.ID)+"/rotate-secret", nil, &rotated))

	s.deliver(s.reaction(), 0)

	got := s.forwardedTo(target, 1)[0]
	s.NoError(webhook.Verify(created.Secret, got.header, got.body, time.Now()), "the old secret")
	s.NoError(webhook.Verify(rotated.Secret, got.header, got.body, time.Now()), "the new secret")
}

// forwardedTo waits until a destination took count requests, and returns them.
func (s *EventForwardingSuite) forwardedTo(target *destination, count int) []forwarded {
	s.Require().Eventually(func() bool { return len(target.requests()) >= count }, settleFor, 10*time.Millisecond,
		"the destination took %d forwards, not %d", len(target.requests()), count)
	return target.requests()
}

// pendingForwards is how many forwards to the test app's destinations are not done with.
func (s *EventForwardingSuite) pendingForwards() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		`SELECT count(*) FROM connector_event_deliveries d JOIN connector_event_destinations t ON t.id = d.destination_id
		 WHERE t.customer_id = ?`, s.customerID()).Scan(&count))
	return count
}

func (s *EventForwardingSuite) eventsURL() string {
	return s.server.URL + providerAppEventsPath + "slack_bot/" + s.app.ProviderAppID
}

// deliver posts a Slack event to the test app's events URL, signed with its secret.
func (s *EventForwardingSuite) deliver(body []byte, retry int) (int, string) {
	return s.slack.Deliver(s.eventsURL(), s.secret, body, retry)
}

// event is a Slack event_callback in the test's workspace (https://docs.slack.dev/apis/events-api/),
// visible to the app's bot install.
func (s *EventForwardingSuite) event(inner string) []byte {
	return []byte(`{"token":"synthetic","team_id":"` + s.workspace + `","api_app_id":"` + s.app.ProviderAppID +
		`","event":` + inner + `,"type":"event_callback","event_id":"Ev` + strings.ReplaceAll(s.utils.uuid(), "-", "") +
		`","event_time":` + fmt.Sprint(time.Now().Unix()) + `,"authorizations":[{"team_id":"` + s.workspace +
		`","user_id":"U0000BOT","is_bot":true}]}`)
}

// message is a message.channels event by user in C0000CHAN
// (https://docs.slack.dev/reference/events/message.channels) that mentions the bot, which
// starts a thread (AI-989).
func (s *EventForwardingSuite) message(user, text, ts string) []byte {
	raw, err := json.Marshal(map[string]string{"type": "message", "channel": "C0000CHAN", "user": user, "text": botMention + text, "ts": ts, "channel_type": "channel"})
	s.Require().NoError(err)
	return s.event(string(raw))
}

// reply is a reply by user in the thread of threadTS in C0000CHAN, which does not mention the
// bot (https://docs.slack.dev/reference/events/message.channels, thread_ts).
func (s *EventForwardingSuite) reply(user, text, ts, threadTS string) []byte {
	raw, err := json.Marshal(map[string]string{"type": "message", "channel": "C0000CHAN", "user": user, "text": text, "ts": ts, "thread_ts": threadTS, "channel_type": "channel"})
	s.Require().NoError(err)
	return s.event(string(raw))
}

// reaction is a reaction_added event (https://docs.slack.dev/reference/events/reaction_added),
// which the router reads nothing from.
func (s *EventForwardingSuite) reaction() []byte {
	return s.event(`{"type":"reaction_added","user":"U0000ALICE","reaction":"thumbsup","item_user":"U0000BOT",` +
		`"item":{"type":"message","channel":"C0000CHAN","ts":"1759740000.000100"},"event_ts":"1759740001.000200"}`)
}

// interaction is the form body Slack posts an interaction payload in
// (https://docs.slack.dev/interactivity/handling-user-interaction).
func (s *EventForwardingSuite) interaction(payload string) []byte {
	return []byte(url.Values{"payload": {payload}}.Encode())
}

// providerApp is a Slack app of the test's customer with signingSecret sealed for it, as
// SlackChannelSuite.providerApp writes one.
func (s *EventForwardingSuite) providerApp(signingSecret string) store.ConnectorOAuthClient {
	record := &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: "slack_bot", Registration: core.ClientCustomer,
		ClientID: "synthetic-client", AuthMethod: core.AuthClientSecretPost,
		ProviderAppID: "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", "")),
	}
	var err error
	record.SecretSealed, err = s.sealer.SealWithAAD("synthetic-client-secret", oauthClientAAD(record.CustomerID, "slack_bot"))
	s.Require().NoError(err)
	record.KEKVersion = s.sealer.CurrentVersion()
	record.SigningSecretSealed, err = s.sealer.SealWithAAD(signingSecret, providerAppAAD(record.CustomerID, "slack_bot", record.ProviderAppID))
	s.Require().NoError(err)
	record.SigningKEKVersion = s.sealer.CurrentVersion()
	_, err = s.store.PutConnectorOAuthClient(context.Background(), record)
	s.Require().NoError(err)
	return *record
}

// connectedBot is the app's slack_bot connection of the test's workspace, connected with a bot
// token, as SlackChannelSuite.connectedBot makes one.
func (s *EventForwardingSuite) connectedBot() string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned("slack_bot"), &created))
	ref := core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: created.ID}
	payload, err := json.Marshal(map[string]string{"access_token": s.slack.InstallBot(), "token_type": "bot"})
	s.Require().NoError(err)
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	s.Require().NoError(credentials.Update(context.Background(), ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = core.StoredCredentials{Scheme: oauth2code.Name, Version: 1, Payload: payload}
		state.Status = store.ConnectionConnected
		state.AccountID = s.workspace
		state.Metadata = map[string]string{"team_id": s.workspace, "bot_user_id": "U0000BOT"}
		state.ConnectedAt = time.Now().UTC()
		return true, nil
	}))
	return created.ID
}

// answering is an agent config of the test's app that binds the bot connection as fixed.
func (s *EventForwardingSuite) answering(connectionID string) {
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "slack-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		Connectors: []store.ConnectorBinding{{
			Name: "slack", ConnectorID: "slack_bot",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: connectionID},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
}

// threadChannel is the test customer's one thread channel, once the bridge linked it.
func (s *EventForwardingSuite) threadChannel() string {
	var channel string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT channel_id FROM channel_threads WHERE customer_id = ?", s.customerID()).Scan(&channel))
	return channel
}
